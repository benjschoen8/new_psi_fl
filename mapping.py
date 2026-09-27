"""Relation-table strategies. No common interface requires a server."""
from contracts import MappingInputs, RelationTable


class ByClassMapping:
    """Oracle: explicit local-index → semantic-class metadata, including permutations.

    Names are case sensitive: EMNIST 'A' and 'a' are different classes.
    Cross-dataset synonyms must be canonicalized explicitly by the caller.
    """
    def __init__(self, label_spaces):
        self.label_spaces = {key: tuple(names) for key, names in label_spaces.items()}
        if not self.label_spaces or any(not names for names in self.label_spaces.values()):
            raise ValueError('Ground truth requires nonempty label spaces')

    def __call__(self, inputs=None) -> RelationTable:
        names = sorted({name for labels in self.label_spaces.values() for name in labels})
        ids = {name: i for i, name in enumerate(names)}
        return {key: {i: ids[name] for i, name in enumerate(labels)}
                for key, labels in self.label_spaces.items()}


def validate_mapping(table, num_classes):
    if set(table) != set(num_classes):
        raise ValueError('Mapping must cover exactly the active groups')
    for key, count in num_classes.items():
        if set(table[key]) != set(range(count)):
            raise ValueError(f'Mapping must cover every local label in {key}')
        if any(not isinstance(gid, int) or gid < 0 for gid in table[key].values()):
            raise ValueError('Mapping global IDs must be nonnegative integers')
    # Legacy register_mapping can leave gaps when two existing sets merge.
    ids = sorted({gid for labels in table.values() for gid in labels.values()})
    dense = {gid: i for i, gid in enumerate(ids)}
    return {key: {label: dense[gid] for label, gid in labels.items()}
            for key, labels in table.items()}


class ImageBiMapping:
    """Baseline image-bi entropy filtering + bidirectional cycle consistency."""
    def __init__(self, logger, device='cpu', noise_dim=128, entropy_ratio=.25,
                 use_new_entropy=True, samples=32):
        if noise_dim < 1 or samples < 1 or entropy_ratio < 0:
            raise ValueError('Invalid image mapping sampling/entropy parameters')
        self.logger, self.device, self.noise_dim = logger, device, noise_dim
        self.entropy_ratio, self.use_new_entropy, self.samples = entropy_ratio, use_new_entropy, samples

    def __call__(self, inputs: MappingInputs) -> RelationTable:
        import torch
        from label_mapping_utils import label_mapping
        cache = {}

        def images(group, label):
            key = (group, label)
            if key not in cache:
                generator = inputs.generators[group]
                generator.eval()
                z = torch.randn(self.samples, self.noise_dim, device=self.device)
                y = torch.full((self.samples,), label, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    cache[key] = generator(z, y)
            return cache[key]

        # The learned strategy sees opaque label tokens, never true semantic names.
        spaces = {key: [str(i) for i in range(count)] for key, count in inputs.num_classes.items()}
        for models in inputs.classifiers.values():
            for model in models:
                model.to(self.device).eval()
        return validate_mapping(label_mapping(
            get_images_func=images, dataset_ids=list(spaces),
            clients_dict=inputs.classifiers, label_space_meta=spaces,
            entropy_ratio=self.entropy_ratio, use_new_entropy_method=self.use_new_entropy,
            logger=self.logger,
        ), inputs.num_classes)



def _group_map(table, num_classes):
    from rt_protocol import to_group_map
    return validate_mapping(to_group_map(table, {g: g for g in num_classes}), num_classes)


def _descriptions(label_names, langs):
    """Each owner describes its OWN classes in its own language; nothing is shared."""
    from rt_descriptions import describe
    if label_names is None:
        raise ValueError('PSI mapping requires owner label names in MappingInputs')
    return {g: [describe(None, name, langs[k % len(langs)]) for name in label_names[g]]
            for k, g in enumerate(sorted(label_names))}


class PSITrivialCircuit:
    """Exact-match PSI on owner-written class descriptions (ideal functionality).

    Owners reveal only salted hashes of their descriptions; equal hashes become
    edges, merged by rt_protocol.global_table. No images, no classifiers.
    Different languages or wordings never match: the baseline fuzzy PSI fixes.
    """
    needs_classifiers = False

    def __init__(self, langs=('en',), salt=b'rt-session'):
        if not langs:
            raise ValueError('At least one description language is required')
        self.langs, self.salt = tuple(langs), salt

    def __call__(self, inputs: MappingInputs) -> RelationTable:
        import hashlib
        from rt_protocol import global_table
        # ponytail: plaintext hash comparison = ideal PSI output; swap in a real PSI if cost is measured
        tags = {g: [hashlib.sha256(self.salt + d.strip().lower().encode()).digest() for d in ds]
                for g, ds in _descriptions(inputs.label_names, self.langs).items()}
        groups = sorted(tags)
        edges = [(i, a, j, b, 1) for x, i in enumerate(groups) for j in groups[x + 1:]
                 for a, ha in enumerate(tags[i]) for b, hb in enumerate(tags[j]) if ha == hb]
        table = global_table(edges, {g: list(range(len(tags[g]))) for g in groups})
        return _group_map(table, inputs.num_classes)


class FuzzyPSICircuit:
    """rt_protocol LabeledFuzzyPSI between group owners (method filter or affscan).

    Owner-side signals: descriptions of its own classes (own language) and
    prototypes of its generator's images in a public encoder, profiled against
    public probes. Only PSI ranks leave an owner; classifiers are not used.
    """
    needs_classifiers = False

    def __init__(self, method='filter', psi='plain', langs=('en',), samples=32, noise_dim=128,
                 encoder=None, probes=None, n_probes=256, device='cpu', seed=0, logger=None):
        if method not in ('filter', 'affscan'):
            raise ValueError('FuzzyPSICircuit supports method filter or affscan')
        if psi not in ('plain', 'he') or samples < 1 or n_probes < 2 or not langs:
            raise ValueError('Invalid fuzzy PSI parameters')
        # ponytail: pixel space + seeded Gaussian probes (random projection of prototypes);
        # pass a pretrained encoder and real public probe images for semantic affinity
        self.encoder = encoder or (lambda x: x.flatten(1))
        self.method, self.psi, self.langs, self.samples = method, psi, tuple(langs), samples
        self.noise_dim, self.probes, self.n_probes = noise_dim, probes, n_probes
        self.device, self.seed, self.logger = device, seed, logger

    def __call__(self, inputs: MappingInputs) -> RelationTable:
        import numpy as np
        import torch
        from rt_protocol import run_rt_protocol
        descriptions = _descriptions(inputs.label_names, self.langs)
        clients, shape = {}, None
        for g, count in inputs.num_classes.items():
            generator, summ = inputs.generators[g].to(self.device).eval(), {}
            for a in range(count):
                z = torch.randn(self.samples, self.noise_dim, device=self.device)
                y = torch.full((self.samples,), a, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    images = generator(z, y)
                    summ[a] = self.encoder(images).mean(0).cpu().double().numpy()
                shape = images.shape[1:]
            clients[g] = dict(summ=summ, names=dict(enumerate(descriptions[g])),
                              count={a: self.samples for a in range(count)})
        probes = self.probes
        if probes is None:
            probes = torch.randn((self.n_probes, *shape),
                                 generator=torch.Generator().manual_seed(self.seed))
        with torch.no_grad():
            P = self.encoder(probes.to(self.device)).cpu().double().numpy()
        log = self.logger.log if self.logger else (lambda *_: None)
        table, _, _ = run_rt_protocol(clients, P - P.mean(0), method=self.method, psi=self.psi,
                                      min_samples=1, seed=self.seed, log=log)
        return _group_map(table, inputs.num_classes)


def _release_keys(strategy, table):
    """Alignment and key release as one circuit (ideal functionality).

    Each distinct global label gets a random slot in [0, max_labels) (secfl.keydist.assign_slots);
    slots, not dense ids, index keys, uploads and broadcasts, so a client learns only the
    public bound max_labels, never how many labels exist. Inside the circuit,
    b = [client holds the label] selects the client's output: K_slot if b else fresh random
    (release_keys_in_circuit): no choice bit to flip, and the Aggregator never sees b.
    Sets strategy.slots ({global id: slot}), strategy.keys ({slot: K}, every slot has a key
    so padded broadcasts look alike) and strategy.client_keys ({group: {slot: K}} held only).
    """
    from secfl.keydist import KeyDealer, assign_slots, release_keys_in_circuit
    gids = sorted({g for labels in table.values() for g in labels.values()})
    strategy.slots = assign_slots(gids, strategy.max_labels)
    dealer = KeyDealer(range(strategy.max_labels))
    strategy.keys, strategy.client_keys = dealer.keys, {}
    for group, labels in table.items():
        held = {strategy.slots[g] for g in labels.values()}
        got = release_keys_in_circuit(dealer, [int(slot in held) for slot in dealer.labels])
        strategy.client_keys[group] = {slot: got[slot] for slot in sorted(held)}
    return table


class PSITrivialCircuitWithKey(PSITrivialCircuit):
    """Exact PSI alignment + per-label key release inside the circuit."""
    max_labels = 256          # public bound M_max on distinct labels; set before calling

    def __call__(self, inputs: MappingInputs) -> RelationTable:
        return _release_keys(self, super().__call__(inputs))


class FuzzyPSICircuitWithKey(FuzzyPSICircuit):
    """Fuzzy PSI alignment + per-label key release inside the circuit."""
    max_labels = 256          # public bound M_max on distinct labels; set before calling

    def __call__(self, inputs: MappingInputs) -> RelationTable:
        return _release_keys(self, super().__call__(inputs))
