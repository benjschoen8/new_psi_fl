import os
import json
import random
import torch
import numpy as np
from collections import Counter, defaultdict
from torchvision import datasets, transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
from dirichlet_noniid import partition_data, partition_data_noniid_label, partition_data_quantity_skew, partition_data_quantity_skew_equalSize

class Global_Dataset(Dataset):
    def __init__(self, dataset, mapping):
        self.dataset = dataset
        self.mapping = mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, local_label = self.dataset[idx]
        global_label = self.mapping[int(local_label)]
        return img, global_label

class LabelPermutedDataset(Dataset):
    def __init__(self, original_dataset, mapping_dict):
        self.dataset = original_dataset
        self.mapping_dict = mapping_dict
        
        if hasattr(self.dataset, 'targets'):
            self.targets = [self.mapping_dict[int(y)] for y in self.dataset.targets]
        elif hasattr(self.dataset, 'labels'):
            self.labels = [self.mapping_dict[int(y)] for y in self.dataset.labels]
        else:
            self.targets = [self.mapping_dict[int(y)] for _, y in self.dataset.samples]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, original_label = self.dataset[idx]
        permuted_label = self.mapping_dict[int(original_label)]
        return img, permuted_label

class ClassSubsetDataset(Dataset):
    """A client's view of a dataset restricted to its own classes: labels renumbered 0..k-1 in the
    order of `classes` (global label ids); `classes` holds their names (setup.label_names reads it)."""
    def __init__(self, base, classes, names):
        self.base, self.remap = base, {int(c): i for i, c in enumerate(classes)}
        self.classes = [names[int(c)] for c in classes]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        return img, self.remap[int(y)]


def partition_class_subsets(train_labels, test_labels, n_clients, lo, hi, seed, min_holders=2, full=False, cover0=None):
    """Every client draws k in [lo, hi] classes; classes go to the least-covered ones first (random
    ties), so coverage is even; redrawn until every class has >= min_holders clients (if possible).
    A class's train / test samples are split evenly at random among its holders (full=True: every
    holder gets all of them).
    cover0: holders a class already has elsewhere (the CIFAR-10 + STL-10 special clients), counted in
    'least covered'; greedy keeps max - min cover <= 1, so >= 2 holders per class whenever
    sum(k) + sum(cover0) >= 2 C.
    Returns (classes per client (sorted global ids), train idcs, test idcs)."""
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    C, rng = int(max(train_labels.max(), test_labels.max())) + 1, np.random.default_rng(seed)
    for _ in range(1000):
        ks = rng.integers(lo, hi + 1, n_clients)
        cover, own = np.zeros(C, int) if cover0 is None else np.array(cover0, int), []
        for k in ks:
            order = np.lexsort((rng.random(C), cover))                  # least covered first, random ties
            mine = np.sort(order[:k])
            cover[mine] += 1
            own.append(mine)
        if cover.min() >= min(min_holders, cover.sum() // C):
            break
    # full: nothing is divided, so a class no client drew is simply absent (e.g. 6 clients x 3-4 of
    # EMNIST's 62 classes); split keeps the old check that every image has a holder
    return _split_among_holders(own, train_labels, test_labels, C, rng, full, allow_uncovered=full)


def partition_even(train_labels, test_labels, n_clients, seed, holders=2, full=False):
    """Classes and images spread evenly: every class goes to exactly `holders` clients, every client
    gets the same number of classes (+-1) and about the same number of images (classes, largest first,
    go to the least-loaded clients with a free class slot: LPT scheduling). A class's samples are split
    evenly among its holders. Same return values as partition_class_subsets."""
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    C, rng = int(max(train_labels.max(), test_labels.max())) + 1, np.random.default_rng(seed)
    if not 1 <= holders <= n_clients:
        raise ValueError(f'holders must be in 1..{n_clients}')
    share = np.bincount(train_labels, minlength=C) / holders           # images per holder of a class
    slots = np.full(n_clients, C * holders // n_clients)
    slots[rng.permutation(n_clients)[:C * holders % n_clients]] += 1
    load, own = np.zeros(n_clients), [[] for _ in range(n_clients)]
    for left, c in zip(range(C, 0, -1), np.lexsort((rng.random(C), -share))):   # largest first
        free = np.flatnonzero(slots)
        must = slots[free] == left              # a slot for every class left: must take this one too
        pick = free[np.lexsort((rng.random(len(free)), load[free], ~must))[:holders]]
        slots[pick] -= 1
        load[pick] += share[c]
        for h in pick:
            own[h].append(c)
    return _split_among_holders([np.sort(m) for m in own], train_labels, test_labels, C, rng, full)


def _split_among_holders(own, train_labels, test_labels, C, rng, full=False, allow_uncovered=False):
    """A class's train / test samples split evenly at random among the clients holding it
    (full=True: every holder gets all of them)."""
    n = len(own)
    tr, te = {i: [] for i in range(n)}, {i: [] for i in range(n)}
    for c in range(C):
        holders = [i for i in range(n) if c in own[i]]
        if not holders and allow_uncovered:
            continue
        if not holders:
            raise ValueError(f'class {c} has no client: too few classes per client to cover all {C}')
        for labels, out in ((train_labels, tr), (test_labels, te)):
            idx = rng.permutation(np.flatnonzero(labels == c))
            for h, part in zip(holders, [idx] * len(holders) if full else np.array_split(idx, len(holders))):
                out[h] += part.tolist()
    return [m.tolist() for m in own], tr, te


def _targets(ds):
    if hasattr(ds, 'targets'):
        return list(ds.targets)
    if hasattr(ds, 'labels'):
        return list(ds.labels)
    return [label for _, label in ds.samples]


class MergedClassDataset(Dataset):
    """One client's labels over several datasets with the same class names (CIFAR-10 + STL-10): every
    part is a ClassSubsetDataset view renumbered in the same name order, so 'cat' is one local label
    with the images of both datasets. classes / remap / indices as label_names and the metadata expect
    (indices: the second dataset's image ids are offset by the first dataset's length)."""
    def __init__(self, parts, classes, offsets):
        self.parts, self.classes, self.remap = parts, list(classes), {i: i for i in range(len(classes))}
        self.indices = [o + int(i) for p, o in zip(parts, offsets) for i in p.indices]
        self._starts = np.cumsum([0] + [len(p) for p in parts])

    def __len__(self):
        return int(self._starts[-1])

    def __getitem__(self, idx):
        k = int(np.searchsorted(self._starts, idx, side='right')) - 1
        return self.parts[k][idx - int(self._starts[k])]


def mixed_choice(n_clients, lo, hi, seed, root):
    """Class names of each special client: k in [lo, hi] of the names CIFAR-10 and STL-10 share,
    least covered first (random ties). Computed before the per-dataset split, which counts them."""
    rng = np.random.default_rng(seed)
    stl = set(get_readable_class_names('STL10', root=root))
    shared = [x for x in get_readable_class_names('CIFAR10', root=root) if x in stl]   # CIFAR-10 order
    cover, out = np.zeros(len(shared), int), []
    for _ in range(n_clients):
        mine = sorted(np.lexsort((rng.random(len(shared)), cover))[:int(rng.integers(lo, hi + 1))])
        cover[mine] += 1
        out.append([shared[j] for j in mine])
    return out


def mixed_cifar_stl_clients(n_clients, lo, hi, seed, root, batch_size):
    """Special clients holding CIFAR-10 and STL-10 together: each draws k in [lo, hi] of the classes the
    two share by name (least covered first, random ties), and gets ALL train / test images of those
    classes from BOTH datasets, merged into one label per name. Their test data stays per dataset:
    entry['tests'] = [('CIFAR10', loader), ('STL10', loader)] (accuracy counts towards each dataset)."""
    parts = {}
    for d in ('CIFAR10', 'STL10'):
        names = list(get_readable_class_names(d, root=root))
        tr, te = get_raw_dataset_transform(d, root, train=True), get_raw_dataset_transform(d, root, train=False)
        parts[d] = (names, tr, te, np.asarray(_targets(tr)), np.asarray(_targets(te)))
    entries = []
    for i, classes in enumerate(mixed_choice(n_clients, lo, hi, seed, root)):
        views = {}
        for split in ('train', 'test'):
            vs, offs, off = [], [], 0
            for d in ('CIFAR10', 'STL10'):
                names, tr, te, ytr, yte = parts[d]
                base, y = (tr, ytr) if split == 'train' else (te, yte)
                ids = [names.index(x) for x in classes]                        # same local order in both
                vs.append(Subset(ClassSubsetDataset(base, ids, names), np.flatnonzero(np.isin(y, ids)).tolist()))
                offs.append(off)
                off += len(base)
            views[split] = (vs, offs)
        print(f"CIFAR10+STL10 client {i}: {len(classes)} classes {classes}, "
              f"{sum(len(v) for v in views['train'][0])} train images (both datasets, merged per name)")
        entries.append({
            'train': DataLoader(MergedClassDataset(*views['train'][:1], classes, views['train'][1]),
                                batch_size=batch_size, shuffle=True, num_workers=0),
            'test': DataLoader(MergedClassDataset(*views['test'][:1], classes, views['test'][1]),
                               batch_size=batch_size, shuffle=False, num_workers=0),
            'tests': [(d, DataLoader(v, batch_size=batch_size, shuffle=False, num_workers=0))
                      for d, v in zip(('CIFAR10', 'STL10'), views['test'][0])]})
    return entries


def get_split_cache_path(DATA_ROOT, dataset_name, alpha, total_clients, num_new_clients, seed):
    cache_dir = os.path.join(DATA_ROOT, "splits")
    os.makedirs(cache_dir, exist_ok=True)

    # 把dirichlet alpha的.去掉改成p (e.g. 0.1 -> 0p1)
    alpha_str = f"{alpha:.1f}".replace(".", "p")

    filename = f"{dataset_name}_C{total_clients}_New{num_new_clients}_alpha{alpha_str}_seed{seed}.json"
    return os.path.join(cache_dir, filename)

# ==========================================
# Get Label Counts
# ==========================================
def get_label_counts(dataset, indices):
    """
    label count of subset
    """
    if hasattr(dataset, 'targets'):
        all_labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        all_labels = np.array(dataset.labels)
    else:
        all_labels = np.array([y for x, y in dataset.samples])
    
    subset_labels = all_labels[indices]

    counter = Counter(subset_labels)
    
    return ", ".join([f"{k}:{v}" for k, v in sorted(counter.items())])

# ==========================================
# Dataset Transform
# ==========================================
def get_transforms(name):
    if name == 'USPS':                                            # 16x16 digits fill the frame; MNIST digits sit
        return transforms.Compose([                               # in a 20x20 box with a border: shrink + pad so
            transforms.Resize((22, 22)),                          # both look alike (same 32x32 framing)
            transforms.Pad(5),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    if name in ['MNIST', 'FashionMNIST', 'USPS']:
        return transforms.Compose([
            transforms.Resize((32, 32)),                 
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))

            # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])

    elif name == 'EMNIST':
        return transforms.Compose([
            transforms.Resize((32, 32)),                 
            transforms.Grayscale(num_output_channels=3),
            # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    elif name == 'STL10':                                         # 96x96 photos -> 32x32 like CIFAR
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    elif name == 'EuroSAT':                                       # 64x64 satellite photos -> 32x32
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    elif name in ['CIFAR10', 'CIFAR100', 'SVHN']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    return None

# ==========================================
# Reading Raw Datasets
# ==========================================
def get_raw_dataset_transform(name, root, train=True):
    transform = get_transforms(name)
    
    if name == 'MNIST':
        return datasets.MNIST(root, train=train, download=True, transform=transform)
    
    elif name == 'FashionMNIST':
        return datasets.FashionMNIST(root, train=train, download=True, transform=transform)
    
    elif name == 'EMNIST':
        # return datasets.EMNIST(root, split='balanced', train=train, download=False, transform=transform)
        # return datasets.EMNIST(root, split='bymerge', train=train, download=False, transform=transform)
        return datasets.EMNIST(root, split='byclass', train=train, download=True, transform=transform)
    
    elif name == 'CIFAR10':
        return datasets.CIFAR10(root, train=train, download=True, transform=transform)

    elif name == 'EuroSAT':                                       # one folder of 27,000 images: fixed 80 / 20
        return EuroSATSplit(datasets.EuroSAT(root, download=True, transform=transform), train)
    
    elif name == 'CIFAR100':
        return datasets.CIFAR100(root, train=train, download=True, transform=transform)

    elif name == 'USPS':
        return datasets.USPS(root, train=train, download=True, transform=transform)

    elif name == 'STL10':                                         # labelled split only (500 / 800 per class)
        d = datasets.STL10(root, split='train' if train else 'test', download=True, transform=transform)
        d.classes = stl10_classes()
        return d

    elif name == 'SVHN':                                          # digits 0-9 (torchvision maps label 10 -> 0)
        d = datasets.SVHN(root, split='train' if train else 'test', download=True, transform=transform)
        d.classes = [str(i) for i in range(10)]
        return d
    
EUROSAT_CLASSES = ('AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture',
                   'PermanentCrop', 'Residential', 'River', 'SeaLake')          # torchvision's (sorted folder) order


class EuroSATSplit(Dataset):
    """EuroSAT has no official split: a fixed stratified 80 / 20 split (seed 0, independent of the run seed)."""
    def __init__(self, base, train):
        y = np.asarray(base.targets)
        rng = np.random.default_rng(0)
        pick = []
        for c in range(len(base.classes)):
            idx = rng.permutation(np.flatnonzero(y == c))
            cut = int(round(.8 * len(idx)))
            pick += (idx[:cut] if train else idx[cut:]).tolist()
        self.base, self.idx = base, sorted(pick)
        self.targets = y[self.idx].tolist()
        self.classes = list(base.classes)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]


def stl10_classes():
    """STL-10's classes in its label order, named like CIFAR-10's (public dictionary: 'car' is
    'automobile'); 9 of 10 are CIFAR-10 classes, 'monkey' is STL-only (CIFAR-10's 'frog' is CIFAR-only)."""
    return ['airplane', 'bird', 'automobile', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']


# ==========================================
# Get Readable Class Names
# ==========================================
def get_readable_class_names(name, root='./data/raw'):
    if name in ['MNIST', 'USPS', 'SVHN']:
        return [str(i) for i in range(10)]

    elif name == 'FashionMNIST':
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
    elif name == 'EMNIST':
        # d = datasets.EMNIST(root, split='balanced', train=True, download=False)
        # d = datasets.EMNIST(root, split='bymerge', train=True, download=True)
        d = datasets.EMNIST(root, split='byclass', train=True, download=True)
        return d.classes
        
    elif name == 'STL10':
        return stl10_classes()

    elif name == 'EuroSAT':
        return list(EUROSAT_CLASSES)

    elif name == 'CIFAR10':
        # CIFAR10 in-build classes: ['airplane', 'automobile', 'bird', ...]
        d = datasets.CIFAR10(root, train=True, download=True)
        return d.classes
    
    elif name == 'CIFAR100':
        # CIFAR100 in-build classes: ['apple', 'aquarium_fish', ...]
        d = datasets.CIFAR100(root, train=True, download=True)
        return d.classes        

# ==========================================
# Loading Datasets
# ==========================================
def load_partitioned_datasets(args, DATA_ROOT, **exp_conf):
    # configs
    dataset_configs = {
        'MNIST': (args.num_train_mnist + args.num_new_clients) if args.num_train_mnist > 0 else 0,
        'FashionMNIST': (args.num_train_fashionmnist + args.num_new_clients) if args.num_train_fashionmnist > 0 else 0,
        'EMNIST': (args.num_train_emnist + args.num_new_clients) if args.num_train_emnist > 0 else 0,
        'CIFAR10': (args.num_train_cifar10 + args.num_new_clients) if args.num_train_cifar10 > 0 else 0,
        'CIFAR100': (args.num_train_cifar100 + args.num_new_clients) if args.num_train_cifar100 > 0 else 0,
        'USPS': (args.num_train_usps + args.num_new_clients) if args.num_train_usps > 0 else 0,
        'STL10': (getattr(args, 'num_train_stl10', 0) + args.num_new_clients) if getattr(args, 'num_train_stl10', 0) > 0 else 0,
        'SVHN': (getattr(args, 'num_train_svhn', 0) + args.num_new_clients) if getattr(args, 'num_train_svhn', 0) > 0 else 0,
        'EuroSAT': (getattr(args, 'num_train_eurosat', 0) + args.num_new_clients) if getattr(args, 'num_train_eurosat', 0) > 0 else 0
    }
    batch_size = exp_conf.get('batch_size', 64)
    dirichlet_alpha = exp_conf.get('dirichlet_alpha', 0.1)
    public_ratio = exp_conf.get('public_data_ratio', 1.0)

    # Partition Datasets
    print(f"{'='*100}")
    if getattr(args, 'class_subsets', None):
        print(f"Loading Datasets with label split ({args.class_subsets} classes per client, "
              f"class share {getattr(args, 'class_share', 'split')})")
    else:
        print(f"Loading Datasets with Non-IID Split (Dirichlet distribution, Alpha={dirichlet_alpha})")
    print(f"{'='*100}")

    all_client_data_loaders = {}
    server_train_loaders = {}
    server_test_loaders = {}
    usps_label_mapping = None

    n_loaded = 0                     # class-subset seed offset: the k-th loaded dataset draws its own
    mixed = getattr(args, 'num_train_cifar10stl10', 0)            # CIFAR-10 + STL-10 special clients
    for d_name, n_clients in dataset_configs.items():   # subsets (the first keeps args.seed, as before)
        if n_clients == 0:
            continue
        sub_seed, n_loaded = args.seed + 1000003 * n_loaded, n_loaded + 1
        
        # --- Readable Class Names (e.g. dog, cat ...) ---
        class_names = get_readable_class_names(d_name, root=DATA_ROOT)
        print(f"[-] Readable Class Names ({len(class_names)} classes): {class_names}\n")

        print(f"\n>>> Processing {d_name} ({n_clients} Clients)")   

        # Getting Datasets
        train_dataset = get_raw_dataset_transform(d_name, DATA_ROOT, train=True)
        test_dataset = get_raw_dataset_transform(d_name, DATA_ROOT, train=False)

        if d_name == 'USPS' and getattr(args, 'usps_shuffle', True):   # original experiment: permuted USPS labels
            if usps_label_mapping is None:
                original_labels = list(range(10))
                shuffled_labels = original_labels.copy()
                random.shuffle(shuffled_labels)
                usps_label_mapping = {ori: shf for ori, shf in zip(original_labels, shuffled_labels)}
                print(f"[!] USPS Label Shuffled Mapping: {usps_label_mapping}")
            
            train_dataset = LabelPermutedDataset(train_dataset, usps_label_mapping)
            test_dataset = LabelPermutedDataset(test_dataset, usps_label_mapping)

        # Getting Labels
        # 拿這個 dataset 底下每一筆資料的 label
        if hasattr(train_dataset, 'targets'): 
            train_labels = train_dataset.targets
        elif hasattr(train_dataset, 'labels'): 
            train_labels = train_dataset.labels
        else: 
            train_labels = []
            for path, label in train_dataset.samples:
                train_labels.append(label)

        if hasattr(test_dataset, 'targets'): 
            test_labels = test_dataset.targets
        elif hasattr(test_dataset, 'labels'): 
            test_labels = test_dataset.labels
        else: 
            test_labels = []
            for path, label in test_dataset.samples:
                test_labels.append(label)
                
        train_labels_for_split = train_labels
        test_labels_for_split  = test_labels

        # Dirichlet Non-IID Partition
        cache_path = get_split_cache_path(
            DATA_ROOT,
            d_name,
            dirichlet_alpha,
            n_clients,
            args.num_new_clients,
            args.seed 
        )

        subsets = getattr(args, 'class_subsets', None)
        full = getattr(args, 'class_share', 'split') == 'full'
        if subsets == 'even':                                          # own classes per client (no cache:
            own, train_idcs, test_idcs = partition_even(train_labels, test_labels, n_clients, sub_seed, full=full)
        elif subsets:                                                  # deterministic from the seed)
            lo, hi = map(int, subsets.split(','))
            cover0 = None                                              # special clients count as holders
            if mixed and d_name in ('CIFAR10', 'STL10'):
                picked = sum(mixed_choice(mixed, lo, hi, args.seed + 7777777, DATA_ROOT), [])
                cover0 = [picked.count(x) for x in class_names]
            own, train_idcs, test_idcs = partition_class_subsets(train_labels, test_labels, n_clients, lo, hi,
                                                                 sub_seed, full=full, cover0=cover0)
        if subsets:
            names = list(class_names)                                  # readable names ('3', not '3 - three')
            client_loaders = []
            for i in range(n_clients):
                print(f"{i:<6} | {len(train_idcs[i]):<6} | {len(test_idcs[i]):<6} | {len(own[i])} classes: "
                      f"{[names[c] for c in own[i]]}")
                client_loaders.append({
                    'train': DataLoader(Subset(ClassSubsetDataset(train_dataset, own[i], names), train_idcs[i]),
                                        batch_size=batch_size, shuffle=True, num_workers=0),
                    'test': DataLoader(Subset(ClassSubsetDataset(test_dataset, own[i], names), test_idcs[i]),
                                       batch_size=batch_size, shuffle=False, num_workers=0)})
            all_client_data_loaders[d_name] = client_loaders
            continue
        if args.noniid_partition in ["noniid_label", "quantity_skew", "quantity_skew_equalSize"]:
            cache_dir = os.path.dirname(cache_path)
            cache_name = f"{d_name}_C{n_clients}_New{args.num_new_clients}_{args.noniid_partition}_alpha{str(dirichlet_alpha).replace('.', 'p')}_seed{args.seed}.json"
            cache_path = os.path.join(cache_dir, cache_name)

        if os.path.exists(cache_path):
            print(f"Found existing split for {d_name}, loading from {cache_path}")
            with open(cache_path, "r") as f:
                cached = json.load(f)

            # json 會把 key 變成字串所以要轉回 int
            train_idcs = {int(k): v for k, v in cached["train"].items()}
            test_idcs  = {int(k): v for k, v in cached["test"].items()}
        else:
            print(f"No existing split for {d_name}, generating new partition...")
            if args.noniid_partition == "dirichlet":
                train_idcs, test_idcs = partition_data(train_labels_for_split, test_labels_for_split, alpha=dirichlet_alpha, total_clients=n_clients, num_new_clients=args.num_new_clients)
            elif args.noniid_partition == "noniid_label":
                train_idcs, test_idcs = partition_data_noniid_label(train_labels_for_split, test_labels_for_split, alpha=dirichlet_alpha, total_clients=n_clients, num_new_clients=args.num_new_clients)
            elif args.noniid_partition == "quantity_skew":
                train_idcs, test_idcs = partition_data_quantity_skew(train_labels_for_split, test_labels_for_split, alpha=dirichlet_alpha, total_clients=n_clients, num_new_clients=args.num_new_clients)
            elif args.noniid_partition == "quantity_skew_equalSize":
                train_idcs, test_idcs = partition_data_quantity_skew_equalSize(train_labels_for_split, test_labels_for_split, alpha=dirichlet_alpha, total_clients=n_clients, num_new_clients=args.num_new_clients)

            # 把用這次參數切的資料集存在 json
            to_save = {
                "train": train_idcs,
                "test": test_idcs,
                "meta": {
                    "dataset": d_name,
                    "alpha": dirichlet_alpha,
                    "total_clients": n_clients,
                    "num_new_clients": args.num_new_clients,
                    "seed": args.seed,
                }
            }
            with open(cache_path, "w") as f:
                json.dump(to_save, f, indent=2)
            print(f"Saved split for {d_name} to {cache_path}")

        # Dataloader
        client_loaders = []
        print(f"{'Client':<6} | {'Train':<6} | {'Test':<6} | {'Sample'}")
        print("-" * 120)

        for i in range(n_clients):
            train_subset = Subset(train_dataset, train_idcs[i])
            test_subset = Subset(test_dataset, test_idcs[i])

            train_cnt = len(train_subset)
            test_cnt = len(test_subset)
            train_info_str = get_label_counts(train_dataset, train_idcs[i])
            test_info_str  = get_label_counts(test_dataset,  test_idcs[i])

            print(f"{i:<6} | {train_cnt:<6} | {test_cnt:<6} | Train: [{train_info_str}]")
            print(f"{'':<6} | {'':<6} | {'':<6} | Test : [{test_info_str}]")
            print("-" * 60) 
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            client_loaders.append({
                'train': train_loader,
                'test': test_loader
            })

        all_client_data_loaders[d_name] = client_loaders

        global_samples_per_class = exp_conf.get('global_samples_per_class', 1)

        class_indices = defaultdict(list)
        for idx in range(len(train_dataset)):
            lbl = int(train_labels_for_split[idx])
            class_indices[lbl].append(idx)
            
        selected_indices = []
        for lbl, idcs in class_indices.items():
            if len(idcs) >= global_samples_per_class:
                selected_indices.extend(random.sample(idcs, global_samples_per_class))
            else:
                selected_indices.extend(random.choices(idcs, k=global_samples_per_class))
                
        public_train_dataset = Subset(train_dataset, selected_indices)

        server_train_loaders[d_name] = DataLoader(
            #public_train_dataset, 
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,  
            num_workers=0
        )

        server_test_loaders[d_name] = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )

    if mixed:
        if args.num_new_clients:
            raise ValueError('--num-train-cifar10stl10 does not support --num-new-clients')
        subsets = getattr(args, 'class_subsets', None)
        lo, hi = map(int, subsets.split(',')) if subsets and subsets != 'even' else (3, 4)
        all_client_data_loaders['CIFAR10+STL10'] = mixed_cifar_stl_clients(
            mixed, lo, hi, args.seed + 7777777, DATA_ROOT, batch_size)

    print(f"{'='*100}\n")

    return all_client_data_loaders, server_train_loaders, server_test_loaders

