"""Separate setup-only version; the original circuit_union is unchanged.

Pairwise 2PC produces masked match shares. Global honest-majority Shamir MPC
groups these shares and reveals final keys privately to their row owners.
The single-host harness possesses all inputs/outputs for measurement, as in
the original simulator. It is not an isolated distributed deployment.
"""
import os
import time

from label_union import circuit_union as original
from label_union.oprf_union import _union_from_tags, _pk_secagg, slot_key, BUCKET_BITS


def circuit_union_with_keys(client_labels, client_keywords, client_sets=None, fuzzy=False,
                            tau=original.TAU, t=2, D=1, m=None, workers=1,
                            bucket_bits=BUCKET_BITS, session=b'label-union-hybrid', secure=True,
                            mpc_backend='pairwise', mpc_options=None):
    if mpc_backend not in ('global', 'pairwise'):
        raise ValueError('unknown MPC backend')
    if not secure or mpc_backend == 'global':
        return original.circuit_union_with_keys(client_labels, client_keywords, client_sets,
            fuzzy=fuzzy, tau=tau, t=t, D=D, m=m, workers=workers,
            bucket_bits=bucket_bits, session=session, secure=secure)
    model = bool((mpc_options or {}).get('model'))             # ideal grouping + MPC cost model, no MP-SPDZ
    if not model and not os.environ.get('MPSPDZ'):
        raise RuntimeError('hybrid setup requires real MPC; set MPSPDZ to an installation (or the cost model)')
    if len(client_labels) < 3 or any(not labels for labels in client_labels):
        raise ValueError('hybrid setup requires at least three nonempty clients')
    if D != 1:
        raise ValueError('hybrid grouping checks convergence after every propagation step (D=1)')
    start = time.perf_counter()
    options = dict(mpc_options or {})
    options.pop('model', None)
    pca_dim = options.pop('pca_dim', None)                   # public PCA of embeddings (new option)
    kws = [original.keyword_rows(keywords, fuzzy) for keywords in client_keywords]
    if pca_dim and fuzzy:
        from label_union.pca import project
        kws = [project(own, pca_dim) for own in kws]
    imgs = [original.image_rows(sets) for sets in client_sets] if client_sets else None
    owners, rows = [], []
    for c, labels in enumerate(client_labels):
        for label in labels:
            owners.append((c, label))
            rows.append((kws[c][label], imgs[c][label] if imgs else None))
    # No plaintext graph/oracle is evaluated in a measured hybrid setup.
    # Correctness against the ideal functionality is checked in integration tests.
    if model:
        import secrets
        from label_union import mpc_model
        root, steps = original.group(rows, tau, t, [c for c, _ in owners])
        kappa = {r: secrets.randbits(126) for r in set(root)}
        keys = [kappa[r] for r in root]
        embs = [kw[1] for kw, _ in rows if kw[0] == 'emb']
        measured = mpc_model.estimate(len(client_labels), m or max(map(len, client_labels)),
                                      len(embs[0]) if embs else 0, len(rows[0][1]) if imgs else 0, steps,
                                      options.get('gc_protocol', 'semi-bin'), options.get('pair_concurrency', 2),
                                      options.get('pair_workers', 8), options.get('pad_max', 'mpc'))
    else:
        from label_union.mpspdz_pairwise import mpspdz_pairwise_group
        _, measured, keys = mpspdz_pairwise_group(rows, [c for c, _ in owners],
            tau=tau, t=t, m=m, **options)
    measured['pca_dim'] = pca_dim if fuzzy else None
    K = [{} for _ in client_labels]
    for (c, label), key in zip(owners, keys):
        K[c][label] = key.to_bytes(17, 'big', signed=True)
    grouping_seconds = time.perf_counter() - start
    tags = [{key: key for key in set(own.values())} for own in K]
    idx, size, stats = _union_from_tags(tags, workers, bucket_bits, session + b'/union')
    sks = [{key: slot_key(key) for key in own} for own in tags]
    pks = _pk_secagg([{idx[c][key]: sk for key, sk in own.items()} for c, own in enumerate(sks)],
                     size, session + b'/pk', workers, stats['secagg'])
    index = [{label: idx[c][key] for label, key in own.items()} for c, own in enumerate(K)]
    sk_out = [{label: sks[c][key] for label, key in own.items()} for c, own in enumerate(K)]
    stats.update(method='circuit-psi-hybrid',
                 mpc=dict(estimated=False, tau=tau, t=t, D=D, measured=measured),
                 group_seconds=grouping_seconds, seconds=time.perf_counter()-start,
                 setup_upload_bytes_per_client=None, setup_download_bytes_per_client=None)
    return index, sk_out, pks, size, stats
