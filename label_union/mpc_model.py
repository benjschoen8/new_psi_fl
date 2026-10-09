"""Cost model of the hybrid setup MPC (setup_smoke_hybrid --mpc-model): the grouping itself runs as its
ideal functionality, the MP-SPDZ traffic, rounds and compute time are predicted from fits to real
MP-SPDZ 0.4.2 sessions of the same circuits.

Configuration modelled: pair version hegc (pair_inner.mpc on hemi, 61-bit prime, + pair_gc.mpc with
symbols and 45 image anchors), grouping version 2 on atlas with XOR-share input, optional padding MPC.
Fit points (bytes per party = MP-SPDZ 'Data sent', rounds = its '~rounds'):
  HE     m in {16, 32, 62, 100}, d in {48, 64, 128}      error <= 5 % for m >= 32
  GC     m in {16, 32, 62, 100}, semi-bin and yao         error <= 5 % for m >= 32
  group  n in {3, 5, 10}, m in {8 .. 62}, 1 or 2 steps, plus n=10/m=62 (GX10 run)   error ~10-20 %
  pad    n in {3, 10, 50}
Extrapolation to n = 50, m = 100 is beyond the fitted range: treat as an estimate (factor ~1.5).
Compute seconds use throughputs of the GX10 runs (10 clients, m = 62): pair 0.02 s/MB, grouping
0.053 s per MB per party.
"""
import math

PAIR_S_PER_MB = 0.02
GROUP_S_PER_PARTY_MB = 0.053


def he(m, d):
    """HE inner products of one pair: (MB both parties, rounds)."""
    return 1.0 + m * m * (6.2e-4 + 2.88e-4 * d), 15 + 2.6e-3 * m * m * d


def gc(m, protocol):
    """Garbled / GMW comparison of one pair: (MB both parties, rounds, share of the MB sent by party 0)."""
    if protocol == 'yao':
        return 9.6e-3 * m * m, 4 + .5 * m, .96
    return .5 + 4.65e-3 * m * m, 25 + .041 * m * m, .5


def group(n, m, steps):
    """Global grouping (atlas, version 2): (MB per party, rounds)."""
    N = n * m
    mb = .42 * n + (5.5e-5 + 6.3e-6 * n) * N * N
    rounds = .6 * N + 21.6 * n - 7
    extra = max(0, steps - 1)
    mb += extra * (1.1e-3 + 2.7e-4 * n) * N * N
    rounds += extra * N * math.log2(N) * (.4 + .08 * n)
    return mb, rounds


def group_memory(n, m, block_rows=64):
    """Peak RSS of one grouping party (MB): the dense N x N share matrix (16 B per entry), plus the
    row-block work space (comparison registers of the min tree), plus the process. Fitted to
    N = 186 .. 1240, block 16 / 64 / 256 (psutil peak RSS, atlas-party.x)."""
    N = n * m
    return 20 + 16e-6 * N * N + 1.04e-3 * N * min(N, block_rows)


def pad(n):
    """Padding-size MPC: (MB per party, rounds)."""
    return .005 * n, 14 * n


def estimate(n, m, d, nimg, steps, gc_protocol='semi-bin', pair_concurrency=2, pair_workers=8, pad_max='mpc'):
    """Same keys as mpspdz_pairwise_group's stats (estimated=True)."""
    if nimg not in (0, 45):
        raise ValueError('the model is fitted with 45 image anchors')
    he_mb, he_r = he(m, d)
    gc_mb, gc_r, gc_share0 = gc(m, gc_protocol)
    g_mb, g_r = group(n, m, steps)
    p_mb, p_r = pad(n) if pad_max == 'mpc' else (0., 0.)
    pairs = n * (n - 1) // 2
    pair_mb = he_mb + gc_mb
    sent, recv, rounds = [0.] * n, [0.] * n, [0.] * n
    for p in range(n):
        for q in range(p + 1, n):
            s0 = he_mb / 2 + gc_mb * gc_share0                     # party 0 of the pair = the smaller id
            for me, mine in ((p, s0), (q, pair_mb - s0)):
                sent[me] += mine
                recv[me] += pair_mb - mine
                rounds[me] += (he_r + gc_r) / pair_concurrency     # a client's pairs run concurrently
    for c in range(n):
        sent[c] += g_mb + p_mb
        recv[c] += g_mb + p_mb
        rounds[c] += g_r + p_r
    matching_seconds = pairs * pair_mb * PAIR_S_PER_MB / max(1, min(pair_workers, pairs))
    group_seconds = g_mb * GROUP_S_PER_PARTY_MB
    return dict(backend='model', estimated=True, n=n, m=m, rows=n * m, d=d, nimg=nimg, propagation_steps=steps,
                pair_protocol='hegc', gc_protocol=gc_protocol, group_protocol='atlas', group_version=2,
                pad_max=pad_max, pair_sessions=pairs, pair_he_MB=pairs * he_mb, pair_gc_MB=pairs * gc_mb,
                pair_global_MB=pairs * pair_mb, group_global_MB=n * g_mb, pad_max_MB=n * p_mb,
                global_MB=pairs * pair_mb + n * (g_mb + p_mb),
                client_sent_MB=sent, client_received_MB=recv, client_rounds=rounds,
                matching_seconds=matching_seconds, group_seconds=group_seconds,
                wall_seconds=matching_seconds + group_seconds, compile_seconds=0.,
                group_peak_MB_per_party=group_memory(n, m), group_peak_MB_host=n * group_memory(n, m))


if __name__ == '__main__':
    # self-check against measured points (MB): HE m=62 d=64 73.2, GC semi-bin m=62 18.4,
    # group n=10 m=62 (GX10) 495 total, n=10 m=16 2 steps 1041 total
    assert abs(he(62, 64)[0] - 73.2) / 73.2 < .05
    assert abs(gc(62, 'semi-bin')[0] - 18.4) / 18.4 < .05
    assert abs(group(10, 62, 1)[0] * 10 - 495) / 495 < .1
    assert abs(group(10, 16, 2)[0] * 10 - 1041) / 1041 < .25
    assert abs(group_memory(10, 100) - 103) / 103 < .1 and abs(group_memory(20, 62) - 126) / 126 < .15
    print('ok')
