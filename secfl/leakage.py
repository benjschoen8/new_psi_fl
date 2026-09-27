"""Dropout-differencing policy and the explicit leakage log.

Two aggregates over survivor sets that differ by one client reveal that client's
contribution. Pick one policy:
  'abort' : a round counts only if its survivor set equals the reference set; else discard it.
  'dp'    : always accept; every client adds noise so the sum carries at least sigma^2
            even with only t survivors (distributed DP).
  'leak'  : always accept; the survivor-set difference is written into the leakage log.
The log is what the Aggregator learns (the paper's leakage function), round by round.
"""
import json

import numpy as np

POLICIES = ('abort', 'dp', 'leak')


class DropoutPolicy:
    def __init__(self, policy, threshold, sigma=0.0):
        if policy not in POLICIES:
            raise ValueError(f'policy must be one of {POLICIES}')
        if threshold < 1 or sigma < 0 or (policy == 'dp' and sigma == 0):
            raise ValueError('threshold >= 1; dp needs sigma > 0')
        self.policy, self.t, self.sigma = policy, threshold, sigma
        self.reference = None

    def client_noise(self, size, rng=None):
        """Per-client Gaussian share: any >= t survivors sum to variance >= sigma^2.

        ponytail: continuous Gaussian added before rounding; a rigorous guarantee over
        Z_{2^k} needs the distributed discrete Gaussian (Kairouz et al., ICML 2021).
        """
        if self.policy != 'dp':
            return np.zeros(size)
        rng = rng or np.random.default_rng()
        return rng.normal(0, self.sigma / np.sqrt(self.t), size)

    def accept(self, survivors) -> bool:
        survivors = frozenset(survivors)
        if len(survivors) < self.t:
            return False
        if self.policy != 'abort':
            return True
        if self.reference is None:
            self.reference = survivors               # first accepted round fixes the set
        return survivors == self.reference


class LeakageLog:
    """Everything the Aggregator observes, per round."""

    def __init__(self):
        self.rounds = []

    def record(self, round_index, active_labels, N, group_sizes, grad_sums, survivors,
               accepted, previous_survivors=None):
        row = dict(round=round_index, active_labels=sorted(map(str, active_labels)),
                   N={str(k): int(v) for k, v in N.items()},
                   group_size={str(k): int(v) for k, v in group_sizes.items()},
                   aggregate_grad_norm={str(k): float(np.linalg.norm(v)) for k, v in grad_sums.items()},
                   survivors=sorted(map(str, survivors)), accepted=bool(accepted))
        if previous_survivors is not None:
            prev, cur = set(map(str, previous_survivors)), set(row['survivors'])
            row['dropped_since_last'] = sorted(prev - cur)
            row['joined_since_last'] = sorted(cur - prev)
        self.rounds.append(row)
        return row

    def to_json(self) -> str:
        return json.dumps(self.rounds, indent=2)
