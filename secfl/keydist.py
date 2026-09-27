"""Phase 1 output side: release K_L to exactly the clients whose PSI bit is 1.

Circuit-PSI leaves each (client, L) match bit as XOR shares b = b_c xor b_s,
b_c with the client and b_s with the Aggregator. Per label, one 1-out-of-2 OT
with the Aggregator as sender: m_x = K_L if x xor b_s = 1 else fresh random.
The client chooses x = b_c, so it gets K_L iff b = 1; the Aggregator never learns b.
"""
import secrets

from . import ot

KEY_BYTES = 16


def share_bits(bits):
    """Stand-in for a Circuit-PSI output: split plain bits into XOR shares (b_c, b_s)."""
    b_c = [secrets.randbelow(2) for _ in bits]
    return b_c, [b ^ c for b, c in zip(bits, b_c)]


class KeyDealer:
    """Aggregator side: one 128-bit key per global label."""

    def __init__(self, labels):
        self.labels = tuple(labels)
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError('labels must be nonempty and unique')
        self.keys = {L: secrets.token_bytes(KEY_BYTES) for L in self.labels}

    def ot_pairs(self, b_s):
        if len(b_s) != len(self.labels) or any(b not in (0, 1) for b in b_s):
            raise ValueError('one share bit per label')
        pairs = []
        for L, s in zip(self.labels, b_s):
            decoy = secrets.token_bytes(KEY_BYTES)          # fresh per slot
            pairs.append((self.keys[L], decoy) if s else (decoy, self.keys[L]))  # m_x real iff x^s=1
        return pairs


def release_keys(dealer: KeyDealer, b_s, b_c, session: bytes = b''):
    """Run the OT batch for one client. Returns {L: 16 bytes}: K_L if matched, else random."""
    if len(b_c) != len(dealer.labels):
        raise ValueError('one client share bit per label')
    pairs = dealer.ot_pairs(b_s)
    run = ot.run_iknp if len(pairs) >= ot.KAPPA else ot.run_base_ot    # extension pays off past kappa
    return dict(zip(dealer.labels, run(pairs, list(b_c), session)))


def release_keys_in_circuit(dealer: KeyDealer, bits, session: bytes = b''):
    """Key release computed INSIDE the PSI circuit (ideal functionality).

    The circuit takes b (never revealed) and the Aggregator's K_L and outputs to the
    client only mux(b, K_L, fresh random). The client has no choice bit to flip, so it
    cannot swap its own keys for the keys of labels it does not hold (see release_keys,
    which is only semi-honest secure). Realized as a garbled/GMW circuit, this costs
    128 AND gates per label on top of the PSI circuit; the Aggregator learns nothing.
    ponytail: simulated in the clear; replace with the real circuit's output wires.
    """
    if len(bits) != len(dealer.labels) or any(b not in (0, 1) for b in bits):
        raise ValueError('one bit per label')
    return {L: dealer.keys[L] if b else secrets.token_bytes(KEY_BYTES) for L, b in zip(dealer.labels, bits)}


def assign_slots(labels, max_labels: int):
    """Random injective map label -> slot in [0, max_labels): hides how many labels exist."""
    labels = list(labels)
    if len(set(labels)) != len(labels) or not 0 < len(labels) <= max_labels:
        raise ValueError('need 1..max_labels distinct labels')
    return dict(zip(labels, secrets.SystemRandom().sample(range(max_labels), len(labels))))
