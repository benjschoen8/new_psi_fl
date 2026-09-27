"""Fallback key release: OPRF-based PSI-Payload (RFC 9497, OPRF mode, ristretto255-SHA512).

Used when alignment ran client-to-client and the Aggregator holds no Circuit-PSI
output shares. The Aggregator (OPRF server) publishes, per label L,
tag_L = KDF(F(sk, L), "tag") and ct_L = K_L xor KDF(F(sk, L), "key").
A client obtains F(sk, x) for its labels through blind evaluation, with the
number of queries padded to a fixed maximum, then matches tags and unmasks K_L.
Aggregator learns: nothing but the (padded) query count. Client learns: K_L
for its own labels that are in the table, and the table size.
"""
import hashlib
import secrets

from . import ristretto as rg

CONTEXT = b'OPRFV1-' + bytes([0]) + b'-ristretto255-SHA512'
HASH_TO_GROUP_DST = b'HashToGroup-' + CONTEXT
HASH_TO_SCALAR_DST = b'HashToScalar-' + CONTEXT


def _i2osp(n, k):
    return n.to_bytes(k, 'big')


def derive_key_pair(seed: bytes, info: bytes):
    derive_input = seed + _i2osp(len(info), 2) + info
    for counter in range(256):
        sk = rg.hash_to_scalar(derive_input + _i2osp(counter, 1), b'DeriveKeyPair' + CONTEXT)
        if sk:
            return sk, rg.BASE * sk
    raise ValueError('DeriveKeyPairError')


def blind(data: bytes, r: int = None):
    r = r if r is not None else rg.random_scalar()
    P = rg.hash_to_group(data, HASH_TO_GROUP_DST)
    if P == rg.IDENTITY:
        raise ValueError('InvalidInputError')
    return r, (P * r).encode()


def blind_evaluate(sk: int, blinded: bytes) -> bytes:
    return (rg.Point.decode(blinded) * sk).encode()


def finalize(data: bytes, r: int, evaluated: bytes) -> bytes:
    unblinded = (rg.Point.decode(evaluated) * pow(r, -1, rg.L)).encode()
    return hashlib.sha512(_i2osp(len(data), 2) + data + _i2osp(len(unblinded), 2)
                          + unblinded + b'Finalize').digest()


def evaluate(sk: int, data: bytes) -> bytes:
    """Server-side direct PRF evaluation (no blinding), equal to the blinded protocol output."""
    return finalize(data, 1, (rg.hash_to_group(data, HASH_TO_GROUP_DST) * sk).encode())


def _kdf(y: bytes, purpose: bytes, n: int) -> bytes:
    return hashlib.shake_256(b'psi-payload/' + purpose + b'/' + y).digest(n)


# ------------------------------------------------------------------ PSI-Payload
class PayloadServer:
    """Aggregator: OPRF key + table {label: payload} (payload = K_L)."""

    def __init__(self, payloads: dict, sk: int = None):
        if not payloads or len({len(v) for v in payloads.values()}) != 1:
            raise ValueError('need nonempty payloads of equal length')
        self.sk = sk or rg.random_scalar()
        self._payloads = {self.encode_label(k): v for k, v in payloads.items()}

    @staticmethod
    def encode_label(label) -> bytes:
        return str(label).encode('utf-8')

    def table(self):
        """Published once (e.g. on the BB), shuffled so order reveals nothing."""
        rows = []
        for label, payload in self._payloads.items():
            y = evaluate(self.sk, label)
            rows.append((_kdf(y, b'tag', 16), bytes(a ^ b for a, b in zip(payload, _kdf(y, b'key', len(payload))))))
        secrets.SystemRandom().shuffle(rows)
        return rows

    def respond(self, blinded_queries):
        return [blind_evaluate(self.sk, q) for q in blinded_queries]


class PayloadClient:
    def __init__(self, labels, max_queries: int):
        labels = list(dict.fromkeys(labels))
        if len(labels) > max_queries:
            raise ValueError('more labels than the public query bound')
        # Pad with random dummies: the server always sees exactly max_queries elements.
        self._inputs = [PayloadServer.encode_label(x) for x in labels] + \
                       [b'\0dummy' + secrets.token_bytes(16) for _ in range(max_queries - len(labels))]
        self._labels = labels
        self._blinds = []

    def queries(self):
        out = []
        for data in self._inputs:
            r, q = blind(data)
            self._blinds.append(r)
            out.append(q)
        return out

    def recover(self, evaluated, table):
        lookup = dict(table)
        got = {}
        for label, data, r, e in zip(self._labels, self._inputs, self._blinds, evaluated):
            y = finalize(data, r, e)
            ct = lookup.get(_kdf(y, b'tag', 16))
            if ct is not None:
                got[label] = bytes(a ^ b for a, b in zip(ct, _kdf(y, b'key', len(ct))))
        return got
