"""A1: two-party exact PSI from OPRF (RFC 9497 ristretto255-SHA512). Semi-honest.

Sender S holds items Y and an OPRF key k; receiver R holds items X.
  1. R blinds each x (padded to a public max), S evaluates blindly.
  2. S publishes tags T(y) = trunc(F_k(y)) for its items, shuffled.
  3. R finalizes F_k(x) and keeps x with T(x) in the tag set.
R learns X ∩ Y (and |Y|); S learns only the padded query count.

Key version (psi_with_keys): S attaches a payload per item (K for that label); R obtains the
payload of each intersecting item and nothing for the others (secfl.oprf.PayloadServer).
"""
import hashlib
import secrets

from secfl import oprf, ristretto as rg

TAG_BYTES = 16


def encode(item) -> bytes:
    return item if isinstance(item, bytes) else str(item).encode('utf-8')


def _tag(y: bytes) -> bytes:
    return hashlib.shake_256(b'psi/tag/' + y).digest(TAG_BYTES)


class PSISender:
    def __init__(self, items, sk: int = None):
        self.items = [encode(y) for y in dict.fromkeys(items)]
        if not self.items:
            raise ValueError('sender set must be nonempty')
        self.sk = sk or rg.random_scalar()

    def respond(self, blinded):
        return [oprf.blind_evaluate(self.sk, q) for q in blinded]

    def tags(self):
        out = [_tag(oprf.evaluate(self.sk, y)) for y in self.items]
        secrets.SystemRandom().shuffle(out)
        return out


class PSIReceiver:
    def __init__(self, items, max_queries: int):
        self.items = list(dict.fromkeys(items))
        if not self.items or len(self.items) > max_queries:
            raise ValueError('need 1..max_queries receiver items')
        self._inputs = [encode(x) for x in self.items] + \
                       [b'\0pad' + secrets.token_bytes(16) for _ in range(max_queries - len(self.items))]
        self._blinds = []

    def queries(self):
        out = []
        for data in self._inputs:
            r, q = oprf.blind(data)
            self._blinds.append(r)
            out.append(q)
        return out

    def intersect(self, evaluated, tags):
        tags = set(tags)
        return [x for x, data, r, e in zip(self.items, self._inputs, self._blinds, evaluated)
                if _tag(oprf.finalize(data, r, e)) in tags]


def psi(receiver_items, sender_items, max_queries=None):
    """Local driver. Returns what the receiver learns: X ∩ Y in its own order."""
    max_queries = max_queries or len(set(receiver_items))
    s, r = PSISender(sender_items), PSIReceiver(receiver_items, max_queries)
    return r.intersect(s.respond(r.queries()), s.tags())


def psi_with_keys(receiver_items, sender_payloads: dict, max_queries=None):
    """Key version: sender_payloads {item: key bytes}. Returns {x: key} for x in X ∩ Y."""
    max_queries = max_queries or len(set(receiver_items))
    server = oprf.PayloadServer(sender_payloads)
    client = oprf.PayloadClient(receiver_items, max_queries)
    return client.recover(server.respond(client.queries()), server.table())
