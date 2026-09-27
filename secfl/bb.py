"""Phase 0: bulletin board, public parameters, X25519 identity keys."""
from dataclasses import dataclass
import hashlib

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey


@dataclass(frozen=True)
class Entry:
    index: int
    author: str
    topic: str
    payload: bytes
    prev: bytes      # hash of the previous entry: the board is a hash chain


class BulletinBoard:
    """Append-only, publicly readable.

    read() always returns the whole board, so who reads what is not observable.
    head() is a hash-chain digest: parties compare it to detect a forked view.
    """

    def __init__(self):
        self._entries = []

    def post(self, author, topic, payload: bytes) -> int:
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError('payload must be bytes')
        entry = Entry(len(self._entries), str(author), str(topic), bytes(payload), self.head())
        self._entries.append(entry)
        return entry.index

    def read(self):
        return tuple(self._entries)

    def head(self) -> bytes:
        if not self._entries:
            return b'\0' * 32
        e = self._entries[-1]
        return hashlib.sha256(b'|'.join((e.prev, str(e.index).encode(), e.author.encode(),
                                         e.topic.encode(), e.payload))).digest()

    def verify(self) -> bool:
        """Recompute the chain; False if any entry was altered."""
        prev = b'\0' * 32
        for e in self._entries:
            if e.prev != prev:
                return False
            prev = hashlib.sha256(b'|'.join((e.prev, str(e.index).encode(), e.author.encode(),
                                             e.topic.encode(), e.payload))).digest()
        return prev == self.head()


@dataclass(frozen=True)
class PublicParams:
    labels: tuple                # global label set, ordered: index = label id L
    latent_dim: int = 256        # shared latent space d
    modulus_bits: int = 32       # uploads live in Z_{2^32}
    frac_bits: int = 16          # fixed-point precision
    clip: float = 1.0            # L2 clip bound per client upload
    threshold: int = 2           # activation threshold t on |Group_L|

    def __post_init__(self):
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError('labels must be nonempty and unique')
        if self.latent_dim < 1 or not 8 <= self.modulus_bits <= 64 or self.clip <= 0:
            raise ValueError('invalid latent_dim, modulus_bits or clip')
        if not 0 <= self.frac_bits < self.modulus_bits - 1 or self.threshold < 1:
            raise ValueError('invalid frac_bits or threshold')


IDENTITY = 'identity/x25519'


class Identity:
    """Long-term X25519 key of one client; the public half lives on the BB."""

    def __init__(self, client_id, private_key=None):
        self.client_id = str(client_id)
        self.private_key = private_key or X25519PrivateKey.generate()

    @property
    def public_bytes(self) -> bytes:
        return self.private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def publish(self, bb: BulletinBoard) -> int:
        return bb.post(self.client_id, IDENTITY, self.public_bytes)

    def exchange(self, peer_public: bytes) -> bytes:
        return self.private_key.exchange(X25519PublicKey.from_public_bytes(peer_public))


def lookup_identity(bb: BulletinBoard, client_id) -> bytes:
    keys = {e.payload for e in bb.read() if e.topic == IDENTITY and e.author == str(client_id)}
    if len(keys) != 1:
        raise LookupError(f'client {client_id}: expected exactly one identity key, found {len(keys)}')
    return keys.pop()
