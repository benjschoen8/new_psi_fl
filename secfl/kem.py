"""KEM downlink: pk_s encrypts a fresh k, k encrypts G_s (DHKEM on ristretto255 + AES-GCM).

Per slot s the union gives pk_s (public, on the BB) and sk_s (holders only).
  Encaps: e random, E = e*B, ss = e*pk_s, k = HKDF(ss, salt = E || pk_s, info = topic)
  Decaps: ss = sk_s*E
  Payload: AES-GCM(k, pack_state(G_s)), AAD = topic (round, slot).
Every slot in [0, M) gets a ciphertext each round (inactive slots: zero state of the same
shape), so readers cannot count active labels. Structure follows HPKE (RFC 9180) base mode;
ponytail: HPKE has no registered ristretto255 KEM, so this is a DHKEM-style construction.
"""
import secrets

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from . import ristretto as rg
from .broadcast import pack_state, unpack_state, topic


def _kdf(ss: bytes, enc: bytes, pk: bytes, info: bytes) -> bytes:
    return HKDF(hashes.SHA256(), 16, salt=enc + pk, info=b'label-kem/' + info).derive(ss)


def encaps(pk: bytes, info: bytes):
    e = rg.random_scalar()
    enc = (rg.BASE * e).encode()
    return enc, _kdf((rg.Point.decode(pk) * e).encode(), enc, pk, info)


def decaps(sk: int, enc: bytes, pk: bytes, info: bytes) -> bytes:
    return _kdf((rg.Point.decode(enc) * sk).encode(), enc, pk, info)


def seal(pk: bytes, round_index, slot, state: dict) -> bytes:
    info = topic(round_index, slot).encode()
    enc, k = encaps(pk, info)
    nonce = secrets.token_bytes(12)
    return enc + nonce + AESGCM(k).encrypt(nonce, pack_state(state), info)


def unseal(sk: int, pk: bytes, round_index, slot, blob: bytes):
    """State dict, or None if sk is not the slot's key or the blob was altered."""
    info = topic(round_index, slot).encode()
    enc, nonce, ct = blob[:32], blob[32:44], blob[44:]
    try:
        return unpack_state(AESGCM(decaps(sk, enc, pk, info)).decrypt(nonce, ct, info))
    except (InvalidTag, ValueError):
        return None


def post_generators(bb, pks: list, round_index, states: dict, author='aggregator'):
    """pks: pk of every slot 0..M-1; states {slot: state} for active slots."""
    if not states or not all(0 <= s < len(pks) for s in states):
        raise ValueError('active slots must lie in [0, M)')
    template = next(iter(states.values()))
    dummy = {k: np.zeros_like(np.asarray(v)) for k, v in template.items()}
    for slot, pk in enumerate(pks):
        bb.post(author, topic(round_index, slot), seal(pk, round_index, slot, states.get(slot, dummy)))


def fetch_generators(bb, my_keys: dict, pks: list, round_index, author='aggregator'):
    """my_keys {slot: sk}. Reads the whole board, returns {slot: state} it can open."""
    board = {e.topic: e.payload for e in bb.read() if e.author == author}
    out = {}
    for slot, sk in my_keys.items():
        blob = board.get(topic(round_index, slot))
        state = None if blob is None else unseal(sk, pks[slot], round_index, slot, blob)
        if state is not None:
            out[slot] = state
    return out
