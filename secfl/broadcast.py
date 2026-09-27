"""Phase 3a: Aggregator posts AES-GCM(K_L, G_theta_L) on the BB; holders of K_L decrypt.

AAD binds (round, label), so a ciphertext cannot be replayed into another round
or swapped between labels. Everyone reads the whole board (bb.read()), so the
Aggregator does not see which generators a client fetches.
"""
import io
import secrets

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def topic(round_index, label):
    return f'gen/{round_index}/{label}'


def pack_state(state: dict) -> bytes:
    """{name: array} -> bytes; no pickle, so decoding cannot execute code."""
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in state.items()})
    return buf.getvalue()


def unpack_state(data: bytes) -> dict:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def seal(key: bytes, round_index, label, state: dict) -> bytes:
    nonce = secrets.token_bytes(12)
    return nonce + AESGCM(key).encrypt(nonce, pack_state(state), topic(round_index, label).encode())


def unseal(key: bytes, round_index, label, blob: bytes):
    """State dict, or None if key is not K_L (client does not hold L) or blob was tampered."""
    try:
        plain = AESGCM(key).decrypt(blob[:12], blob[12:], topic(round_index, label).encode())
    except InvalidTag:
        return None
    return unpack_state(plain)


def post_generators(bb, keys: dict, round_index, states: dict, author='aggregator'):
    """keys {slot: K} for ALL slots, states {slot: state} for active ones.

    Every slot gets a post each round; inactive slots carry an all-zero state of the
    same shape under their own key, so readers cannot count the active labels.
    """
    if not states or not set(states) <= set(keys):
        raise ValueError('need a key for every posted slot')
    template = next(iter(states.values()))
    dummy = {k: np.zeros_like(np.asarray(v)) for k, v in template.items()}
    for slot in keys:
        bb.post(author, topic(round_index, slot), seal(keys[slot], round_index, slot, states.get(slot, dummy)))


def fetch_generators(bb, my_keys: dict, round_index, author='aggregator'):
    """my_keys {L: key from key release}. Returns {L: state} for labels it can open."""
    board = {e.topic: e.payload for e in bb.read() if e.author == author}   # full read
    out = {}
    for label, key in my_keys.items():
        blob = board.get(topic(round_index, label))
        if blob is not None:
            state = unseal(key, round_index, label, blob)
            if state is not None:
                out[label] = state
    return out
