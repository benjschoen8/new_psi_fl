"""Secure aggregation with dropout tolerance (Bonawitz et al., CCS 2017), semi-honest.

Four client messages per aggregation, driven explicitly (no networking here):
  0 advertise  -> (c_pk, s_pk)                 two ephemeral X25519 keys
  1 share      -> {v: Enc_{c_uv}(share_v(s_sk), share_v(b))}   Shamir, threshold t
  2 masked     -> y_u = x_u + PRG(b_u) + sum_{v>u} PRG(s_uv) - sum_{v<u} PRG(s_uv)   (mod 2^k)
  3 unmask     -> shares of b_u for survivors, of s_sk for dropped clients
The server learns sum_{u in U2} x_u and the survivor set U2, nothing else about x_u.
A client refuses to reveal both kinds of share for the same peer.
"""
import secrets

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .secagg_primitives import pairwise_seed, prg, shamir_share, shamir_reconstruct, word_dtype

RAW = dict(encoding=serialization.Encoding.Raw, format=serialization.PrivateFormat.Raw,
           encryption_algorithm=serialization.NoEncryption())


def _pub(sk):
    return sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _reduce(v, bits):
    """v (word_dtype(bits), wrapped mod 2^width) -> mod 2^bits, in place."""
    if bits < 8 * v.itemsize:
        np.bitwise_and(v, v.dtype.type((1 << bits) - 1), out=v)
    return v


def _pair_masks(s_sk, my_id, peer_pks, session, length, bits, ids_order):
    """sum_{v>u} PRG(s_uv) - sum_{v<u} PRG(s_uv) mod 2^bits, over the given peers (in place, in
    word_dtype(bits): no temporaries, 2-byte words for 16-bit sums)."""
    total = np.zeros(length, word_dtype(bits))
    for v, pk in peer_pks.items():
        seed = pairwise_seed(s_sk.exchange(X25519PublicKey.from_public_bytes(pk)), my_id, v, session + b'/mask')
        (np.add if ids_order[my_id] < ids_order[v] else np.subtract)(total, prg(seed, length, bits), out=total)
    return _reduce(total, bits)


class SecAggClient:
    def __init__(self, client_id, ids, threshold, length, modulus_bits=32, session=b''):
        self.id, self.ids = client_id, list(ids)
        if client_id not in self.ids or not 1 <= threshold <= len(self.ids):
            raise ValueError('client must be in ids and 1 <= threshold <= n')
        self.t, self.length, self.bits, self.session = threshold, length, modulus_bits, session
        self.order = {u: k for k, u in enumerate(self.ids)}
        self._c_sk, self._s_sk = X25519PrivateKey.generate(), X25519PrivateKey.generate()
        self._b = secrets.token_bytes(32)
        self._received = {}

    def x_of(self, u):                          # Shamir evaluation point of client u
        return self.order[u] + 1

    # round 0
    def advertise(self):
        return _pub(self._c_sk), _pub(self._s_sk)

    def _channel(self, peer_c_pk, u, v):
        key = pairwise_seed(self._c_sk.exchange(X25519PublicKey.from_public_bytes(peer_c_pk)),
                            u, v, self.session + b'/channel')
        return AESGCM(key)

    # round 1
    def share(self, keys):
        """keys: {u: (c_pk, s_pk)} for U1 (clients that advertised). Returns {v: ciphertext}."""
        self.U1 = dict(keys)
        if len(self.U1) < self.t or self.id not in self.U1:
            raise ValueError('fewer than t clients advertised')
        peers = [v for v in self.U1]
        s_shares = shamir_share(self._s_sk.private_bytes(**RAW), self.t, [self.x_of(v) for v in peers])
        b_shares = shamir_share(self._b, self.t, [self.x_of(v) for v in peers])
        out = {}
        for v, (x, sy), (_, by) in zip(peers, s_shares, b_shares):
            plain = x.to_bytes(4, 'big') + sy.to_bytes(66, 'big') + by.to_bytes(66, 'big')
            nonce = secrets.token_bytes(12)
            aad = f'{self.id}->{v}'.encode()
            out[v] = nonce + self._channel(self.U1[v][0], self.id, v).encrypt(nonce, plain, aad)
        return out

    # round 2
    def masked_input(self, x, ciphertexts):
        """x: uint64 vector already in Z_{2^k}. ciphertexts: {u: ct addressed to me} from U2."""
        x = np.asarray(x, np.uint64)
        if x.shape != (self.length,):
            raise ValueError('input length mismatch')
        self._received = dict(ciphertexts)
        self.U2 = set(ciphertexts)
        if len(self.U2) < self.t:
            raise ValueError('fewer than t clients shared keys')
        peers = {v: self.U1[v][1] for v in self.U2 if v != self.id}
        y = x.astype(word_dtype(self.bits))                       # x < 2^bits: exact
        np.add(y, prg(self._b, self.length, self.bits), out=y)
        np.add(y, _pair_masks(self._s_sk, self.id, peers, self.session, self.length, self.bits, self.order), out=y)
        return _reduce(y, self.bits)

    # round 3
    def unmask(self, survivors):
        """survivors: U3 (sent masked input). Reveal b-shares for U3, s-shares for U2 \\ U3."""
        survivors = set(survivors)
        if not survivors <= self.U2 or len(survivors) < self.t:
            raise ValueError('survivor set must be a subset of U2 with at least t clients')
        out = {'b': {}, 's': {}}
        for u in self.U2:
            ct = self._received[u]
            plain = self._channel(self.U1[u][0], u, self.id).decrypt(ct[:12], ct[12:], f'{u}->{self.id}'.encode())
            x = int.from_bytes(plain[:4], 'big')
            if x != self.x_of(self.id):
                raise ValueError('share addressed to a different client')
            s_y, b_y = int.from_bytes(plain[4:70], 'big'), int.from_bytes(plain[70:], 'big')
            # never both kinds for one peer: that would expose x_u
            if u in survivors:
                out['b'][u] = (x, b_y)
            else:
                out['s'][u] = (x, s_y)
        return out


class SecAggServer:
    def __init__(self, ids, threshold, length, modulus_bits=32, session=b''):
        self.ids, self.t, self.length, self.bits, self.session = list(ids), threshold, length, modulus_bits, session
        self.order = {u: k for k, u in enumerate(self.ids)}

    def collect_keys(self, adverts):
        if len(adverts) < self.t:
            raise ValueError('abort: fewer than t clients advertised')
        self.U1 = dict(adverts)
        return self.U1

    def route_shares(self, shares_by_sender):
        """{u: {v: ct}} -> {v: {u: ct}}; U2 = senders."""
        if len(shares_by_sender) < self.t:
            raise ValueError('abort: fewer than t clients shared')
        self.U2 = set(shares_by_sender)
        inbox = {v: {} for v in self.U1}
        for u, cts in shares_by_sender.items():
            for v, ct in cts.items():
                inbox[v][u] = ct
        return {v: box for v, box in inbox.items() if v in self.U2}

    def collect_masked(self, masked):
        if len(masked) < self.t:
            raise ValueError('abort: fewer than t masked inputs')
        self.U3 = set(masked)
        self._masked = {u: np.asarray(y).astype(word_dtype(self.bits), copy=False) for u, y in masked.items()}
        return sorted(self.U3, key=self.order.get)

    def aggregate(self, reveals):
        """reveals: {v: client.unmask output} from at least t clients of U3."""
        if len(reveals) < self.t:
            raise ValueError('abort: fewer than t unmasking responses')
        total = np.zeros(self.length, word_dtype(self.bits))
        for y in self._masked.values():
            np.add(total, y, out=total)
        for u in self.U3:
            b = shamir_reconstruct([r['b'][u] for r in reveals.values()][: self.t])
            np.subtract(total, prg(b, self.length, self.bits), out=total)
        for u in self.U2 - self.U3:                   # dropped after sharing: cancel their pair masks
            s_sk = X25519PrivateKey.from_private_bytes(
                shamir_reconstruct([r['s'][u] for r in reveals.values()][: self.t]))
            peers = {v: self.U1[v][1] for v in self.U3}
            # survivors added +/- PRG(s_uv) with u; recompute u's view and add it back
            np.add(total, _pair_masks(s_sk, u, peers, self.session, self.length, self.bits, self.order), out=total)
        return _reduce(total, self.bits).astype(np.uint64)                # callers get uint64, as before


def run_secagg(inputs, threshold, modulus_bits=32, session=b'', drop_before_masking=(), drop_before_unmask=(),
               workers=1, stats=None):
    """Local driver. inputs {id: uint64 vector}. Returns (sum over U3, U3).

    stats: optional dict; bytes on the wire are ADDED to it (summed over all clients, star topology
    through the server): payload_up (masked inputs, ceil(k/8) bytes per word), control_up
    (keys, encrypted shares, unmask shares), control_down (key list, routed shares, survivor list).

    workers > 1 computes the clients' masked inputs (the PRG-heavy part) in parallel threads;
    ChaCha20 and numpy release the GIL, so this scales with cores.
    """
    ids = list(inputs)
    length = len(next(iter(inputs.values())))
    clients = {u: SecAggClient(u, ids, threshold, length, modulus_bits, session) for u in ids}
    server = SecAggServer(ids, threshold, length, modulus_bits, session)
    U1 = server.collect_keys({u: c.advertise() for u, c in clients.items()})
    inbox = server.route_shares({u: c.share(U1) for u, c in clients.items()})
    senders = [u for u in inbox if u not in drop_before_masking]
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as pool:
            masked = dict(zip(senders, pool.map(lambda u: clients[u].masked_input(inputs[u], inbox[u]), senders)))
    else:
        masked = {u: clients[u].masked_input(inputs[u], inbox[u]) for u in senders}
    U3 = server.collect_masked(masked)
    reveals = {u: clients[u].unmask(U3) for u in U3 if u not in drop_before_unmask}
    if stats is not None:
        n1, shares = len(U1), [ct for u in inbox for ct in inbox[u].values()]
        up = n1 * 64 + sum(map(len, shares)) + sum(70 * (len(r['b']) + len(r['s'])) for r in reveals.values())
        down = n1 * n1 * 64 + sum(map(len, shares)) + len(masked) * 4 * len(U3)
        for k, v in dict(payload_up=len(masked) * length * -(-modulus_bits // 8), control_up=up,
                         control_down=down).items():
            stats[k] = stats.get(k, 0) + v
    return server.aggregate(reveals), U3
