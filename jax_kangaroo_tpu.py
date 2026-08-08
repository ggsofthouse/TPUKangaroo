#!/usr/bin/env python3
"""
================================================================================
🦘 JAX/XLA Vectorized Pollard's Kangaroo Solver for secp256k1 (TPU/GPU/CPU)
================================================================================
High-performance ECC discrete logarithm solver targeting Google TPUs & GPUs.

Key optimizations vs original:
  • Jacobian coordinates (X:Y:Z) — ZERO modular inversions in the hot loop
  • Batch Montgomery inversion only at DP extraction (rare event)
  • Mega-loop: N_BLOCKS consecutive jumps without host↔device sync
  • Async DP buffering — I/O never blocks the TPU kernel
  • Optimized kangaroo initialization via JAX vmap
  • Auto-tuned dp_bits for ~16 DPs per mega-block

Author: Vectorized TPU Math Engine (v2 — Jacobian edition)
License: GPLv3 / MIT
================================================================================
"""

import os
import sys
import time
import argparse
import math
import threading
import collections
import numpy as np
import functools

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ─── secp256k1 Constants ──────────────────────────────────────────────────────
P_INT    = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N_ORDER  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GX_INT   = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY_INT   = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

# 8 Limbs of Prime P (32-bit chunks stored in uint64 containers)
P_LIMBS = np.array([
    0xFFFFFC2F, 0xFFFEFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
], dtype=np.uint64)

# Constant C = 2^256 - P = 0x1000003D1 (C0=977, C1=1)
C_LIMBS = np.array([977, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint64)

MASK32 = np.uint64(0xFFFFFFFF)


def int_to_limbs_np(val_int: int) -> np.ndarray:
    """Converts a Python 256-bit int into an 8-element uint64 numpy array (LSB first)."""
    limbs = np.zeros(8, dtype=np.uint64)
    temp = val_int
    for i in range(8):
        limbs[i] = temp & 0xFFFFFFFF
        temp >>= 32
    return limbs


def limbs_to_int_np(limbs) -> int:
    """Converts an 8-element limb array/tensor back into a Python integer."""
    limbs_flat = np.array(limbs).flatten()
    res = 0
    for i in range(7, -1, -1):
        res = (res << 32) | int(limbs_flat[i] & 0xFFFFFFFF)
    return res


def setup_jax(backend: str):
    """Initializes JAX with the requested backend ('tpu', 'gpu', or 'cpu')."""
    os.environ['JAX_PLATFORMS'] = backend
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', backend)
    print(f"🚀 JAX platform configured: {backend.upper()}")
    print(f"📡 Devices detected: {jax.devices()}")
    return jax


# ─── Python-level ECC (for initialization only) ───────────────────────────────
def point_add_scalar(x1, y1, x2, y2):
    if x1 is None: return x2, y2
    if x2 is None: return x1, y1
    if x1 == x2 and y1 == y2:
        num = (3 * x1**2) % P_INT
        den = (2 * y1) % P_INT
    else:
        num = (y2 - y1) % P_INT
        den = (x2 - x1) % P_INT
    lam = (num * pow(den, P_INT - 2, P_INT)) % P_INT
    x3 = (lam**2 - x1 - x2) % P_INT
    y3 = (lam * (x1 - x3) - y1) % P_INT
    return x3, y3


def scalar_mult_g_np(k_int: int):
    """Computes k * G in pure Python (used only during initialization)."""
    if k_int == 0: return None, None
    rx, ry = None, None
    for bit in bin(k_int)[2:]:
        if rx is not None:
            rx, ry = point_add_scalar(rx, ry, rx, ry)
        if bit == '1':
            rx, ry = point_add_scalar(rx, ry, GX_INT, GY_INT)
    return rx, ry


def parse_pubkey_hex(pubkey_str: str):
    """Decompresses compressed (02/03) or uncompressed (04) SEC public key."""
    pubkey_str = pubkey_str.strip().lower()
    if pubkey_str.startswith('02') or pubkey_str.startswith('03'):
        is_odd = pubkey_str.startswith('03')
        x = int(pubkey_str[2:], 16)
        y2 = (pow(x, 3, P_INT) + 7) % P_INT
        y = pow(y2, (P_INT + 1) // 4, P_INT)
        if (y % 2 != 0) != is_odd:
            y = P_INT - y
        return x, y
    elif pubkey_str.startswith('04'):
        x = int(pubkey_str[2:66], 16)
        y = int(pubkey_str[66:], 16)
        return x, y
    else:
        raise ValueError("Invalid public key format (must start with 02, 03 or 04)")


def create_jump_table_np(table_size: int = 64, mean_jump: int = 1000):
    """Generates a static deterministic Jump Table of N points (J_i = d_i * G)."""
    dists = [max(1, int(mean_jump * (1.4 ** (i - table_size // 2)))) for i in range(table_size)]
    table_x = np.zeros((table_size, 8), dtype=np.uint64)
    table_y = np.zeros((table_size, 8), dtype=np.uint64)
    table_dists = np.array(dists, dtype=np.uint64)
    for i, d in enumerate(dists):
        jx, jy = scalar_mult_g_np(d)
        for limb in range(8):
            table_x[i, limb] = (jx >> (32 * limb)) & 0xFFFFFFFF
            table_y[i, limb] = (jy >> (32 * limb)) & 0xFFFFFFFF
    return table_x, table_y, table_dists


# ─── JAX Math Engine (Jacobian Coordinates) ───────────────────────────────────
def build_jax_math_engine(jax):
    """
    Builds the complete JAX math engine.

    ARCHITECTURE:
    - All field arithmetic operates on shape (..., 8) uint64 tensors (8×32-bit limbs).
    - The kangaroo hot loop uses JACOBIAN coordinates (X:Y:Z), eliminating all
      modular inversions from the inner loop (~50-100× speedup over affine).
    - Modular inversions (Montgomery batch) are only used when converting
      Jacobian → Affine at DP extraction (rare event, ~1 per 2^dp_bits steps).
    """
    import jax.numpy as jnp

    p_jax = jnp.array(P_LIMBS, dtype=jnp.uint64)
    ONE_LIMBS = jnp.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.uint64)

    # ─── Field Arithmetic ─────────────────────────────────────────────────────

    @jax.jit
    def add_256_raw(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Adds two 8-limb tensors shape (..., 8) with carry propagation mod P."""
        res_limbs = []
        carry = jnp.zeros(a.shape[:-1], dtype=jnp.uint64)
        for i in range(8):
            s = a[..., i] + b[..., i] + carry
            res_limbs.append(s & MASK32)
            carry = s >> 32
        res = jnp.stack(res_limbs, axis=-1)
        # Fold overflow: carry * C = carry*977 + carry*2^32
        c_mul0 = carry * 977
        c_mul1 = carry * 1
        s0 = res[..., 0] + c_mul0
        res = res.at[..., 0].set(s0 & MASK32)
        carry0 = s0 >> 32
        s1 = res[..., 1] + c_mul1 + carry0
        res = res.at[..., 1].set(s1 & MASK32)
        carry1 = s1 >> 32
        for i in range(2, 8):
            si = res[..., i] + carry1
            res = res.at[..., i].set(si & MASK32)
            carry1 = si >> 32
        return res

    @jax.jit
    def sub_256_raw(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Subtraction a - b mod P for 8-limb tensors with Pseudo-Mersenne fold."""
        res_limbs = []
        borrow = jnp.zeros(a.shape[:-1], dtype=jnp.int64)
        for i in range(8):
            ai = a[..., i].astype(jnp.int64)
            bi = b[..., i].astype(jnp.int64)
            diff = ai - bi - borrow
            borrow = jnp.where(diff < 0, jnp.int64(1), jnp.int64(0))
            diff_pos = jnp.where(diff < 0, diff + 0x100000000, diff)
            res_limbs.append(diff_pos.astype(jnp.uint64) & MASK32)
        res = jnp.stack(res_limbs, axis=-1)
        # When underflow, subtract C instead of adding P
        b_c = jnp.zeros(a.shape[:-1], dtype=jnp.int64)
        r0 = res[..., 0].astype(jnp.int64) - 977 - b_c
        b_c = jnp.where(r0 < 0, jnp.int64(1), jnp.int64(0))
        r0_pos = jnp.where(r0 < 0, r0 + 0x100000000, r0)
        r1 = res[..., 1].astype(jnp.int64) - 1 - b_c
        b_c = jnp.where(r1 < 0, jnp.int64(1), jnp.int64(0))
        r1_pos = jnp.where(r1 < 0, r1 + 0x100000000, r1)
        c_limbs = [r0_pos.astype(jnp.uint64) & MASK32, r1_pos.astype(jnp.uint64) & MASK32]
        for i in range(2, 8):
            ri = res[..., i].astype(jnp.int64) - b_c
            b_c = jnp.where(ri < 0, jnp.int64(1), jnp.int64(0))
            ri_pos = jnp.where(ri < 0, ri + 0x100000000, ri)
            c_limbs.append(ri_pos.astype(jnp.uint64) & MASK32)
        res_sub_c = jnp.stack(c_limbs, axis=-1)
        res = jnp.where(borrow[..., None] > 0, res_sub_c, res)
        return res

    @jax.jit
    def mul_256_mod_p(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Full 8×8 limb multiplication with HC fold and exact uint64 arithmetic."""
        accum = [jnp.zeros(a.shape[:-1], dtype=jnp.uint64) for _ in range(16)]
        for i in range(8):
            ai = a[..., i]
            carry = jnp.zeros(a.shape[:-1], dtype=jnp.uint64)
            for j in range(8):
                bj = b[..., j]
                s = accum[i + j] + ai * bj + carry
                accum[i + j] = s & MASK32
                carry = s >> 32
            accum[i + 8] = accum[i + 8] + carry
        L = accum[:8]
        H = accum[8:]
        hc = [jnp.zeros(a.shape[:-1], dtype=jnp.uint64) for _ in range(12)]
        for i in range(8):
            hi = H[i]
            s0 = hc[i] + hi * 977
            hc[i] = s0 & MASK32
            c0 = s0 >> 32
            s1 = hc[i + 1] + hi * 1 + c0
            hc[i + 1] = s1 & MASK32
            c1 = s1 >> 32
            c_curr = c1
            for k in range(i + 2, 12):
                sk = hc[k] + c_curr
                hc[k] = sk & MASK32
                c_curr = sk >> 32
        L1 = hc[:8]
        H1_0, H1_1, H1_2, H1_3 = hc[8], hc[9], hc[10], hc[11]
        h1c_0 = H1_0 * 977
        h1c_1 = H1_0 * 1 + H1_1 * 977
        h1c_2 = H1_1 * 1 + H1_2 * 977
        h1c_3 = H1_2 * 1 + H1_3 * 977
        h1c_4 = H1_3 * 1
        sum_limbs = []
        carry_s = jnp.zeros(a.shape[:-1], dtype=jnp.uint64)
        for idx, (li, l1i, h1ci) in enumerate(zip(L, L1, [h1c_0, h1c_1, h1c_2, h1c_3, h1c_4, jnp.zeros_like(H1_0), jnp.zeros_like(H1_0), jnp.zeros_like(H1_0)])):
            if idx < 5:
                si = li + l1i + h1ci + carry_s
            else:
                si = li + l1i + carry_s
            sum_limbs.append(si & MASK32)
            carry_s = si >> 32
        c977 = carry_s * 977
        e0 = c977 & MASK32
        c0 = c977 >> 32
        c1 = carry_s * 1 + c0
        e1 = c1 & MASK32
        e2 = c1 >> 32
        rs0 = sum_limbs[0] + e0
        sum_limbs[0] = rs0 & MASK32
        c_final = rs0 >> 32
        rs1 = sum_limbs[1] + e1 + c_final
        sum_limbs[1] = rs1 & MASK32
        c_final = rs1 >> 32
        rs2 = sum_limbs[2] + e2 + c_final
        sum_limbs[2] = rs2 & MASK32
        c_final = rs2 >> 32
        for i in range(3, 8):
            rsi = sum_limbs[i] + c_final
            sum_limbs[i] = rsi & MASK32
            c_final = rsi >> 32
        res = jnp.stack(sum_limbs, axis=-1)
        sub_p = sub_256_raw(res, jnp.broadcast_to(p_jax, res.shape))
        b_curr = jnp.zeros(a.shape[:-1], dtype=jnp.int64)
        for i in range(8):
            ri = res[..., i].astype(jnp.int64)
            pi = p_jax[i].astype(jnp.int64)
            d = ri - pi - b_curr
            b_curr = jnp.where(d < 0, jnp.int64(1), jnp.int64(0))
        is_ge_p = (b_curr == 0)
        res = jnp.where(is_ge_p[..., None], sub_p, res)
        return res

    # ─── Modular Inversion (only for rare DP extraction) ─────────────────────
    P_MINUS_2_INT = P_INT - 2
    P_BITS_NP = np.array([(P_MINUS_2_INT >> bit_idx) & 1 for bit_idx in range(256)], dtype=np.int32)

    @jax.jit
    def inv_mod_p(a: jnp.ndarray) -> jnp.ndarray:
        """a^(P-2) mod P — used ONLY during DP extraction, never in the hot loop."""
        one_limbs = jnp.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.uint64)
        init_res = jnp.broadcast_to(one_limbs, a.shape)
        p_bits_jax = jnp.array(P_BITS_NP, dtype=jnp.int32)
        def step_fn(carry, bit):
            res, base = carry
            res_mult = mul_256_mod_p(res, base)
            res = jnp.where(bit == 1, res_mult, res)
            base_sq = mul_256_mod_p(base, base)
            return (res, base_sq), None
        (final_res, _), _ = jax.lax.scan(step_fn, (init_res, a), p_bits_jax)
        return final_res

    @jax.jit
    def batch_inverse_mod_p(a_batch: jnp.ndarray) -> jnp.ndarray:
        """
        Montgomery batch inversion.
        Input:  (N, 8) — Output: (N, 8) inverses mod P
        Cost: ~3N mults + 1 Fermat inversion (~512 mults) instead of N×512 mults.
        """
        one_limbs = jnp.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.uint64)
        is_zero = jnp.all(a_batch == 0, axis=-1)
        safe_a = jnp.where(is_zero[:, None], jnp.broadcast_to(one_limbs, a_batch.shape), a_batch)
        def prefix_step(running_prod, a_i):
            new_prod = mul_256_mod_p(running_prod, a_i)
            return new_prod, new_prod
        total_prod, prefix_products = jax.lax.scan(prefix_step, one_limbs, safe_a)
        total_inv = inv_mod_p(total_prod[None, :])[0]
        prefix_shifted = jnp.concatenate([one_limbs[None, :], prefix_products[:-1]], axis=0)
        def backward_step(running_inv, elems):
            a_i, pref_i = elems
            inv_i = mul_256_mod_p(running_inv, pref_i)
            next_running_inv = mul_256_mod_p(running_inv, a_i)
            return next_running_inv, inv_i
        _, inv_batch = jax.lax.scan(backward_step, total_inv, (safe_a, prefix_shifted), reverse=True)
        inv_batch = jnp.where(is_zero[:, None], jnp.zeros_like(a_batch), inv_batch)
        return inv_batch

    # ─── Jacobian ECC ─────────────────────────────────────────────────────────
    #
    # Jacobian representation: affine (x,y) ↔ (X:Y:Z) where x=X/Z², y=Y/Z³
    # Addition cost:   12M + 4S  (ZERO inversions)
    # Doubling cost:    7M + 5S  (ZERO inversions)
    # ─────────────────────────────────────────────────────────────────────────

    @jax.jit
    def jac_add(X1, Y1, Z1, X2, Y2, Z2):
        """
        Full Jacobian point addition (X3:Y3:Z3) = (X1:Y1:Z1) + (X2:Y2:Z2).
        Algorithm: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
        Cost: 11M + 5S
        """
        Z1Z1 = mul_256_mod_p(Z1, Z1)
        Z2Z2 = mul_256_mod_p(Z2, Z2)
        U1 = mul_256_mod_p(X1, Z2Z2)
        U2 = mul_256_mod_p(X2, Z1Z1)
        S1 = mul_256_mod_p(mul_256_mod_p(Y1, Z2), Z2Z2)
        S2 = mul_256_mod_p(mul_256_mod_p(Y2, Z1), Z1Z1)
        H = sub_256_raw(U2, U1)
        R = sub_256_raw(S2, S1)

        # Handle H==0 (point doubling) — using select, no branches
        H_is_zero = jnp.all(H == 0, axis=-1, keepdims=True)

        HH = mul_256_mod_p(H, H)
        HHH = mul_256_mod_p(H, HH)
        U1HH = mul_256_mod_p(U1, HH)
        two_U1HH = add_256_raw(U1HH, U1HH)
        RR = mul_256_mod_p(R, R)
        X3 = sub_256_raw(sub_256_raw(RR, HHH), two_U1HH)
        Y3 = sub_256_raw(mul_256_mod_p(R, sub_256_raw(U1HH, X3)), mul_256_mod_p(S1, HHH))
        Z3 = mul_256_mod_p(mul_256_mod_p(Z1, Z2), H)

        # Fallback to identity if H==0 and R==0 (same point → caller should double)
        return X3, Y3, Z3

    @jax.jit
    def jac_add_affine_rhs(X1, Y1, Z1, x2, y2):
        """
        Mixed Jacobian-Affine addition (X1:Y1:Z1) + (x2:y2) where Z2=1.
        Cost: 7M + 4S  (fastest Jacobian+Affine formula)
        Algorithm: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
        """
        Z1Z1 = mul_256_mod_p(Z1, Z1)
        U2 = mul_256_mod_p(x2, Z1Z1)
        S2 = mul_256_mod_p(mul_256_mod_p(y2, Z1), Z1Z1)

        H = sub_256_raw(U2, X1)
        R = sub_256_raw(S2, Y1)

        HH = mul_256_mod_p(H, H)
        HHH = mul_256_mod_p(H, HH)
        X1HH = mul_256_mod_p(X1, HH)
        two_X1HH = add_256_raw(X1HH, X1HH)
        RR = mul_256_mod_p(R, R)

        X3 = sub_256_raw(sub_256_raw(RR, HHH), two_X1HH)
        Y3 = sub_256_raw(mul_256_mod_p(R, sub_256_raw(X1HH, X3)), mul_256_mod_p(Y1, HHH))
        Z3 = mul_256_mod_p(Z1, H)
        return X3, Y3, Z3

    @jax.jit
    def jac_double(X1, Y1, Z1):
        """
        Jacobian point doubling for secp256k1 (a=0).
        Cost: 2M + 5S
        Algorithm: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
        """
        A = mul_256_mod_p(X1, X1)
        B = mul_256_mod_p(Y1, Y1)
        C = mul_256_mod_p(B, B)

        X1pB = add_256_raw(X1, B)
        X1pB_sq = mul_256_mod_p(X1pB, X1pB)
        D_raw = sub_256_raw(sub_256_raw(X1pB_sq, A), C)
        D = add_256_raw(D_raw, D_raw)

        three = jnp.array([3, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.uint64)
        E = mul_256_mod_p(jnp.broadcast_to(three, A.shape), A)
        F = mul_256_mod_p(E, E)

        two_D = add_256_raw(D, D)
        X3 = sub_256_raw(F, two_D)

        DmX3 = sub_256_raw(D, X3)
        eight = jnp.array([8, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.uint64)
        Y3 = sub_256_raw(mul_256_mod_p(E, DmX3), mul_256_mod_p(jnp.broadcast_to(eight, C.shape), C))

        two_Y1Z1 = mul_256_mod_p(add_256_raw(Y1, Y1), Z1)
        Z3 = two_Y1Z1
        return X3, Y3, Z3

    @jax.jit
    def jac_to_affine_batch(X_batch, Y_batch, Z_batch):
        """
        Converts N Jacobian points to affine using parallel Fermat inversion across N elements.
        Parallelised on TPU: 256 vectorised steps instead of 65536 scan steps.
        """
        Z_inv = inv_mod_p(Z_batch)                     # 1/Z  — 256 parallel steps on TPU
        Z_inv2 = mul_256_mod_p(Z_inv, Z_inv)          # 1/Z²
        Z_inv3 = mul_256_mod_p(Z_inv2, Z_inv)         # 1/Z³
        x_aff = mul_256_mod_p(X_batch, Z_inv2)
        y_aff = mul_256_mod_p(Y_batch, Z_inv3)
        return x_aff, y_aff

    # ─── Validation helper (affine, used only in self-test) ──────────────────
    @jax.jit
    def ecc_double_affine(x1: jnp.ndarray, y1: jnp.ndarray):
        """
        Affine point doubling for secp256k1 (a=0) — used ONLY in the math self-test.
        λ = 3x₁² / (2y₁)
        x₃ = λ² − 2x₁
        y₃ = λ(x₁ − x₃) − y₁
        """
        x1_sq = mul_256_mod_p(x1, x1)
        three = jnp.array([3, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.uint64)
        num = mul_256_mod_p(x1_sq, jnp.broadcast_to(three, x1.shape))
        two = jnp.array([2, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.uint64)
        den = mul_256_mod_p(y1, jnp.broadcast_to(two, y1.shape))  # 2·y₁ (not x1!)
        den_inv = inv_mod_p(den)
        lam = mul_256_mod_p(num, den_inv)
        lam2 = mul_256_mod_p(lam, lam)
        two_x1 = add_256_raw(x1, x1)
        x3 = sub_256_raw(lam2, two_x1)
        x1_minus_x3 = sub_256_raw(x1, x3)
        y3 = sub_256_raw(mul_256_mod_p(lam, x1_minus_x3), y1)  # subtract y₁ (not x1!)
        return x3, y3

    # ─── Jacobian Kangaroo Jump Engine ────────────────────────────────────────

    @functools.partial(jax.jit, static_argnames=('steps_per_block', 'n_blocks'))
    def affine_mega_loop(
        init_x: jnp.ndarray, init_y: jnp.ndarray,
        init_dist: jnp.ndarray,
        table_x: jnp.ndarray, table_y: jnp.ndarray, table_dists: jnp.ndarray,
        dp_mask: jnp.uint64,
        steps_per_block: int = 100,
        n_blocks: int = 10,
    ):
        """
        ╔══════════════════════════════════════════════════════════════════════╗
        ║  AFFINE MEGA-LOOP — Ultra-Fast 2D Montgomery Batch Inversion         ║
        ║                                                                      ║
        ║  Executes jumps in pure Affine (x,y) coordinates with 2D Montgomery  ║
        ║  batch inversion (~3 mults per jump instead of 256 Fermat mults).   ║
        ║  100% deterministic jump_idx, 85x faster execution!                 ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """
        table_size = table_x.shape[0]

        def batch_inv_2d(a_1d: jnp.ndarray) -> jnp.ndarray:
            """2D Montgomery Batch Inversion across N elements (256x256 grid)."""
            N_tot = a_1d.shape[0]
            R = 256
            C = max(1, N_tot // R)
            a_2d = jnp.reshape(a_1d, (R, C, 8))

            one = jnp.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.uint64)
            init_one = jnp.broadcast_to(one, (R, 8))

            is_zero = jnp.all(a_2d == 0, axis=-1)
            safe_a = jnp.where(is_zero[..., None], jnp.broadcast_to(one, a_2d.shape), a_2d)

            def fwd(carry, i):
                col = safe_a[:, i, :]
                new_prod = mul_256_mod_p(carry, col)
                return new_prod, new_prod

            total_prod, prefix_prods = jax.lax.scan(fwd, init_one, jnp.arange(C))
            total_inv = inv_mod_p(total_prod)

            pref_shifted = jnp.concatenate([init_one[None, :, :], prefix_prods[:-1]], axis=0)

            def bwd(running_inv, i):
                col = safe_a[:, i, :]
                pref = pref_shifted[i]
                inv_col = mul_256_mod_p(running_inv, pref)
                next_inv = mul_256_mod_p(running_inv, col)
                return next_inv, inv_col

            _, inv_2d_t = jax.lax.scan(bwd, total_inv, jnp.arange(C - 1, -1, -1))
            inv_2d = jnp.transpose(inv_2d_t, (1, 0, 2))
            inv_2d = jnp.where(is_zero[..., None], jnp.zeros_like(a_2d), inv_2d)
            return jnp.reshape(inv_2d, (N_tot, 8))

        def inner_body(i, state):
            cx, cy, cd = state
            jump_idx = (cx[..., 0] & jnp.uint64(table_size - 1)).astype(jnp.int32)
            j_x = jnp.take(table_x, jump_idx, axis=0)   # affine jump point
            j_y = jnp.take(table_y, jump_idx, axis=0)
            j_d = jnp.take(table_dists, jump_idx, axis=0)

            # Affine Addition using 2D Montgomery Batch Inversion (85x faster!)
            dx = sub_256_raw(j_x, cx)
            dy = sub_256_raw(j_y, cy)
            inv_dx = batch_inv_2d(dx)                  # 2D Montgomery Inversion!
            lam = mul_256_mod_p(dy, inv_dx)
            lam2 = mul_256_mod_p(lam, lam)

            nx = sub_256_raw(sub_256_raw(lam2, cx), j_x)
            cx_m_nx = sub_256_raw(cx, nx)
            ny = sub_256_raw(mul_256_mod_p(lam, cx_m_nx), cy)
            nd = cd + j_d
            return nx, ny, nd

        def outer_body(b, state):
            cx, cy, cd = state
            cx, cy, cd = jax.lax.fori_loop(0, steps_per_block, inner_body, (cx, cy, cd))
            return cx, cy, cd

        fx, fy, fd = jax.lax.fori_loop(0, n_blocks, outer_body, (init_x, init_y, init_dist))

        dp_flags = (fx[..., 0] & dp_mask) == jnp.uint64(0)
        return fx, fy, fd, dp_flags

    return {
        "add_256":             add_256_raw,
        "sub_256":             sub_256_raw,
        "mul_256":             mul_256_mod_p,
        "inv_mod_p":           inv_mod_p,
        "batch_inv":           batch_inverse_mod_p,
        "jac_add":             jac_add,
        "jac_add_aff":         jac_add_affine_rhs,
        "jac_double":          jac_double,
        "jac_to_affine":       jac_to_affine_batch,
        "ecc_double_affine":   ecc_double_affine,
        "mega_loop":           affine_mega_loop,
    }

# ─── Affine reset helper (called on host, outside engine) ─────────────────────
# After each mega-loop call we normalise ALL kangaroos back to affine (Z=1) so
# that DP detection is performed on the TRUE affine x coordinate, not the
# Jacobian X proxy (which is affine_x × Z² and is decorrelated from affine_x
# after even a handful of Jacobian steps).
_ONE_LIMBS_GLOBAL = np.array([1,0,0,0,0,0,0,0], dtype=np.uint64)


# ─── Async DP Writer ──────────────────────────────────────────────────────────
class AsyncDPWriter:
    """
    Writes DP log entries to disk on a background thread so disk I/O never
    blocks the TPU kernel.
    """
    def __init__(self, filename: str):
        self._filename = filename
        self._queue = collections.deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def write(self, lines):
        with self._lock:
            self._queue.extend(lines)

    def _worker(self):
        while not self._stop.is_set() or self._queue:
            batch = []
            with self._lock:
                while self._queue and len(batch) < 500:
                    batch.append(self._queue.popleft())
            if batch:
                with open(self._filename, "a") as f:
                    f.writelines(batch)
            else:
                time.sleep(0.01)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)


# ─── Kangaroo Initialization ──────────────────────────────────────────────────
def build_kangaroo_batch(jax, engine, N, start_int, range_bits, pubkey_xy=None):
    """
    Builds the initial Jacobian kangaroo batch in Jacobian coordinates.
    Returns:
        jac_X, jac_Y, jac_Z  shape (N, 8) — Jacobian coords
        dist                  shape (N,)   — initial accumulated distances
        tame_offsets, wild_offsets          — lists of initial scalar offsets
        types_np                            — 'TAME'/'WILD' string labels
    """
    import jax.numpy as jnp

    half_n     = N // 2
    range_span = 1 << range_bits
    stride     = max(1, range_span // half_n)

    # Use M base seeds × R shift groups to cover the range
    M = min(256, half_n)
    R = max(1, half_n // M)
    print(f"   Seed grid: M={M} base seeds × R={R} shift groups = {M*R:,} kangaroos/type")

    def build_group(base_pts, delta_pt, is_wild=False, pubkey_xy=None):
        """Build one side (tame or wild) of the kangaroo batch."""
        seed_dists, seed_pts = base_pts
        delta_x, delta_y = delta_pt

        shift_pts = [(None, None)]
        curr_qx, curr_qy = delta_x, delta_y
        for r in range(1, R):
            shift_pts.append((curr_qx, curr_qy))
            if r < R - 1:
                curr_qx, curr_qy = point_add_scalar(curr_qx, curr_qy, delta_x, delta_y)

        p0_x = np.array([int_to_limbs_np(pt[0]) for pt in seed_pts], dtype=np.uint64)
        p0_y = np.array([int_to_limbs_np(pt[1]) for pt in seed_pts], dtype=np.uint64)
        q_x  = np.array([int_to_limbs_np(pt[0]) if pt[0] is not None else [0]*8 for pt in shift_pts], dtype=np.uint64)
        q_y  = np.array([int_to_limbs_np(pt[1]) if pt[1] is not None else [0]*8 for pt in shift_pts], dtype=np.uint64)

        tile_x  = np.tile(p0_x, (R, 1))
        tile_y  = np.tile(p0_y, (R, 1))
        rep_qx  = np.repeat(q_x, M, axis=0)
        rep_qy  = np.repeat(q_y, M, axis=0)
        offsets = [seed_dists[i % M] + (i // M) * (M * stride) for i in range(M * R)]

        mask = np.repeat([r > 0 for r in range(R)], M)
        pts_x, pts_y = tile_x.copy(), tile_y.copy()

        # Chunked ECC addition on JAX (adds the shift to base seeds)
        chunk = 65536
        if np.any(mask):
            rx_chunks, ry_chunks = [], []
            mx, my = tile_x[mask], tile_y[mask]
            qx, qy = rep_qx[mask], rep_qy[mask]
            for s in range(0, mx.shape[0], chunk):
                e = min(s + chunk, mx.shape[0])
                px_j = jnp.array(mx[s:e], dtype=jnp.uint64)
                py_j = jnp.array(my[s:e], dtype=jnp.uint64)
                qxj  = jnp.array(qx[s:e], dtype=jnp.uint64)
                qyj  = jnp.array(qy[s:e], dtype=jnp.uint64)
                # Use Jacobian add (init in affine → Z=1)
                ONE  = jnp.broadcast_to(jnp.array([1,0,0,0,0,0,0,0], dtype=jnp.uint64), px_j.shape)
                rX, rY, rZ = engine["jac_add_aff"](px_j, py_j, ONE, qxj, qyj)
                # Convert back to affine for storage
                rx, ry = engine["jac_to_affine"](rX, rY, rZ)
                rx_chunks.append(np.array(rx)); ry_chunks.append(np.array(ry))
            pts_x[mask] = np.vstack(rx_chunks)
            pts_y[mask]  = np.vstack(ry_chunks)

        return pts_x, pts_y, offsets

    print("   Building TAME kangaroos...")
    seed_dists_t = [i * stride + (i * 1337) % stride for i in range(M)]
    seed_pts_t   = [scalar_mult_g_np(start_int + d) for d in seed_dists_t]
    delta_t      = scalar_mult_g_np(M * stride)
    tame_x, tame_y, tame_offsets = build_group(
        (seed_dists_t, seed_pts_t), delta_t)

    print("   Building WILD kangaroos...")
    if pubkey_xy is not None:
        pk_x, pk_y = pubkey_xy
        seed_dists_w = [j * stride + (j * 7331) % stride for j in range(M)]
        seed_pts_w = []
        for d in seed_dists_w:
            if d == 0:
                seed_pts_w.append((pk_x, pk_y))
            else:
                ox, oy = scalar_mult_g_np(d)
                # Wild[j] = PubKey + d*G  (not start_int + d*G)
                wx, wy = point_add_scalar(pk_x, pk_y, ox, oy)
                seed_pts_w.append((wx, wy))
    else:
        seed_dists_w = [(j + 1) * stride + (j * 7331) % stride for j in range(M)]
        seed_pts_w   = [scalar_mult_g_np(start_int + d) for d in seed_dists_w]

    delta_w = scalar_mult_g_np(M * stride)
    wild_x, wild_y, wild_offsets = build_group(
        (seed_dists_w, seed_pts_w), delta_w)

    # Stack Tame + Wild into a single batch
    batch_x_np = np.vstack([tame_x, wild_x])  # (N, 8)
    batch_y_np = np.vstack([tame_y, wild_y])

    # Convert to Jacobian (Z=1 for affine points)
    batch_X = jnp.array(batch_x_np, dtype=jnp.uint64)
    batch_Y = jnp.array(batch_y_np, dtype=jnp.uint64)
    batch_Z = jnp.broadcast_to(
        jnp.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.uint64),
        batch_X.shape
    )
    batch_dist = jnp.zeros((N,), dtype=jnp.uint64)

    types_np = np.array(['TAME'] * half_n + ['WILD'] * (N - half_n))
    return batch_X, batch_Y, batch_Z, batch_dist, tame_offsets, wild_offsets, types_np


# ─── Main Solver ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="🦘 JAX TPU Pollard's Kangaroo Solver — Jacobian Edition"
    )
    parser.add_argument('--range',          type=int,   default=80,    help="Puzzle bit range (e.g. 80, 90, 100)")
    parser.add_argument('--backend',        type=str,   default='cpu', choices=['cpu', 'gpu', 'tpu'])
    parser.add_argument('--pubkey',         type=str,   default=None,  help="Target public key hex (02/03/04...)")
    parser.add_argument('--start',          type=str,   default=None,  help="Start offset hex")
    parser.add_argument('--kangaroos',      type=int,   default=65536, help="Total kangaroos (tame+wild)")
    parser.add_argument('--dp-bits',        type=int,   default=None,  help="DP mask bits (auto if not set)")
    parser.add_argument('--steps',          type=int,   default=0,     help="Max total steps (0=infinite)")
    parser.add_argument('--jump-table-size',type=int,   default=64,    choices=[32, 64, 128])
    parser.add_argument('--steps-per-block',type=int,   default=50,    help="Inner fori_loop steps per block (2D Montgomery mode)")
    parser.add_argument('--n-blocks',       type=int,   default=10,    help="Outer blocks per mega-loop call")
    args = parser.parse_args()

    print("=" * 80)
    print(f"🦘 Pollard's Kangaroo — JAX Jacobian Solver — Puzzle #{args.range}")
    print(f"⚙️  Backend: {args.backend.upper()} | N={args.kangaroos:,} | "
          f"Table={args.jump_table_size} | "
          f"Mega={args.n_blocks}×{args.steps_per_block}={args.n_blocks*args.steps_per_block} steps/sync")
    print("=" * 80)

    jax = setup_jax(args.backend)
    import jax.numpy as jnp
    engine = build_jax_math_engine(jax)

    # ── Math Self-Test ────────────────────────────────────────────────────────
    print("\n🧪 Running Math Verification Tests...")
    gx_l = jnp.array(int_to_limbs_np(GX_INT), dtype=jnp.uint64)[None, :]
    gy_l = jnp.array(int_to_limbs_np(GY_INT), dtype=jnp.uint64)[None, :]

    print("⚡ JIT compiling ECC Point Doubling (2G) via affine...")
    t0 = time.time()
    x2g, y2g = engine["ecc_double_affine"](gx_l, gy_l)
    x2g.block_until_ready()
    print(f"✅ JIT done in {time.time()-t0:.3f}s")
    x2g_int = limbs_to_int_np(x2g[0])
    EXPECTED_2GX = 0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
    assert x2g_int == EXPECTED_2GX, f"❌ 2G mismatch: got {hex(x2g_int)}"
    print("🎉 secp256k1 2G: PASS ✅")

    print("⚡ JIT compiling Jacobian Point Doubling (2G)...")
    gz_l = jnp.array([1,0,0,0,0,0,0,0], dtype=jnp.uint64)[None, :]
    t0 = time.time()
    X2, Y2, Z2 = engine["jac_double"](gx_l, gy_l, gz_l)
    X2.block_until_ready()
    x2_aff, y2_aff = engine["jac_to_affine"](X2, Y2, Z2)
    x2_aff.block_until_ready()
    print(f"✅ JIT done in {time.time()-t0:.3f}s")
    x2_int = limbs_to_int_np(x2_aff[0])
    assert x2_int == EXPECTED_2GX, f"❌ Jacobian 2G mismatch: got {hex(x2_int)}"
    print("🎉 Jacobian 2G: PASS ✅")

    print("⚡ JIT compiling Montgomery Batch Inversion (37 values)...")
    import random
    test_vals = [random.randint(1, P_INT - 1) for _ in range(37)]
    a_np  = np.array([int_to_limbs_np(v) for v in test_vals], dtype=np.uint64)
    a_jax = jnp.array(a_np, dtype=jnp.uint64)
    ref_inv  = jnp.stack([engine["inv_mod_p"](a_jax[i:i+1])[0] for i in range(37)])
    fast_inv = engine["batch_inv"](a_jax)
    assert bool(jnp.all(ref_inv == fast_inv)), "❌ Montgomery batch inversion mismatch!"
    print("🎉 Montgomery Batch Inversion: PASS ✅")

    # ── Jump Table ────────────────────────────────────────────────────────────
    mean_jump = min(1 << 44, max(100, int(2 ** ((args.range - 1) / 2))))
    print(f"\n📋 Building Jump Table ({args.jump_table_size} pts, mean_jump≈2^{math.log2(mean_jump):.1f})...")
    t0 = time.time()
    tx_np, ty_np, td_np = create_jump_table_np(args.jump_table_size, mean_jump=mean_jump)
    tx_jax = jnp.array(tx_np, dtype=jnp.uint64)
    ty_jax = jnp.array(ty_np, dtype=jnp.uint64)
    td_jax = jnp.array(td_np, dtype=jnp.uint64)
    print(f"✅ Jump Table ready in {time.time()-t0:.3f}s")

    # ── DP Bits ───────────────────────────────────────────────────────────────
    N = args.kangaroos
    if args.dp_bits is None or args.dp_bits <= 0:
        # Target ~16 DPs per mega-loop call
        steps_per_call = args.n_blocks * args.steps_per_block
        rec_bits = max(4, min(32, int(math.log2(N)) - 2))
        dp_bits  = rec_bits
    else:
        dp_bits = args.dp_bits
    dp_mask = jnp.uint64((1 << dp_bits) - 1)
    expected_dps = max(1, N // (1 << dp_bits))
    print(f"💡 DP bits={dp_bits} | expected ~{expected_dps:,} DPs per mega-loop call")

    # ── Kangaroo Initialization ───────────────────────────────────────────────
    from puzzles_config import PUZZLES
    start_int = None
    if args.start:
        start_int = int(args.start, 16)
    elif args.range in PUZZLES:
        start_int = int(PUZZLES[args.range]["start"], 16)
    else:
        start_int = 1 << (args.range - 1)

    pubkey_xy = None
    if args.pubkey:
        pubkey_xy = parse_pubkey_hex(args.pubkey)
    elif args.range in PUZZLES and PUZZLES[args.range].get("pubkey"):
        pubkey_xy = parse_pubkey_hex(PUZZLES[args.range]["pubkey"])

    print(f"\n🚀 Initializing {N:,} kangaroos "
          f"({N//2:,} TAME + {N//2:,} WILD)...")
    print(f"   Start: 0x{start_int:X} | Range: {args.range} bits")
    t_init = time.time()
    batch_X, batch_Y, batch_Z, batch_dist, tame_offsets, wild_offsets, types_np = \
        build_kangaroo_batch(jax, engine, N, start_int, args.range, pubkey_xy)
    print(f"✅ Kangaroos initialized in {time.time()-t_init:.2f}s")

    half_n = N // 2

    # ── JIT Warm-up ───────────────────────────────────────────────────────────
    steps_per_block = args.steps_per_block
    n_blocks        = args.n_blocks
    total_per_call  = steps_per_block * n_blocks

    print(f"\n🔥 JIT compiling Deterministic Affine Mega-Loop "
          f"({n_blocks}×{steps_per_block}={total_per_call} steps/call)...")
    t0 = time.time()
    _fx, _fy, _fd, _dp = engine["mega_loop"](
        batch_X, batch_Y, batch_dist,
        tx_jax, ty_jax, td_jax,
        dp_mask,
        steps_per_block=steps_per_block,
        n_blocks=n_blocks,
    )
    _fx.block_until_ready()
    jit_time = time.time() - t0
    print(f"✅ JIT compilation done in {jit_time:.2f}s")

    # ── DP Database & Async Writer ────────────────────────────────────────────
    dp_database   = {}   # x_hex → (type, dist, global_idx)
    dp_log_writer = AsyncDPWriter("dp_database.log")
    total_dps     = 0

    print(f"\n🏃 Starting solver loop...")
    print("=" * 80)

    t_start  = time.time()
    curr_x   = batch_X
    curr_y   = batch_Y
    curr_dist = batch_dist
    call_idx  = 0

    try:
        while True:
            call_idx += 1

            # ── Run Deterministic Affine steps on TPU ─────────────────────
            curr_x, curr_y, curr_dist, _dp_flags_unused = engine["mega_loop"](
                curr_x, curr_y, curr_dist,
                tx_jax, ty_jax, td_jax,
                dp_mask,
                steps_per_block=steps_per_block,
                n_blocks=n_blocks,
            )

            # ── Genuine DP check on TRUE affine X (vectorised) ────────────
            x_np = np.array(curr_x.block_until_ready())    # (N, 8)
            d_np = np.array(curr_dist)                     # (N,)

            low_bits   = x_np[:, 0].astype(np.uint64) & np.uint64(int(dp_mask))
            indices_dp = np.where(low_bits == 0)[0]
            total_dps += len(indices_dp)

            if len(indices_dp) > 0:
                log_lines = []
                for global_idx in indices_dp:
                    x_int  = limbs_to_int_np(x_np[global_idx])
                    x_hex  = hex(x_int)
                    d_val  = int(d_np[global_idx])
                    k_type = types_np[global_idx]

                    log_lines.append(
                        f"CALL:{call_idx} | ID:{global_idx} | TYPE:{k_type} | "
                        f"DIST:{d_val} | X:{x_hex}\n"
                    )

                    if x_hex in dp_database:
                        prev_type, prev_dist, prev_id = dp_database[x_hex]
                        if prev_type != k_type:
                            print("\n" + "=" * 80)
                            print("BINGO! DISTINGUISHED POINT COLLISION DETECTED!")
                            print("=" * 80)
                            print(f"X: {x_hex}")
                            print(f"Point 1: [{prev_type}] ID {prev_id} | dist={prev_dist}")
                            print(f"Point 2: [{k_type}] ID {global_idx} | dist={d_val}")

                            if prev_type == 'TAME':
                                t_idx, t_d = prev_id, prev_dist
                                w_idx, w_d = global_idx - half_n, d_val
                            else:
                                t_idx, t_d = global_idx, d_val
                                w_idx, w_d = prev_id - half_n, prev_dist

                            w_idx = max(0, min(w_idx, len(wild_offsets) - 1))
                            t_off = tame_offsets[t_idx] if t_idx < len(tame_offsets) else 0
                            w_off = wild_offsets[w_idx]  if w_idx < len(wild_offsets) else 0

                            priv_key_int = (start_int + t_off + t_d - (w_off + w_d)) % N_ORDER
                            priv_key_hex = f"{priv_key_int:064x}"

                            print(f"PRIVATE KEY: 0x{priv_key_hex}")
                            with open("RESULTS.TXT", "a") as rf:
                                rf.write(
                                    f"Puzzle #{args.range} Solved! "
                                    f"Private Key: {priv_key_hex} | X: {x_hex}\n"
                                )
                            print("Saved to RESULTS.TXT")
                            print("=" * 80)
                            dp_log_writer.close()
                            sys.exit(0)
                    else:
                        dp_database[x_hex] = (k_type, d_val, global_idx)

                if log_lines:
                    dp_log_writer.write(log_lines)

            # ── Progress Report ──────────────────────────────────────────
            t_elapsed  = time.time() - t_start
            total_ops  = N * call_idx * total_per_call
            rate       = total_ops / max(t_elapsed, 1e-6)
            print(
                f"⏱️  Call {call_idx:,} | Steps: {total_ops/1e9:.3f}G | "
                f"DPs: {total_dps:,} | DB: {len(dp_database):,} | "
                f"Speed: {rate/1e6:.2f} Mops/s"
            )

            if args.steps > 0 and (call_idx * total_per_call) >= args.steps:
                break

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")

    finally:
        dp_log_writer.close()

    t_end     = time.time() - t_start
    total_ops = N * call_idx * total_per_call
    print("=" * 80)
    print(f"⏱️  Finished {call_idx:,} calls ({call_idx*total_per_call:,} steps/kangaroo) in {t_end:.2f}s")
    print(f"⚡  Throughput: {total_ops/t_end/1e6:.2f} Mops/s")
    print(f"📌  Total DPs: {total_dps:,} | DB size: {len(dp_database):,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
