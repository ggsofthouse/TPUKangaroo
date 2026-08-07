#!/usr/bin/env python3
"""
================================================================================
🦘 JAX/XLA Vectorized Pollard's Kangaroo Solver for secp256k1 (TPU/GPU/CPU)
================================================================================
High-performance ECC discrete logarithm solver targeting Google TPUs & GPUs.
Simulates 256-bit BigInt arithmetic via 8-limb uint32/uint64 arrays in JAX tensors.

Author: Vectorized TPU Math Engine
License: GPLv3 / MIT
================================================================================
"""

import os
import sys
import time
import argparse
import math
import numpy as np
import functools

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# secp256k1 Constants
P_INT = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GX_INT = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY_INT = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

# 8 Limbs of Prime P (32-bit uint64 containers)
P_LIMBS = np.array([
    0xFFFFFC2F, 0xFFFEFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
], dtype=np.uint64)

# Constant C = 2^256 - P = 0x1000003D1 (C0 = 977, C1 = 1)
C_LIMBS = np.array([977, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint64)


def int_to_limbs_np(val_int: int) -> np.ndarray:
    """Converts a Python 256-bit int into an 8-element uint64 numpy array (least-significant limb first)."""
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


MASK32 = np.uint64(0xFFFFFFFF)

def setup_jax(backend: str):
    """Initializes JAX with the requested backend ('tpu', 'gpu', or 'cpu')."""
    os.environ['JAX_PLATFORMS'] = backend
    import jax
    jax.config.update("jax_enable_x64", True) # Enable 64-bit limb accumulators for 32x32 products
    jax.config.update('jax_platform_name', backend)
    print(f"🚀 JAX platform configured: {backend.upper()}")
    print(f"📡 Devices detected: {jax.devices()}")
    return jax


def build_jax_math_engine(jax):
    import jax.numpy as jnp

    p_jax = jnp.array(P_LIMBS, dtype=jnp.uint64)
    c_jax = jnp.array(C_LIMBS, dtype=jnp.uint64)

    # --------------------------------------------------------------------------
    # 1. ADDITION & SUBTRACTION MODULO P
    # --------------------------------------------------------------------------
    @jax.jit
    def add_256_raw(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Adds two 8-limb tensors shape (..., 8) with carry propagation."""
        res_limbs = []
        carry = jnp.zeros(a.shape[:-1], dtype=jnp.uint64)
        for i in range(8):
            s = a[..., i] + b[..., i] + carry
            res_limbs.append(s & MASK32)
            carry = s >> 32
        res = jnp.stack(res_limbs, axis=-1)
        # Add overflow carry * C
        c_mul0 = carry * 977
        c_mul1 = carry * 1
        
        # Add c_mul to res
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
        """Subtractions a - b modulo P for 8-limb tensors shape (..., 8) with Pseudo-Mersenne fold."""
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

        # When underflow (borrow > 0) occurs, res_limbs represents (a - b) mod 2^256.
        # Since 2^256 = P + C, adding P is equivalent to subtracting C (977 + 2^32) from res.
        b_c = jnp.zeros(a.shape[:-1], dtype=jnp.int64)
        
        # Limb 0 - 977
        r0 = res[..., 0].astype(jnp.int64) - 977 - b_c
        b_c = jnp.where(r0 < 0, jnp.int64(1), jnp.int64(0))
        r0_pos = jnp.where(r0 < 0, r0 + 0x100000000, r0)
        
        # Limb 1 - 1
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





    # --------------------------------------------------------------------------
    # 2. MULTIPLICATION & REDUCTION MODULO P (secp256k1)
    # --------------------------------------------------------------------------
    @jax.jit
    def mul_256_mod_p(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Full 8x8 limb multiplication with 12-limb HC fold & exact uint64 integer arithmetic."""
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

        # Split into Low 256 bits (L) and High 256 bits (H)
        L = accum[:8]
        H = accum[8:]

        # H * C where C = 977 + 2^32 (Expanded to 12 limbs for clean carry chain)
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
        H1_0 = hc[8]
        H1_1 = hc[9]
        H1_2 = hc[10]
        H1_3 = hc[11]

        # H1 * C
        h1c_0 = H1_0 * 977
        h1c_1 = H1_0 * 1 + H1_1 * 977
        h1c_2 = H1_1 * 1 + H1_2 * 977
        h1c_3 = H1_2 * 1 + H1_3 * 977
        h1c_4 = H1_3 * 1

        # Sum L + L1 + H1C
        sum_limbs = []
        carry_s = jnp.zeros(a.shape[:-1], dtype=jnp.uint64)

        s0 = L[0] + L1[0] + h1c_0 + carry_s
        sum_limbs.append(s0 & MASK32)
        carry_s = s0 >> 32

        s1 = L[1] + L1[1] + h1c_1 + carry_s
        sum_limbs.append(s1 & MASK32)
        carry_s = s1 >> 32

        s2 = L[2] + L1[2] + h1c_2 + carry_s
        sum_limbs.append(s2 & MASK32)
        carry_s = s2 >> 32

        s3 = L[3] + L1[3] + h1c_3 + carry_s
        sum_limbs.append(s3 & MASK32)
        carry_s = s3 >> 32

        s4 = L[4] + L1[4] + h1c_4 + carry_s
        sum_limbs.append(s4 & MASK32)
        carry_s = s4 >> 32

        for i in range(5, 8):
            si = L[i] + L1[i] + carry_s
            sum_limbs.append(si & MASK32)
            carry_s = si >> 32

        # Final overflow fold with complete 3-limb carry expansion (e0, e1, e2)
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

        # Subtract P if res >= P
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





    # --------------------------------------------------------------------------
    # 3. MODULAR INVERSION VIA FERMAT'S LITTLE THEOREM (a^(P-2) mod P)
    # --------------------------------------------------------------------------
    P_MINUS_2_INT = P_INT - 2
    P_BITS_NP = np.array([(P_MINUS_2_INT >> bit_idx) & 1 for bit_idx in range(256)], dtype=np.int32)

    @jax.jit
    def inv_mod_p(a: jnp.ndarray) -> jnp.ndarray:
        """Computes modular inverse a^(P-2) mod P using LSB-to-MSB binary exponentiation via scan."""
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
        Montgomery's batch (simultaneous) modular inversion.
        Input:  a_batch shape (N, 8)
        Output: shape (N, 8) -- a_batch[i]^-1 mod P
        Cost: ~3N multiplications + 1 Fermat inversion (~256 mults) instead of N * 256 mults!
        """
        one_limbs = jnp.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.uint64)

        is_zero = jnp.all(a_batch == 0, axis=-1)
        safe_a = jnp.where(
            is_zero[:, None], jnp.broadcast_to(one_limbs, a_batch.shape), a_batch
        )

        def prefix_step(running_prod, a_i):
            new_prod = mul_256_mod_p(running_prod, a_i)
            return new_prod, new_prod

        total_prod, prefix_products = jax.lax.scan(prefix_step, one_limbs, safe_a)
        total_inv = inv_mod_p(total_prod[None, :])[0]

        prefix_shifted = jnp.concatenate(
            [one_limbs[None, :], prefix_products[:-1]], axis=0
        )

        def backward_step(running_inv, elems):
            a_i, pref_i = elems
            inv_i = mul_256_mod_p(running_inv, pref_i)
            next_running_inv = mul_256_mod_p(running_inv, a_i)
            return next_running_inv, inv_i

        _, inv_batch = jax.lax.scan(
            backward_step, total_inv, (safe_a, prefix_shifted), reverse=True
        )

        inv_batch = jnp.where(is_zero[:, None], jnp.zeros_like(a_batch), inv_batch)
        return inv_batch

    # --------------------------------------------------------------------------
    # 4. ECC POINT ADDITION AND DOUBLING (secp256k1)
    # --------------------------------------------------------------------------
    @jax.jit
    def ecc_add_affine(x1: jnp.ndarray, y1: jnp.ndarray, x2: jnp.ndarray, y2: jnp.ndarray):
        """Vectorized Affine Point Addition: (X3, Y3) = (X1, Y1) + (X2, Y2)."""
        dy = sub_256_raw(y2, y1)
        dx = sub_256_raw(x2, x1)
        dx_inv = inv_mod_p(dx)
        lam = mul_256_mod_p(dy, dx_inv)

        lam2 = mul_256_mod_p(lam, lam)
        x3 = sub_256_raw(sub_256_raw(lam2, x1), x2)
        
        x1_minus_x3 = sub_256_raw(x1, x3)
        y3 = sub_256_raw(mul_256_mod_p(lam, x1_minus_x3), y1)
        return x3, y3

    @jax.jit
    def ecc_double_affine(x1: jnp.ndarray, y1: jnp.ndarray):
        """Vectorized Affine Point Doubling for secp256k1 (a=0): (X3, Y3) = 2*(X1, Y1)."""
        x1_sq = mul_256_mod_p(x1, x1)
        three_limbs = jnp.array([3, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.uint64)
        num = mul_256_mod_p(x1_sq, jnp.broadcast_to(three_limbs, x1.shape))

        two_limbs = jnp.array([2, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.uint64)
        den = mul_256_mod_p(y1, jnp.broadcast_to(two_limbs, y1.shape))
        den_inv = inv_mod_p(den)
        lam = mul_256_mod_p(num, den_inv)

        lam2 = mul_256_mod_p(lam, lam)
        two_x1 = add_256_raw(x1, x1)
        x3 = sub_256_raw(lam2, two_x1)

        x1_minus_x3 = sub_256_raw(x1, x3)
        y3 = sub_256_raw(mul_256_mod_p(lam, x1_minus_x3), y1)
        return x3, y3

    @jax.jit
    def ecc_add_mixed_jacobian(x1: jnp.ndarray, y1: jnp.ndarray, z1: jnp.ndarray,
                               x2: jnp.ndarray, y2: jnp.ndarray):
        """
        Mixed Jacobian + Affine Point Addition for secp256k1 (a=0):
        (X3, Y3, Z3) = (X1, Y1, Z1) + (X2, Y2, 1)
        Cost: ZERO INVERSIONS, 8 modular multiplications!
        """
        z1_sq = mul_256_mod_p(z1, z1)
        u2 = mul_256_mod_p(x2, z1_sq)
        z1_cub = mul_256_mod_p(z1, z1_sq)
        s2 = mul_256_mod_p(y2, z1_cub)
        h = sub_256_raw(u2, x1)
        r = sub_256_raw(s2, y1)
        h_sq = mul_256_mod_p(h, h)
        h_cub = mul_256_mod_p(h, h_sq)
        v = mul_256_mod_p(x1, h_sq)
        
        r_sq = mul_256_mod_p(r, r)
        two_v = add_256_raw(v, v)
        x3 = sub_256_raw(sub_256_raw(r_sq, h_cub), two_v)
        
        v_minus_x3 = sub_256_raw(v, x3)
        y1_hcub = mul_256_mod_p(y1, h_cub)
        y3 = sub_256_raw(mul_256_mod_p(r, v_minus_x3), y1_hcub)
        
        z3 = mul_256_mod_p(z1, h)
        return x3, y3, z3

    @jax.jit
    def jacobian_to_affine(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray):
        """Convert Jacobian Coordinates (X, Y, Z) back to Affine (X/Z^2, Y/Z^3)."""
        z_inv = inv_mod_p(z)
        z_inv2 = mul_256_mod_p(z_inv, z_inv)
        z_inv3 = mul_256_mod_p(z_inv, z_inv2)
        ax = mul_256_mod_p(x, z_inv2)
        ay = mul_256_mod_p(y, z_inv3)
        return ax, ay

    # --------------------------------------------------------------------------
    # 5. VECTORIZED KANGAROO JUMP ENGINE (Jump Table + jnp.take)
    # --------------------------------------------------------------------------
    @jax.jit
    def kangaroo_jump_step(curr_x: jnp.ndarray, curr_y: jnp.ndarray, curr_dist: jnp.ndarray,
                           table_x: jnp.ndarray, table_y: jnp.ndarray, table_dists: jnp.ndarray,
                           dp_mask: jnp.uint64):
        """
        Vectorized jump step for batch of kangaroos shape (Batch, 8) using jnp.take.
        Jump index is chosen by masking LSB of current X coordinate.
        """
        N = table_x.shape[0]
        jump_idx = (curr_x[..., 0] & jnp.uint64(N - 1)).astype(jnp.int32)

        # Lookup jump coordinates & distance using advanced indexing
        jump_x = jnp.take(table_x, jump_idx, axis=0)
        jump_y = jnp.take(table_y, jump_idx, axis=0)
        jump_dist = jnp.take(table_dists, jump_idx, axis=0)

        # ECC Point Addition across all kangaroos simultaneously
        next_x, next_y = ecc_add_affine(curr_x, curr_y, jump_x, jump_y)
        next_dist = curr_dist + jump_dist

        # Native TPU DP check
        is_dp = (next_x[..., 0] & dp_mask) == jnp.uint64(0)

        return next_x, next_y, next_dist, is_dp

    @functools.partial(jax.jit, static_argnames=('steps_per_block',))
    def tpu_conconfined_loop(init_x: jnp.ndarray, init_y: jnp.ndarray, init_z: jnp.ndarray, init_dist: jnp.ndarray,
                             table_x: jnp.ndarray, table_y: jnp.ndarray, table_dists: jnp.ndarray,
                             dp_mask: jnp.uint64, steps_per_block: int = 10):
        """
        Executes zero-inversion Jacobian ECC jumps DIRECTLY on TPU hardware,
        without any data transfer or bus interruption per step.
        """
        table_size = table_x.shape[0]

        def body_fun(i, state):
            cx, cy, cz, cd = state
            jump_idx = (cx[..., 0] & jnp.uint64(table_size - 1)).astype(jnp.int32)
            j_x = jnp.take(table_x, jump_idx, axis=0)
            j_y = jnp.take(table_y, jump_idx, axis=0)
            j_d = jnp.take(table_dists, jump_idx, axis=0)
            nx, ny, nz = ecc_add_mixed_jacobian(cx, cy, cz, j_x, j_y)
            nd = cd + j_d
            return nx, ny, nz, nd

        init_state = (init_x, init_y, init_z, init_dist)
        final_jx, final_jy, final_jz, final_dist = jax.lax.fori_loop(0, steps_per_block, body_fun, init_state)
        final_ax, final_ay = jacobian_to_affine(final_jx, final_jy, final_jz)
        final_dp_flags = (final_ax[..., 0] & dp_mask) == jnp.uint64(0)
        return final_ax, final_ay, final_jx, final_jy, final_jz, final_dist, final_dp_flags

    return {
        "add_256": add_256_raw,
        "sub_256": sub_256_raw,
        "mul_256": mul_256_mod_p,
        "inv_mod_p": inv_mod_p,
        "batch_inverse_mod_p": batch_inverse_mod_p,
        "ecc_add": ecc_add_affine,
        "ecc_double": ecc_double_affine,
        "ecc_add_mixed_jacobian": ecc_add_mixed_jacobian,
        "jacobian_to_affine": jacobian_to_affine,
        "jump_step": kangaroo_jump_step,
        "tpu_conconfined_loop": tpu_conconfined_loop,
    }



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
    """Computes k * G in pure Python for Jump Table generation."""
    if k_int == 0: return None, None
    rx, ry = None, None
    for bit in bin(k_int)[2:]:
        if rx is not None:
            rx, ry = point_add_scalar(rx, ry, rx, ry)
        if bit == '1':
            rx, ry = point_add_scalar(rx, ry, GX_INT, GY_INT)
    return rx, ry



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


def parse_pubkey_hex(pubkey_str: str):
    """Decompresses compressed (02/03) or uncompressed (04) SEC public key to (X, Y) integers."""
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
        raise ValueError("Formato de Chave Pública inválido (deve começar com 02, 03 ou 04)")


def main():
    parser = argparse.ArgumentParser(description="JAX TPU Pollard's Kangaroo Solver for secp256k1")
    parser.add_argument('--range', type=int, default=80, help="Puzzle bit range (e.g. 40, 60, 80, 140)")
    parser.add_argument('--backend', type=str, default='cpu', choices=['cpu', 'gpu', 'tpu'], help="Hardware backend (tpu, gpu, cpu)")
    parser.add_argument('--pubkey', type=str, default=None, help="Target public key in hex (compressed 02/03... or uncompressed 04...)")
    parser.add_argument('--start', type=str, default="80000000000000000000", help="Start offset hex of the range")
    parser.add_argument('--kangaroos', type=int, default=1024, help="Number of parallel kangaroos per tensor batch")
    parser.add_argument('--dp-bits', type=int, default=None, help="Distinguished point bits (auto-recommended targeting 500-1000 DPs/step if not specified)")
    parser.add_argument('--steps', type=int, default=0, help="Steps to run (0 for infinite loop)")
    parser.add_argument('--jump-table-size', type=int, default=64, choices=[32, 64, 128], help="Jump table size (32, 64 or 128)")
    parser.add_argument('--inner-steps', type=int, default=10, help="TPU hardware inner unrolled steps per cycle (default: 10)")
    args = parser.parse_args()

    print("================================================================================")
    print(f"🦘 Pollard's Kangaroo JAX TPU Solver - Puzzle #{args.range}")
    print(f"⚙️ Target Backend: {args.backend.upper()} | Kangaroos Batch: {args.kangaroos:,} | Jump Table Size: {args.jump_table_size}")
    if args.pubkey:
        print(f"🎯 Target Pubkey: {args.pubkey[:24]}... | Start Range: 0x{args.start}")
    print("================================================================================")

    jax = setup_jax(args.backend)
    import jax.numpy as jnp
    engine = build_jax_math_engine(jax)

    # --------------------------------------------------------------------------
    # VALIDATION TEST 1: Point Doubling 2 * G
    # --------------------------------------------------------------------------
    print("\n🧪 Running Math Verification Tests...")
    gx_limbs = jnp.array(int_to_limbs_np(GX_INT), dtype=jnp.uint64)
    gy_limbs = jnp.array(int_to_limbs_np(GY_INT), dtype=jnp.uint64)

    # Batch of shape (1, 8)
    gx_batch = gx_limbs[None, :]
    gy_batch = gy_limbs[None, :]

    print("⚡ JIT Compiling ECC Point Doubling (2G)...")
    t0 = time.time()
    x2g, y2g = engine["ecc_double"](gx_batch, gy_batch)
    x2g.block_until_ready()
    t_jit = time.time() - t0
    print(f"✅ JIT Compilation completed in {t_jit:.4f}s")

    x2g_int = limbs_to_int_np(x2g[0])
    EXPECTED_2GX = 0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5

    print(f"   Calculated 2G_x: {hex(x2g_int)}")
    print(f"   Expected   2G_x: {hex(EXPECTED_2GX)}")

    if x2g_int == EXPECTED_2GX:
        print("🎉 SECP256K1 POINT DOUBLING MATEMATICAMENTE PERFEITO!")
    else:
        print("❌ ERRO NA VERIFICAÇÃO MATEMÁTICA!")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # VALIDATION TEST 2: Montgomery Batch Inversion Self-Test
    # --------------------------------------------------------------------------
    print("⚡ JIT Compiling Montgomery Batch Inversion Self-Test (37 random values)...")
    import random
    test_vals = [random.randint(1, P_INT - 1) for _ in range(37)]

    a_np = np.array([int_to_limbs_np(v) for v in test_vals], dtype=np.uint64)
    a_jax = jnp.array(a_np, dtype=jnp.uint64)

    ref_inv = jnp.stack([engine["inv_mod_p"](a_jax[i:i+1])[0] for i in range(len(test_vals))])
    fast_inv = engine["batch_inverse_mod_p"](a_jax)

    ok = bool(jnp.all(ref_inv == fast_inv))
    if ok:
        print("🎉 MONTGOMERY BATCH INVERSION MATEMATICAMENTE PERFEITA (PASSOU ✅)!")
    else:
        print("❌ ERRO NA INVERSÃO EM LOTE DE MONTGOMERY!")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # PREPARE JUMP TABLE (DYNAMIC MEAN JUMP FOR RANGE)
    # --------------------------------------------------------------------------
    mean_jump = min(1 << 44, max(100, int(2 ** ((args.range - 1) / 2))))
    print(f"\n📋 Building Static Jump Table ({args.jump_table_size} points, Mean Jump ~2^{math.log2(mean_jump):.1f})...")
    t_jt_start = time.time()
    tx_np, ty_np, td_np = create_jump_table_np(args.jump_table_size, mean_jump=mean_jump)
    
    tx_jax = jnp.array(tx_np, dtype=jnp.uint64)
    ty_jax = jnp.array(ty_np, dtype=jnp.uint64)
    td_jax = jnp.array(td_np, dtype=jnp.uint64)
    print(f"✅ Jump Table generated in {time.time() - t_jt_start:.4f}s")

    # --------------------------------------------------------------------------
    # BENCHMARK & SOLVER INITIALIZATION: FAST SEED TILE SETUP (<0.5s)
    # --------------------------------------------------------------------------
    N = args.kangaroos
    half_n = N // 2
    print(f"\n🚀 Preparing Tensor Batch of {N:,} Kangaroos ({half_n} TAME + {N - half_n} WILD)...")
    
    start_int = int(args.start, 16) if args.start else 1
    range_span = (1 << args.range) if args.range < 256 else (1 << 80)
    stride = max(1, range_span // half_n)

    # Instant Batch Generator for 100% Unique Kangaroo Positions in <0.5s
    M = 256
    R = half_n // M

    # 1. TAME SETUP: Base Seed (M=256) + Shift Points (R=half_n//M)
    seed_dists_t = [i * stride + (i * 1337) % stride for i in range(M)]
    seed_pts_t = [scalar_mult_g_np(start_int + d) for d in seed_dists_t]
    
    delta_t_dist = M * stride
    delta_t_x, delta_t_y = scalar_mult_g_np(delta_t_dist)

    shift_pts_t = [(None, None)]
    curr_qx, curr_qy = delta_t_x, delta_t_y
    for r in range(1, R):
        shift_pts_t.append((curr_qx, curr_qy))
        if r < R - 1:
            curr_qx, curr_qy = point_add_scalar(curr_qx, curr_qy, delta_t_x, delta_t_y)

    p0_t_x = np.array([int_to_limbs_np(pt[0]) for pt in seed_pts_t], dtype=np.uint64)
    p0_t_y = np.array([int_to_limbs_np(pt[1]) for pt in seed_pts_t], dtype=np.uint64)
    q_t_x = np.array([int_to_limbs_np(pt[0]) if pt[0] is not None else [0]*8 for pt in shift_pts_t], dtype=np.uint64)
    q_t_y = np.array([int_to_limbs_np(pt[1]) if pt[1] is not None else [0]*8 for pt in shift_pts_t], dtype=np.uint64)

    tame_p0_x = np.tile(p0_t_x, (R, 1))
    tame_p0_y = np.tile(p0_t_y, (R, 1))
    tame_q_x = np.repeat(q_t_x, M, axis=0)
    tame_q_y = np.repeat(q_t_y, M, axis=0)

    tame_offsets = [seed_dists_t[i] + r * delta_t_dist for r in range(R) for i in range(M)]

    mask_t = np.repeat([r > 0 for r in range(R)], M)
    tame_x_np = tame_p0_x.copy()
    tame_y_np = tame_p0_y.copy()

    def chunked_ecc_add(p_x_np, p_y_np, q_x_np, q_y_np, chunk_size=65536):
        num_pts = p_x_np.shape[0]
        rx_chunks, ry_chunks = [], []
        for s in range(0, num_pts, chunk_size):
            e = min(s + chunk_size, num_pts)
            px_j = jnp.array(p_x_np[s:e], dtype=jnp.uint64)
            py_j = jnp.array(p_y_np[s:e], dtype=jnp.uint64)
            qx_j = jnp.array(q_x_np[s:e], dtype=jnp.uint64)
            qy_j = jnp.array(q_y_np[s:e], dtype=jnp.uint64)
            rx, ry = engine["ecc_add"](px_j, py_j, qx_j, qy_j)
            rx_chunks.append(np.array(rx))
            ry_chunks.append(np.array(ry))
        return np.vstack(rx_chunks), np.vstack(ry_chunks)

    if np.any(mask_t):
        rx_np, ry_np = chunked_ecc_add(tame_p0_x[mask_t], tame_p0_y[mask_t], tame_q_x[mask_t], tame_q_y[mask_t])
        tame_x_np[mask_t] = rx_np
        tame_y_np[mask_t] = ry_np

    # 2. WILD SETUP: Base Seed (M=256) + Shift Points (R=half_n//M)
    if args.pubkey:
        pk_x, pk_y = parse_pubkey_hex(args.pubkey)
        seed_dists_w = [j * stride + (j * 7331) % stride for j in range(M)]
        seed_pts_w = []
        for d in seed_dists_w:
            if d == 0:
                seed_pts_w.append((pk_x, pk_y))
            else:
                ox, oy = scalar_mult_g_np(d)
                num = (oy - pk_y) % P_INT
                den = (ox - pk_x) % P_INT
                lam = (num * pow(den, P_INT - 2, P_INT)) % P_INT
                wx = (lam**2 - pk_x - ox) % P_INT
                wy = (lam * (pk_x - wx) - pk_y) % P_INT
                seed_pts_w.append((wx, wy))

        delta_w_dist = M * stride
        delta_w_x, delta_w_y = scalar_mult_g_np(delta_w_dist)

        shift_pts_w = [(None, None)]
        curr_wqx, curr_wqy = delta_w_x, delta_w_y
        for r in range(1, R):
            shift_pts_w.append((curr_wqx, curr_wqy))
            if r < R - 1:
                curr_wqx, curr_wqy = point_add_scalar(curr_wqx, curr_wqy, delta_w_x, delta_w_y)

        p0_w_x = np.array([int_to_limbs_np(pt[0]) for pt in seed_pts_w], dtype=np.uint64)
        p0_w_y = np.array([int_to_limbs_np(pt[1]) for pt in seed_pts_w], dtype=np.uint64)
        q_w_x = np.array([int_to_limbs_np(pt[0]) if pt[0] is not None else [0]*8 for pt in shift_pts_w], dtype=np.uint64)
        q_w_y = np.array([int_to_limbs_np(pt[1]) if pt[1] is not None else [0]*8 for pt in shift_pts_w], dtype=np.uint64)

        wild_p0_x = np.tile(p0_w_x, (R, 1))
        wild_p0_y = np.tile(p0_w_y, (R, 1))
        wild_q_x = np.repeat(q_w_x, M, axis=0)
        wild_q_y = np.repeat(q_w_y, M, axis=0)

        wild_offsets = [seed_dists_w[j] + r * delta_w_dist for r in range(R) for j in range(M)]

        mask_w = np.repeat([r > 0 for r in range(R)], M)
        wild_x_np = wild_p0_x.copy()
        wild_y_np = wild_p0_y.copy()

        if np.any(mask_w):
            rwx_np, rwy_np = chunked_ecc_add(wild_p0_x[mask_w], wild_p0_y[mask_w], wild_q_x[mask_w], wild_q_y[mask_w])
            wild_x_np[mask_w] = rwx_np
            wild_y_np[mask_w] = rwy_np
    else:
        seed_dists_w = [(j + 1) * stride + (j * 7331) % stride for j in range(M)]
        seed_pts_w = [scalar_mult_g_np(start_int + d) for d in seed_dists_w]

        delta_w_dist = M * stride
        delta_w_x, delta_w_y = scalar_mult_g_np(delta_w_dist)

        shift_pts_w = [(None, None)]
        curr_wqx, curr_wqy = delta_w_x, delta_w_y
        for r in range(1, R):
            shift_pts_w.append((curr_wqx, curr_wqy))
            if r < R - 1:
                curr_wqx, curr_wqy = point_add_scalar(curr_wqx, curr_wqy, delta_w_x, delta_w_y)

        p0_w_x = np.array([int_to_limbs_np(pt[0]) for pt in seed_pts_w], dtype=np.uint64)
        p0_w_y = np.array([int_to_limbs_np(pt[1]) for pt in seed_pts_w], dtype=np.uint64)
        q_w_x = np.array([int_to_limbs_np(pt[0]) if pt[0] is not None else [0]*8 for pt in shift_pts_w], dtype=np.uint64)
        q_w_y = np.array([int_to_limbs_np(pt[1]) if pt[1] is not None else [0]*8 for pt in shift_pts_w], dtype=np.uint64)

        wild_p0_x = np.tile(p0_w_x, (R, 1))
        wild_p0_y = np.tile(p0_w_y, (R, 1))
        wild_q_x = np.repeat(q_w_x, M, axis=0)
        wild_q_y = np.repeat(q_w_y, M, axis=0)

        wild_offsets = [seed_dists_w[j] + r * delta_w_dist for r in range(R) for j in range(M)]

        mask_w = np.repeat([r > 0 for r in range(R)], M)
        wild_x_np = wild_p0_x.copy()
        wild_y_np = wild_p0_y.copy()

        if np.any(mask_w):
            rwx_np, rwy_np = chunked_ecc_add(wild_p0_x[mask_w], wild_p0_y[mask_w], wild_q_x[mask_w], wild_q_y[mask_w])
            wild_x_np[mask_w] = rwx_np
            wild_y_np[mask_w] = rwy_np




    batch_kx_np = np.vstack([tame_x_np, wild_x_np])
    batch_ky_np = np.vstack([tame_y_np, wild_y_np])
    batch_kz_np = np.zeros((N, 8), dtype=np.uint64)
    batch_kz_np[:, 0] = 1
    
    batch_kx = jnp.array(batch_kx_np, dtype=jnp.uint64)
    batch_ky = jnp.array(batch_ky_np, dtype=jnp.uint64)
    batch_kz = jnp.array(batch_kz_np, dtype=jnp.uint64)
    batch_dist = jnp.zeros((N,), dtype=jnp.uint64)


    if args.dp_bits is None or args.dp_bits <= 0:
        # Target ~8 DPs per step (log2(N) - 3) to keep Host CPU dictionary overhead at 0.00001s
        rec_bits = max(4, min(32, int(math.log2(N)) - 3))
        dp_bits = rec_bits
        expected_dps = max(1, N // (1 << dp_bits))
        print(f"💡 recomendação automática ativada: --dp-bits={dp_bits} (Densidade Alvo: ~{expected_dps:,} DPs por passo)")
    else:
        dp_bits = args.dp_bits
        expected_dps = max(1, N // (1 << dp_bits))
        print(f"🎯 Distinguished Points (DP) ativado: Lowest {dp_bits} bits masked (0x{(1 << dp_bits) - 1:X}, ~{expected_dps:,} DPs por passo)")

    dp_mask = jnp.uint64((1 << dp_bits) - 1)
    steps_per_block = args.inner_steps if args.inner_steps > 0 else 10
    print(f"🔥 MÁXIMA FORÇA ATIVADA: JIT Compilando blocos de {steps_per_block:,} saltos confinados (Zero Inversões Jacobianas)...")
    t_jit_jump = time.time()
    test_ax, test_ay, test_jx, test_jy, test_jz, test_fd, test_dp = engine["tpu_conconfined_loop"](
        batch_kx, batch_ky, batch_kz, batch_dist, tx_jax, ty_jax, td_jax, dp_mask, steps_per_block=steps_per_block
    )
    test_ax.block_until_ready()
    print(f"✅ Kernel TPU Confinado JIT Compilado em {time.time() - t_jit_jump:.4f}s")

    print(f"🔥 Executando marcha contínua na matriz sistólica across {N:,} kangaroos ({steps_per_block:,} saltos/bloco)...")

    types_np = np.array(['TAME'] * half_n + ['WILD'] * (N - half_n))

    dp_database = {}
    dp_log_filename = "dp_database.log"
    total_dps = 0

    t_start = time.time()
    curr_jx, curr_jy, curr_jz, curr_dist = batch_kx, batch_ky, batch_kz, batch_dist
    bloco = 0

    while True:
        bloco += 1
        
        curr_ax, curr_ay, curr_jx, curr_jy, curr_jz, curr_dist, dp_flags = engine["tpu_conconfined_loop"](
            curr_jx, curr_jy, curr_jz, curr_dist,
            tx_jax, ty_jax, td_jax,
            dp_mask,
            steps_per_block=steps_per_block
        )

        dp_flags_cpu = np.array(dp_flags.block_until_ready())
        indices_dp = np.where(dp_flags_cpu)[0]
        total_dps += len(indices_dp)

        if len(indices_dp) > 0:
            dp_indices_jax = jnp.array(indices_dp)
            dp_x_np = np.array(curr_ax[dp_indices_jax].block_until_ready())
            dp_dist_np = np.array(curr_dist[dp_indices_jax].block_until_ready())
            
            log_lines = []
            for i, global_idx in enumerate(indices_dp):
                x_hex = hex(limbs_to_int_np(dp_x_np[i]))
                d_val = int(dp_dist_np[i])
                k_type = types_np[global_idx]
                log_lines.append(f"BLOCK:{bloco} | ID:{global_idx} | TYPE:{k_type} | DIST:{d_val} | X:{x_hex}\n")

                if x_hex in dp_database:
                    prev_type, prev_dist, prev_id = dp_database[x_hex]
                    if prev_type != k_type:
                        print("\n" + "=" * 80)
                        print("🎉 BINGO! COLISÃO DE PONTO DISTINTO (DP) DETECTADA!")
                        print("=" * 80)
                        print(f"📍 Ponto X: {x_hex}")
                        print(f"🦘 Ponto 1: [{prev_type}] ID {prev_id} | Distância = {prev_dist}")
                        print(f"🦘 Ponto 2: [{k_type}] ID {global_idx} | Distância = {d_val}")

                        if prev_type == 'TAME':
                            tame_idx, tame_d = prev_id, prev_dist
                            wild_idx, wild_d = global_idx - half_n, d_val
                        else:
                            tame_idx, tame_d = global_idx, d_val
                            wild_idx, wild_d = prev_id - half_n, prev_dist

                        tame_init_off = tame_offsets[tame_idx]
                        wild_init_off = wild_offsets[wild_idx]

                        priv_key_int = (start_int + tame_init_off + tame_d - (wild_init_off + wild_d)) % N_ORDER
                        priv_key_hex = f"{priv_key_int:064x}"

                        print(f"🔑 CHAVE PRIVADA ENCONTRADA: 0x{priv_key_hex}")
                        print(f"💾 Resultado gravado em RESULTS.TXT")
                        with open("RESULTS.TXT", "a") as rf:
                            rf.write(f"Puzzle #{args.range} Solved! Private Key: {priv_key_hex} | X: {x_hex}\n")
                        print("=" * 80)
                        sys.exit(0)
                else:
                    dp_database[x_hex] = (k_type, d_val, global_idx)

            with open(dp_log_filename, "a") as f:
                f.writelines(log_lines)

        t_elapsed = time.time() - t_start
        total_ops = N * bloco * steps_per_block
        rate = total_ops / t_elapsed
        print(f"⏱️ Bloco {bloco:,} | Saltos Totais: {total_ops:,} | DPs: {total_dps:,} | Velocidade: {rate/1e6:.2f} Mops/s ({rate/1e3:.2f} Kops/s) | Turbina Livre!")

        if args.steps > 0 and (bloco * steps_per_block) >= args.steps:
            break

    curr_jx.block_until_ready()
    t_end = time.time() - t_start
    
    total_ops = N * bloco * steps_per_block
    rate = total_ops / t_end
    print("================================================================================")
    print(f"⏱️ Finalizado {bloco:,} blocos ({bloco * steps_per_block:,} saltos/canguru) em {t_end:.4f} segundos")
    print(f"⚡ Throughput Rate: {rate / 1e3:.2f} Kops/sec ({rate / 1e6:.4f} Mops/sec)")
    print(f"📌 Total de Pontos Distintos (DPs) capturados: {total_dps} (DB size: {len(dp_database)})")
    print("================================================================================")


if __name__ == "__main__":
    main()

