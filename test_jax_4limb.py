import os
import sys
import time
import numpy as np

os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

P_INT = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
GX_INT = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY_INT = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
C_INT = 0x1000003D1

# P in 4 limbs of 64-bit uint64
P_LIMBS4 = np.array([
    0xFFFFFFFFEFFFFFC2F,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF
], dtype=np.uint64)

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)

def int_to_limbs4(val: int) -> np.ndarray:
    res = np.zeros(4, dtype=np.uint64)
    temp = val
    for i in range(4):
        res[i] = temp & 0xFFFFFFFFFFFFFFFF
        temp >>= 64
    return res

def limbs4_to_int(limbs) -> int:
    flat = np.array(limbs).flatten()
    val = 0
    for i in range(3, -1, -1):
        val = (val << 64) | int(flat[i])
    return val

p_jax4 = jnp.array(P_LIMBS4, dtype=jnp.uint64)

# ------------------------------------------------------------------------------
# 4-LIMB ADDITION & SUBTRACTION
# ------------------------------------------------------------------------------
@jax.jit
def add_256_4l(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Adds two 4-limb tensors shape (..., 4) modulo P."""
    res_limbs = []
    carry = jnp.zeros(a.shape[:-1], dtype=jnp.uint64)
    for i in range(4):
        s = a[..., i] + b[..., i] + carry
        res_limbs.append(s & MASK64)
        carry = s >> 64
    res = jnp.stack(res_limbs, axis=-1)
    
    # Add overflow carry * C_INT
    c_mul = carry * C_INT
    s0 = res[..., 0] + c_mul
    res = res.at[..., 0].set(s0 & MASK64)
    carry0 = s0 >> 64
    
    for i in range(1, 4):
        si = res[..., i] + carry0
        res = res.at[..., i].set(si & MASK64)
        carry0 = si >> 64
        
    return res

@jax.jit
def sub_256_4l(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Subtracts b from a modulo P for 4-limb tensors shape (..., 4)."""
    res_limbs = []
    borrow = jnp.zeros(a.shape[:-1], dtype=jnp.uint64)
    for i in range(4):
        diff = a[..., i] - b[..., i] - borrow
        borrow = (diff >> 63) & 1
        res_limbs.append(diff & MASK64)
    res = jnp.stack(res_limbs, axis=-1)
    
    p_added = add_256_4l(res, jnp.broadcast_to(p_jax4, res.shape))
    res = jnp.where(borrow[..., None] > 0, p_added, res)
    return res

print("Testing 4-limb module loaded...")
