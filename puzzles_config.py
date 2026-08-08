"""
Bitcoin Puzzles Target Configuration & Colab Launch Commands
"""

PUZZLES = {
    20: {
        "range":   20,
        "start":   "80000",
        "pubkey":  "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c",
        "address": "1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum",
        "reward":  "0.02 BTC"
    },
    30: {
        "range":   30,
        "start":   "20000000",
        "pubkey":  "030d282cf2ff536d2c42f105d0b8588821a915dc3f9a05bd98bb23af67a2e92a5b",
        "address": "1LHtnpd8nU5VHEMKG2TMYYNUjjLc992bps",
        "reward":  "0.03 BTC"
    },
    40: {
        "range":   40,
        "start":   "8000000000",
        "pubkey":  "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4",
        "address": "1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv",
        "reward":  "0.04 BTC"
    },
    50: {
        "range":   50,
        "start":   "200000000000",
        "pubkey":  "03f46f41027bbf44fafd6b059091b900dad41e6845b2241dc3254c7cdd3c5a16c6",
        "address": "1MEzite4ReNuWaL5Ds17ePKt2dCxWEofwk",
        "reward":  "0.05 BTC"
    },
    60: {
        "range":   60,
        "start":   "800000000000000",
        "pubkey":  "0348e843dc5b1bd246e6309b4924b81543d02b16c8083df973a89ce2c7eb89a10d",
        "address": "1Kn5h2qpgw9mWE5jKpk8PP4qvvJ1QVy8su",
        "reward":  "0.60 BTC"
    },
    70: {
        "range":   70,
        "start":   "200000000000000000",
        "pubkey":  "0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483",
        "address": "19YZECXj3SxEZM1oUeJ1yiPsw8xANe7M7QR",  # fixed: removed invalid dash
        "reward":  "0.70 BTC"
    },
    80: {
        "range":   80,
        "start":   "80000000000000000000",
        "pubkey":  "02e9268c4c9894350692516e5a23c32b8886f3d9d300085a30f1ec37e3e7b2cd99",
        "address": "1NpGuN7q8E4xPuhjU82wRziB3k7jHwy2s",
        "reward":  "0.80 BTC"
    },
    # ── Puzzles #90 and #100 added ─────────────────────────────────────────────
    90: {
        "range":   90,
        "start":   "20000000000000000000000",
        "pubkey":  "02e0a8b039282faf6fe0fd769cfbc4b6b4cf8758ba68220eac420e32b91ddfa673",
        "address": "1PWCx5fovoEaoBowAvF5k6fJa1Fs6bDkHG",
        "reward":  "0.90 BTC"
    },
    100: {
        "range":   100,
        "start":   "80000000000000000000000000",
        "pubkey":  "02834b3d2f0e5f0dff50ff27f7a2d2258a1e01f0edf06a7c8b6e8bf18c48af7e12",
        "address": "1MEzite4ReNuWaL5Ds17ePKt2dCxWEofwk",  # placeholder — update when known
        "reward":  "1.00 BTC"
    },
}


def print_colab_command(puzzle_num: int, backend: str = "tpu", kangaroos: int = 131072,
                        steps_per_block: int = 50, n_blocks: int = 20):
    """Prints a ready-to-paste Google Colab launch command."""
    if puzzle_num not in PUZZLES:
        print(f"Puzzle #{puzzle_num} not found. Available: {sorted(PUZZLES.keys())}")
        return
    p = PUZZLES[puzzle_num]
    cmd = f"""
# ==============================================================================
# 🦘 GOOGLE COLAB — Bitcoin Puzzle #{p['range']} ({p['reward']})
# ==============================================================================
%cd /content
!rm -rf TPUKangaroo
!git clone https://github.com/ggsofthouse/TPUKangaroo.git
%cd TPUKangaroo

!python jax_kangaroo_tpu.py \\
  --backend {backend} \\
  --range {p['range']} \\
  --start {p['start']} \\
  --pubkey "{p['pubkey']}" \\
  --kangaroos {kangaroos} \\
  --steps-per-block {steps_per_block} \\
  --n-blocks {n_blocks} \\
  --steps 0
"""
    print(cmd)


if __name__ == "__main__":
    import sys
    puzzle_id = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    backend   = sys.argv[2] if len(sys.argv) > 2 else "tpu"
    print_colab_command(puzzle_id, backend=backend)
