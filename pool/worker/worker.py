import os
import sys
import time
import json
import re
import socket
import argparse
import platform
import subprocess
import queue
import threading
import urllib.request
import urllib.parse
from typing import Optional, Dict, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# SECP256K1 Mathematical Verification to prevent ANY false positives
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

def _point_add(p1, p2):
    if p1 is None: return p2
    if p2 is None: return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and y1 != y2: return None
    if x1 == x2:
        m = (3 * x1 * x1) * pow(2 * y1, P - 2, P) % P
    else:
        m = (y2 - y1) * pow(x2 - x1, P - 2, P) % P
    x3 = (m * m - x1 - x2) % P
    y3 = (m * (x1 - x3) - y1) % P
    return (x3, y3)

def _point_mult(k):
    res = None
    addend = (Gx, Gy)
    while k:
        if k & 1:
            res = _point_add(res, addend)
        addend = _point_add(addend, addend)
        k >>= 1
    return res

def verify_private_key(privkey_hex: str, target_pubkey_hex: str) -> bool:
    try:
        clean_priv = privkey_hex.strip()
        if clean_priv.startswith("0x") or clean_priv.startswith("0X"):
            clean_priv = clean_priv[2:]
        k = int(clean_priv, 16)
        if k <= 0:
            return False
        point = _point_mult(k)
        if not point:
            return False
        x, y = point
        prefix = "02" if y % 2 == 0 else "03"
        calc_pubkey = f"{prefix}{x:064x}".lower()
        target_clean = target_pubkey_hex.strip().lower()
        return calc_pubkey == target_clean
    except Exception:
        return False

def detect_gpus() -> str:
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            gpus = [line.strip() for line in res.stdout.strip().split("\n")]
            return ", ".join(gpus)
    except Exception:
        pass
    return "NVIDIA GPU"

def detect_gpu_mask() -> str:
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            count = len(res.stdout.strip().split("\n"))
            return "".join(str(i) for i in range(count))
    except Exception:
        pass
    return "0"

PRESETS_DISPLAY = {
    40: "Puzzle #40 (40 bits)",
    50: "Puzzle #50 (50 bits)",
    60: "Puzzle #60 (60 bits)",
    66: "Puzzle #66 (66 bits)",
    70: "Puzzle #70 (70 bits)",
    100: "Puzzle #100 (100 bits)",
    130: "Puzzle #130 (130 bits)",
    135: "Puzzle #135 (135 bits)",
    140: "Puzzle #140 (140 bits)",
    145: "Puzzle #145 (145 bits)",
    150: "Puzzle #150 (150 bits)",
}

# Parse command-line flags or fallback to environment variables
parser = argparse.ArgumentParser(description="RCKangaroo Mining Pool Worker")
parser.add_argument("--server", type=str, default=os.environ.get("POOL_SERVER_URL", "https://valyrafi.com.br"), help="Pool Coordinator Server URL")
parser.add_argument("--name", type=str, default=os.environ.get("WORKER_NAME", f"Worker-{socket.gethostname()}"), help="Worker Name")
parser.add_argument("--puzzle", type=int, default=int(os.environ.get("TARGET_PUZZLE", "140")), help="Target Bitcoin Puzzle Number")
parser.add_argument("--start-pct", type=float, default=float(os.environ.get("START_PCT", "0.0")), help="Start Range Percentage (0.0 to 100.0)")
parser.add_argument("--end-pct", type=float, default=float(os.environ.get("END_PCT", "100.0")), help="End Range Percentage (0.0 to 100.0)")
parser.add_argument("--gpu", type=str, default=os.environ.get("GPU_MASK", None), help="GPU Device Mask (e.g. 0,1 or 01)")
parser.add_argument("--token", type=str, default=os.environ.get("WORKER_TOKEN", ""), help="Token de autenticação da pool (WORKER_TOKEN)")
parser.add_argument("--non-interactive", action="store_true", help="Skip interactive prompts")

args, unknown = parser.parse_known_args()

SERVER_URL = args.server
WORKER_NAME = args.name
TARGET_PUZZLE = args.puzzle
START_PCT = args.start_pct
END_PCT = args.end_pct
GPU_MASK = args.gpu if args.gpu is not None else detect_gpu_mask()
WORKER_ID = WORKER_NAME
WORKER_TOKEN = args.token

if not WORKER_TOKEN:
    print("\n⛔ ERRO: WORKER_TOKEN não definido!")
    print("   Defina via --token SEU_TOKEN ou variável de ambiente WORKER_TOKEN")
    sys.exit(1)

# Interactive terminal prompt if user launches worker directly without explicit CLI flags
if sys.stdin and sys.stdin.isatty() and not getattr(args, 'non_interactive', False) and not any(arg.startswith("--puzzle") for arg in sys.argv[1:]):
    print("\n==================================================")
    print("🦘 RCKangaroo - Configuração da Busca")
    print("==================================================")
    print("Puzzles Bitcoin Oficiais disponíveis:")
    for num, label in PRESETS_DISPLAY.items():
        rec = " [Padrão]" if num == 66 else ""
        print(f"  [{num}] {label}{rec}")
    print("==================================================")
    
    try:
        p_in = input("\n👉 Escolha o número do Puzzle (ex: 66) [66]: ").strip()
        if p_in.isdigit() and int(p_in) in PRESETS_DISPLAY:
            TARGET_PUZZLE = int(p_in)
        else:
            TARGET_PUZZLE = 66
            
        s_in = input("👉 Porcentagem INICIAL do range (0% a 100%) [0.0]: ").strip()
        if s_in:
            START_PCT = float(s_in)
        else:
            START_PCT = 0.0
            
        e_in = input("👉 Porcentagem FINAL do range (0% a 100%) [100.0]: ").strip()
        if e_in:
            END_PCT = float(e_in)
        else:
            END_PCT = 100.0
    except (KeyboardInterrupt, EOFError):
        print("\nSaindo...")
        sys.exit(0)
    except Exception:
        pass

def http_post(endpoint: str, payload: dict, retries: int = 3) -> dict:
    url = urllib.parse.urljoin(SERVER_URL, endpoint)
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url, data=data,
        headers={
            'Content-Type': 'application/json',
            'X-Worker-Token': WORKER_TOKEN
        }
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            print(f"❌ HTTP Error connecting to pool server ({url}): {e}")
            return {}
    return {}

def http_get(endpoint: str, retries: int = 3) -> dict:
    url = urllib.parse.urljoin(SERVER_URL, endpoint)
    req = urllib.request.Request(
        url,
        headers={
            'X-Worker-Token': WORKER_TOKEN
        }
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            print(f"❌ HTTP Error connecting to pool server ({url}): {e}")
            return {}
    return {}

dp_batch_queue = queue.Queue()

def dp_uploader_loop():
    while True:
        time.sleep(10)
        items = []
        while not dp_batch_queue.empty() and len(items) < 2000:
            try:
                items.append(dp_batch_queue.get_nowait())
            except queue.Empty:
                break
        if items:
            try:
                res = http_post("/api/dp/submit_batch", {
                    "worker_name": WORKER_NAME,
                    "puzzle_number": TARGET_PUZZLE,
                    "dps": items
                })
                if res and res.get("status") == "ok":
                    total_global = res.get("total_global_dps", 0)
                    print(f"📦 [DP Pool] Enviou lote de {len(items)} DPs para a VPS. (Total Global no Servidor: {total_global:,})")
                    if res.get("solved"):
                        print(f"🎉 COLISÃO GLOBAL CONFIRMADA PELO SERVIDOR! Chave: 0x{res.get('private_key')}")
            except Exception as e:
                print(f"⚠️ Erro no upload de lote de DPs: {e}")

_dp_uploader_thread = threading.Thread(target=dp_uploader_loop, daemon=True)
_dp_uploader_thread.start()

def ensure_tames_file(puzzle_number: int, bin_dir: str) -> Optional[str]:
    """Verifica se o servidor da pool possui arquivo de tames pré-gerado para este puzzle.
    Se sim, baixa para o diretório do binário RCKangaroo.
    """
    try:
        tame_info = http_get(f"/api/tames/check/{puzzle_number}")
        if tame_info and tame_info.get("exists"):
            filename = tame_info["filename"]
            download_url = tame_info["download_url"]
            local_path = os.path.join(bin_dir, filename)
            
            if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                print(f"🦘 Baixando tames pré-gerados ({tame_info.get('size_mb', 0)} MB) da Pool...")
                full_url = urllib.parse.urljoin(SERVER_URL, download_url)
                req = urllib.request.Request(full_url, headers={"X-Worker-Token": WORKER_TOKEN})
                with urllib.request.urlopen(req, timeout=180) as resp, open(local_path, "wb") as f:
                    f.write(resp.read())
                print(f"✅ Tames baixados com sucesso: {filename}")
            else:
                print(f"🦘 Usando tames pré-gerados locais: {filename}")
            return filename
    except Exception as e:
        print(f"ℹ️ Conectado sem tames pré-gerados para Puzzle #{puzzle_number} (rodando no modo normal)")
    return None

def get_binary_path() -> str:
    system = platform.system()
    current_dir = os.path.abspath(os.getcwd())
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    if system == "Windows":
        bin_path = os.path.join(root_dir, "RCKangaroo.exe")
        if not os.path.exists(bin_path):
            bin_path = os.path.join(root_dir, "x64", "Release", "RCKangaroo.exe")
        if not os.path.exists(bin_path):
            bin_path = os.path.join(root_dir, "build", "Release", "RCKangaroo.exe")
        if not os.path.exists(bin_path):
            bin_path = os.path.join(current_dir, "RCKangaroo.exe")
        return bin_path
    else:
        # Linux execution (Lightning.ai / Vast.ai / Colab / Kaggle / Modal)
        # Ensure CUDA libraries (libcuda.so.1, libcudart.so) are in LD_LIBRARY_PATH
        cuda_dirs = [
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/lib64/stubs",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/local/nvidia/lib",
            "/usr/local/nvidia/lib64"
        ]
        valid_paths = [d for d in cuda_dirs if os.path.exists(d)]
        if valid_paths:
            cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = ":".join(valid_paths) + (f":{cur_ld}" if cur_ld else "")

        bin_path = os.path.join(current_dir, "rckangaroo")
        if not os.path.exists(bin_path):
            if not os.path.exists(os.path.join(current_dir, "RCKangaroo.cpp")):
                print("⚙️ Baixando código-fonte C++/CUDA do RCKangaroo...")
                subprocess.run("git clone https://github.com/RetiredC/RCKangaroo.git /tmp/rck_src && cp -r /tmp/rck_src/* . && rm -rf /tmp/rck_src", shell=True, check=True)
            
            print("⚙️ Compilando binário RCKangaroo para Linux (nvcc -arch=native)...")
            ld_flags = " ".join([f"-L{d}" for d in valid_paths])
            try:
                build_cmd = f"nvcc -O3 -std=c++17 -arch=native -o rckangaroo RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp CallCubin.cpp RCGpuCore.cu {ld_flags} -lcuda -lcudart -lpthread"
                subprocess.run(build_cmd, shell=True, check=True)
            except Exception:
                build_cmd = f"nvcc -O3 -std=c++17 -o rckangaroo RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp CallCubin.cpp RCGpuCore.cu {ld_flags} -lcuda -lcudart -lpthread"
                subprocess.run(build_cmd, shell=True, check=True)
        return bin_path


def enqueue_output(out, q):
    for line in iter(out.readline, ''):
        if line and line.strip():
            q.put(line)
    out.close()

def main():
    print(f"\n==================================================")
    print(f"🦘 RCKangaroo Worker Node Iniciado")
    print(f"Servidor Pool: {SERVER_URL}")
    print(f"Nome Worker:   {WORKER_NAME}")
    print(f"Puzzle Alvo:   #{TARGET_PUZZLE} (Intervalo {START_PCT}% -> {END_PCT}%)")
    print(f"Worker ID:     {WORKER_ID}")
    print(f"OS:            {platform.system()} {platform.release()}")
    
    gpu_info = detect_gpus()
    print(f"GPU Hardware:  {gpu_info}")
    
    bin_path = get_binary_path()
    if not os.path.exists(bin_path):
        print(f"❌ Error: Binario RCKangaroo nao encontrado em: {bin_path}")
        sys.exit(1)
    
    print(f"Binario Path:  {bin_path}")
    print(f"==================================================")

    results_file = os.path.join(os.path.dirname(bin_path), "RESULTS.TXT")
    if os.path.exists(results_file):
        try:
            os.remove(results_file)
            print("🧹 Old RESULTS.TXT removed.")
        except Exception:
            pass

    # Persistent hashrate across chunks
    last_known_mhs = 0.0

    # Register worker with pool server
    http_post("/api/worker/register", {
        "worker_id": WORKER_ID,
        "name": WORKER_NAME,
        "os_info": f"{platform.system()} {platform.release()}",
        "gpu_info": gpu_info
    })

    # Ensure target puzzle job matches requested choice on pool server
    http_post("/api/worker/ensure_job", {
        "puzzle_number": TARGET_PUZZLE,
        "start_percent": START_PCT,
        "end_percent": END_PCT
    })

    while True:
        try:
            # Request work chunk
            work = http_post("/api/worker/get_work", {
                "worker_id": WORKER_ID,
                "name": WORKER_NAME,
                "hashrate_mhs": last_known_mhs
            })
            
            if not work or work.get("status") == "NO_WORK":
                print(f"⚙️ Ativando Puzzle #{TARGET_PUZZLE} ({START_PCT}% -> {END_PCT}%) no servidor...")
                http_post("/api/worker/ensure_job", {
                    "puzzle_number": TARGET_PUZZLE,
                    "start_percent": START_PCT,
                    "end_percent": END_PCT
                })
                time.sleep(2)
                continue

            if work.get("status") == "WORK_ASSIGNED":
                chunk_id = work["chunk_id"]
                pubkey = work["pubkey"]
                start_hex = work["start_hex"]
                range_bits = work["range_bits"]
                # dp_bits e max_ops calculados dinamicamente pelo servidor
                # (dp = max(14, chunk_bits//2 - 2) | max = chunk_bits/range_bits * 2.5)
                dp_bits = work.get("dp_bits", 31)
                max_ops = work.get("max_ops", "1.62")

                print(f"\n🚀 Recebido Sub-bloco de Trabalho: {chunk_id}")
                print(f"   Pubkey Alvo:  {pubkey}")
                print(f"   Start Hex:    0x{start_hex}")
                print(f"   Range Bits:   {range_bits} bits")
                print(f"   DP Bits:      {dp_bits}")

                # Ensure RESULTS.TXT is deleted before running chunk
                if os.path.exists(results_file):
                    try:
                        os.remove(results_file)
                    except Exception:
                        pass

                # Check and download pre-generated tames file if available on pool server
                tame_file = ensure_tames_file(TARGET_PUZZLE, os.path.dirname(bin_path))

                # Build executable command line
                cmd = [
                    bin_path,
                    "-gpu", GPU_MASK,
                    "-dp", str(dp_bits),
                    "-range", str(range_bits),
                    "-start", start_hex,
                    "-pubkey", pubkey,
                    "-max", str(max_ops),
                    "-stream-dps"
                ]
                if tame_file:
                    cmd.extend(["-tames", tame_file])


                print(f"   Executando: {' '.join(cmd)}")
                chunk_start_time = time.time()

                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=os.path.dirname(bin_path), env=os.environ, bufsize=1)

                
                out_q = queue.Queue()
                reader_thread = threading.Thread(target=enqueue_output, args=(proc.stdout, out_q))
                reader_thread.daemon = True
                reader_thread.start()

                last_heartbeat = time.time()
                found_key = None
                last_known_dps = 0

                while proc.poll() is None or not out_q.empty():
                    try:
                        line_str = out_q.get_nowait()
                    except queue.Empty:
                        time.sleep(0.1)
                        line_str = None
                    
                    if line_str:
                        if line_str.strip():
                            line_str = line_str.strip()
                            if line_str.startswith("DP_ENTRY:"):
                                parts = line_str.split(":")
                                if len(parts) >= 4:
                                    try:
                                        dp_type = int(parts[1])
                                        x_prefix = parts[2]
                                        dist_hex = parts[3]
                                        dp_batch_queue.put({
                                            "x_prefix": x_prefix,
                                            "dist_hex": dist_hex,
                                            "dp_type": dp_type
                                        })
                                    except Exception:
                                        pass
                            else:
                                print(f"[RCK] {line_str}")

                            # Parse Hashrate
                            mhs_match = re.search(r'Speed:\s*(\d+(?:\.\d+)?)\s*MKeys', line_str, re.IGNORECASE)
                            if mhs_match:
                                last_known_mhs = float(mhs_match.group(1))
                            else:
                                ghs_match = re.search(r'Speed:\s*(\d+(?:\.\d+)?)\s*GKeys', line_str, re.IGNORECASE)
                                if ghs_match:
                                    last_known_mhs = float(ghs_match.group(1)) * 1000.0
                                else:
                                    khs_match = re.search(r'Speed:\s*(\d+(?:\.\d+)?)\s*KKeys', line_str, re.IGNORECASE)
                                    if khs_match:
                                        last_known_mhs = float(khs_match.group(1)) / 1000.0

                            # Parse DPs count
                            dp_match = re.search(r'DPs:\s*([\d\.]+[KMG]?)', line_str, re.IGNORECASE)
                            if dp_match:
                                dp_raw = dp_match.group(1).upper()
                                try:
                                    if dp_raw.endswith('K'):
                                        last_known_dps = int(float(dp_raw[:-1]) * 1000)
                                    elif dp_raw.endswith('M'):
                                        last_known_dps = int(float(dp_raw[:-1]) * 1000000)
                                    elif dp_raw.endswith('G'):
                                        last_known_dps = int(float(dp_raw[:-1]) * 1000000000)
                                    else:
                                        last_known_dps = int(float(dp_raw))
                                except Exception:
                                    pass

                            # Parse potential solution line from stdout
                            if "PRIVATE KEY:" in line_str:
                                candidate = line_str.split("PRIVATE KEY:")[-1].strip()
                                # Mathematically verify against target pubkey
                                if verify_private_key(candidate, pubkey):
                                    found_key = candidate
                                else:
                                    print(f"⚠️ Falso positivo descartado pelo worker: {candidate}")

                    # Heartbeat a cada 30s — reduz pressão no SQLite com chunks grandes
                    if time.time() - last_heartbeat >= 30:
                        http_post("/api/worker/heartbeat", {
                            "worker_id": WORKER_ID,
                            "hashrate_mhs": last_known_mhs,
                            "dps_count": last_known_dps
                        })
                        last_heartbeat = time.time()

                    if proc.poll() is not None and out_q.empty():
                        # Final heartbeat for this chunk run
                        http_post("/api/worker/heartbeat", {
                            "worker_id": WORKER_ID,
                            "hashrate_mhs": last_known_mhs
                        })
                        break

                    time.sleep(0.1)

                # Check RESULTS.TXT if generated during this run
                if not found_key and os.path.exists(results_file) and os.path.getmtime(results_file) >= (chunk_start_time - 1):
                    with open(results_file, "r") as rf:
                        for rline in rf.read().split("\n"):
                            if "PRIVATE KEY:" in rline:
                                candidate = rline.split("PRIVATE KEY:")[-1].strip()
                                if verify_private_key(candidate, pubkey):
                                    found_key = candidate
                                else:
                                    print(f"⚠️ Falso positivo em RESULTS.TXT descartado: {candidate}")

                if found_key:
                    print(f"\n🌟 REAL CHAVE PRIVADA VERIFICADA MATEMATICAMENTE: {found_key}! Enviando ao servidor...")
                    http_post("/api/worker/submit_solution", {
                        "worker_id": WORKER_ID,
                        "chunk_id": chunk_id,
                        "pubkey": pubkey,
                        "private_key": found_key
                    })
                    print("🎉 Solucao enviada ao servidor com sucesso!")
                    break

                # Confirma explicitamente a conclusão do chunk ao servidor
                http_post("/api/worker/complete_chunk", {
                    "worker_id": WORKER_ID,
                    "chunk_id": chunk_id
                })
                print(f"✅ Chunk {chunk_id} concluido e confirmado. Solicitando proximo...\n")

        except KeyboardInterrupt:
            print("\n🛑 Worker interrompido pelo usuario.")
            sys.exit(0)
        except Exception as e:
            print(f"⚠️ Erro no Worker: {e}. Tentando novamente em 5s...")
            time.sleep(5)

if __name__ == "__main__":
    main()
