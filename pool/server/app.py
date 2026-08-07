import os
import time
import json
import sqlite3
import shutil
import hashlib
import secrets
import threading
import math
import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

app = FastAPI(title="RCKangaroo Distributed Pool Coordinator", version="6.1")

VPS_LOGS: List[str] = []

def add_vps_log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    VPS_LOGS.append(entry)
    if len(VPS_LOGS) > 100:
        VPS_LOGS.pop(0)

def log_event(msg: str):
    add_vps_log(msg)

@app.get("/api/logs")
def get_vps_logs():
    return {"logs": VPS_LOGS}

@app.get("/worker.py")
def download_worker_script():
    worker_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "worker", "worker.py"))
    return FileResponse(worker_file, media_type="text/x-python", filename="worker.py")

@app.get("/logo.png")
def serve_logo_image():
    logo_file = os.path.join(os.path.dirname(__file__), "logo.png")
    if os.path.exists(logo_file):
        return FileResponse(logo_file, media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo file not found")


TAMES_DIR = os.path.join(os.path.dirname(__file__), "tames")
os.makedirs(TAMES_DIR, exist_ok=True)

@app.get("/tames/{filename}")
def download_tame_file(filename: str):
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(TAMES_DIR, safe_filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Tame file not found")
    return FileResponse(filepath, media_type="application/octet-stream", filename=safe_filename)

@app.get("/api/tames/check/{puzzle_number}")
def check_tame_file(puzzle_number: int):
    filename = f"tames{puzzle_number}.dat"
    filepath = os.path.join(TAMES_DIR, filename)
    exists = os.path.exists(filepath)
    size_bytes = os.path.getsize(filepath) if exists else 0
    return {
        "exists": exists,
        "filename": filename if exists else None,
        "download_url": f"/tames/{filename}" if exists else None,
        "size_mb": round(size_bytes / (1024 * 1024), 2) if exists else 0
    }

# CORS restrito ao domínio configurado na VPS
_ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_ALLOWED_ORIGIN] if _ALLOWED_ORIGIN else [],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

import collections

server_logs = collections.deque(maxlen=150)

def log_event(msg: str):
    timestamp = time.strftime('%H:%M:%S', time.localtime())
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    server_logs.append(formatted)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "pool.db")
db_lock = threading.RLock()
security = HTTPBasic()

# ─── Credenciais do Dashboard (nunca hardcode — lidas do ambiente da VPS) ──────
DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "")
DASHBOARD_PASS = os.environ.get("DASHBOARD_PASS", "")
if not DASHBOARD_USER or not DASHBOARD_PASS:
    raise RuntimeError(
        "⛔ DASHBOARD_USER e DASHBOARD_PASS devem estar definidos como variáveis de ambiente. "
        "Defina-os no arquivo .env da VPS ou via docker-compose."
    )

# ─── Token de autenticação para workers ─────────────────────────────────────────
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
if not WORKER_TOKEN:
    raise RuntimeError(
        "⛔ WORKER_TOKEN deve estar definido como variável de ambiente. "
        "Configure-o no .env da VPS e no start_worker.sh dos containers."
    )

WORKER_ALIVE_SECONDS = 120

def verify_worker_token(request: Request):
    """Dependência usada em todos os endpoints de worker para validar o token."""
    token = request.headers.get("X-Worker-Token", "")
    if not token:
        token = request.query_params.get("token", "")
    if not secrets.compare_digest(token, WORKER_TOKEN):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token de worker inválido")

def authenticate_dashboard(credentials: HTTPBasicCredentials = Depends(security)):
    is_correct_username = secrets.compare_digest(credentials.username, DASHBOARD_USER)
    is_correct_password = secrets.compare_digest(credentials.password, DASHBOARD_PASS)
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Acesso Restrito: Usuario ou senha incorretos",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Base58 and WIF key conversion utilities
BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def base58_encode(b: bytes) -> str:
    n = int.from_bytes(b, 'big')
    res = ''
    while n > 0:
        n, r = divmod(n, 58)
        res = BASE58_ALPHABET[r] + res
    for byte in b:
        if byte == 0:
            res = '1' + res
        else:
            break
    return res

def hex_to_wif(hex_key: str, compressed: bool = True) -> str:
    clean_hex = hex_key.strip()
    if clean_hex.startswith("0x") or clean_hex.startswith("0X"):
        clean_hex = clean_hex[2:]
    clean_hex = clean_hex.zfill(64)
    raw_key = bytes.fromhex(clean_hex)
    extended = b'\x80' + raw_key
    if compressed:
        extended += b'\x01'
    first_sha = hashlib.sha256(extended).digest()
    second_sha = hashlib.sha256(first_sha).digest()
    checksum = second_sha[:4]
    return base58_encode(extended + checksum)

def pubkey_to_address(pubkey_hex: str) -> str:
    try:
        clean = pubkey_hex.strip()
        if clean.startswith("0x") or clean.startswith("0X"):
            clean = clean[2:]
        pub_bytes = bytes.fromhex(clean)
        sha256_res = hashlib.sha256(pub_bytes).digest()
        ripemd160_res = hashlib.new('ripemd160', sha256_res).digest()
        extended = b'\x00' + ripemd160_res
        checksum = hashlib.sha256(hashlib.sha256(extended).digest()).digest()[:4]
        return base58_encode(extended + checksum)
    except Exception:
        return "N/A"


# SECP256K1 Verification Math
_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
_Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
_Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

def _point_add(p1, p2):
    if p1 is None: return p2
    if p2 is None: return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and y1 != y2: return None
    if x1 == x2:
        m = (3 * x1 * x1) * pow(2 * y1, _P - 2, _P) % _P
    else:
        m = (y2 - y1) * pow(x2 - x1, _P - 2, _P) % _P
    x3 = (m * m - x1 - x2) % _P
    y3 = (m * (x1 - x3) - y1) % _P
    return (x3, y3)

def _point_mult(k):
    res = None
    addend = (_Gx, _Gy)
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

def get_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=60.0)
    conn.execute("PRAGMA busy_timeout = 60000;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db_lock:
        try:
            backup_dir = os.path.join(BASE_DIR, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            if os.path.exists(DB_FILE) and os.path.getsize(DB_FILE) > 0:
                backup_path = os.path.join(backup_dir, f"pool_auto_{int(time.time())}.db")
                shutil.copy2(DB_FILE, backup_path)
                print(f"📦 Backup automático do banco criado em: {backup_path}")
        except Exception as e:
            print(f"⚠️ Erro ao criar backup automático: {e}")

        conn = get_db()
        cursor = conn.cursor()
        
        # Jobs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                pubkey TEXT NOT NULL,
                start_hex TEXT NOT NULL,
                range_bits INTEGER NOT NULL,
                dp_bits INTEGER DEFAULT 16,
                chunk_bits INTEGER DEFAULT 66,
                start_percent REAL DEFAULT 0.0,
                end_percent REAL DEFAULT 100.0,
                base_start_hex TEXT NOT NULL,
                current_offset_hex TEXT NOT NULL,
                status TEXT DEFAULT 'ACTIVE',
                created_at REAL,
                solved_at REAL,
                solved_by TEXT,
                private_key TEXT
            )
        ''')

        # Work units / Chunks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                start_hex TEXT NOT NULL,
                range_bits INTEGER NOT NULL,
                assigned_worker TEXT,
                assigned_at REAL,
                status TEXT DEFAULT 'PENDING',
                last_heartbeat REAL
            )
        ''')

        # Workers
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                name TEXT,
                os_info TEXT,
                gpu_info TEXT,
                hashrate_mhs REAL DEFAULT 0.0,
                last_ping REAL,
                completed_chunks INTEGER DEFAULT 0
            )
        ''')

        # Auto-migrate existing jobs table if new columns are missing
        columns_to_add = [
            ("start_percent", "REAL DEFAULT 0.0"),
            ("end_percent", "REAL DEFAULT 100.0"),
            ("base_start_hex", "TEXT DEFAULT '0'")
        ]
        for col_name, col_type in columns_to_add:
            try:
                cursor.execute(f"ALTER TABLE jobs ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass

        try:
            cursor.execute("ALTER TABLE workers ADD COLUMN current_job_id TEXT")
        except Exception:
            pass

        try:
            cursor.execute("ALTER TABLE workers ADD COLUMN dps_count INTEGER DEFAULT 0")
        except Exception:
            pass

        # Adiciona colunas de controle de conclusão real nos chunks
        chunks_cols = [
            ("completed_at", "REAL"),
            ("heartbeat_at", "REAL"),
            ("last_heartbeat", "REAL")
        ]
        for col_name, col_type in chunks_cols:
            try:
                cursor.execute(f"ALTER TABLE chunks ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass

        # Tabela global de DPs acumulados por todos os workers
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS global_dps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                puzzle_number INTEGER DEFAULT 140,
                x_prefix TEXT NOT NULL,
                dist_hex TEXT NOT NULL,
                dp_type INTEGER DEFAULT 0,
                worker_name TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        ''')
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_global_dps_x ON global_dps(puzzle_number, x_prefix)")

        conn.commit()

        # WAL mode: múltiplos leitores simultâneos sem bloquear escritas
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.commit()
        conn.close()

GLOBAL_DP_CACHE: Dict[tuple, List[dict]] = {}

def load_global_dp_cache():
    global GLOBAL_DP_CACHE
    with db_lock:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT puzzle_number, x_prefix, dist_hex, dp_type, worker_name FROM global_dps")
        rows = cursor.fetchall()
        cache = {}
        for r in rows:
            key = (int(r["puzzle_number"]), str(r["x_prefix"]).lower())
            if key not in cache:
                cache[key] = []
            cache[key].append({
                "dist_hex": str(r["dist_hex"]).lower(),
                "type": int(r["dp_type"]),
                "worker": str(r["worker_name"])
            })
        GLOBAL_DP_CACHE = cache
        print(f"🧠 Cache de DPs Globais carregado na memória: {len(rows)} DPs totais.")
        conn.close()

init_db()
load_global_dp_cache()

# ─── Background: recoloca chunks abandonados de volta para PENDING ────────────
# Timeout configurável por env var: chunks grandes (90-bits) podem durar horas
CHUNK_TIMEOUT_SECONDS = int(os.environ.get("CHUNK_TIMEOUT_SECONDS", "3600"))  # default 1h
WORKER_ALIVE_SECONDS  = int(os.environ.get("WORKER_ALIVE_SECONDS",  "90"))    # heartbeat threshold

def _recover_abandoned_chunks():
    """Thread que roda a cada 60s e recupera chunks ASSIGNED cujo worker sumiu."""
    while True:
        time.sleep(60)
        try:
            with db_lock:
                conn = get_db()
                cursor = conn.cursor()
                now = time.time()
                cutoff = now - CHUNK_TIMEOUT_SECONDS

                # Chunks ASSIGNED mas cujo worker não fez ping nos últimos CHUNK_TIMEOUT_SECONDS
                cursor.execute('''
                    UPDATE chunks SET status = 'PENDING', assigned_worker = NULL, assigned_at = NULL
                    WHERE status = 'ASSIGNED'
                    AND chunk_id IN (
                        SELECT c.chunk_id FROM chunks c
                        LEFT JOIN workers w ON w.worker_id = c.assigned_worker
                        WHERE c.status = 'ASSIGNED'
                        AND (w.last_ping IS NULL OR w.last_ping < ?)
                    )
                ''', (cutoff,))

                recovered = cursor.rowcount
                if recovered > 0:
                    print(f"♻️  Recovered {recovered} abandoned chunk(s) back to PENDING (timeout={CHUNK_TIMEOUT_SECONDS}s)")

                conn.commit()
                conn.close()
        except Exception as e:
            print(f"⚠️  Error in chunk recovery thread: {e}")

_recovery_thread = threading.Thread(target=_recover_abandoned_chunks, daemon=True)
_recovery_thread.start()

# Pydantic Schemas
class CreateJobRequest(BaseModel):
    pubkey: Optional[str] = None
    puzzle_number: Optional[int] = None
    start_percent: float = 0.0
    end_percent: float = 100.0
    start_hex: Optional[str] = None
    range_bits: Optional[int] = None
    dp_bits: int = 18
    chunk_bits: int = 66
    worker_id: Optional[str] = None

class WorkerRegisterRequest(BaseModel):
    worker_id: str
    name: str
    os_info: str
    gpu_info: str

class WorkerHeartbeatRequest(BaseModel):
    worker_id: str
    hashrate_mhs: float
    dps_count: Optional[int] = 0

class WorkRequest(BaseModel):
    worker_id: str
    name: Optional[str] = None
    hashrate_mhs: Optional[float] = None

class ChunkHeartbeatRequest(BaseModel):
    worker_id: str
    chunk_id: str
    hashrate_mhs: float = 0.0

class CompleteChunkRequest(BaseModel):
    worker_id: str
    chunk_id: str

class SubmitSolutionRequest(BaseModel):
    worker_id: str
    chunk_id: str
    pubkey: str
    private_key: str

class DPItem(BaseModel):
    x_prefix: str
    dist_hex: str
    dp_type: int = 0

class DPBatchSubmit(BaseModel):
    worker_name: str
    puzzle_number: int = 140
    dps: List[DPItem]

# Preset Puzzles Dictionary (Dados Oficiais dos Desafios Bitcoin)
# chunk_bits: tamanho de cada fatia de trabalho por puzzle
#   Puzzles pequenos (≤66 bits): chunks menores (~66 bits), completam em segundos/minutos
#   Puzzles grandes (130+ bits): chunks de 80-90 bits, cada um dura horas por worker
#   dp (é calculado dinamicamente em get_work): dp = max(14, (chunk_bits//2) - 2)
PUZZLE_PRESETS = {
    40: {
        "pubkey": "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4",
        "bits": 40,
        "base_start": "8000000000",
        "chunk_bits": 34
    },
    50: {
        "pubkey": "03f46f41027bbf44fafd6b059091b900dad41e6845b2241dc3254c7cdd3c5a16c6",
        "bits": 50,
        "base_start": "200000000000",
        "chunk_bits": 40
    },
    60: {
        "pubkey": "0348e843dc5b1bd246e6309b4924b81543d02b16c8083df973a89ce2c7eb89a10d",
        "bits": 60,
        "base_start": "800000000000000",
        "chunk_bits": 50
    },
    66: {
        "pubkey": "024ee2be2d4e9f92d2f5a4a03058617dc45befe22938feed5b7a6b7282dd74cbdd",
        "bits": 66,
        "base_start": "20000000000000000",
        "chunk_bits": 66
    },
    100: {
        "pubkey": "035c38bd9ae4b10e8a250857006f3cfd98ab15a6196d9f4dfd25bc7ecc77d788d5",
        "bits": 100,
        "base_start": "20000000000000000000000",
        "chunk_bits": 75
    },
    130: {
        "pubkey": "03633cbe3ec02b9401c5effa144c5b4d22f87940259634858fc7e59b1c09937852",
        "bits": 130,
        "base_start": "20000000000000000000000000000000",
        "chunk_bits": 80   # ~15-30 min por chunk em GPU potente
    },
    135: {
        "pubkey": "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16",
        "bits": 135,
        "base_start": "4000000000000000000000000000000004",
        "chunk_bits": 80
    },
    140: {
        "pubkey": "031f6a332d3c5c4f2de2378c012f429cd109ba07d69690c6c701b6bb87860d6640",
        "bits": 140,
        "base_start": "80000000000000000000000000000000000",
        "chunk_bits": 90   # ~90 bits por chunk para cobrir a janela ativa de busca
    },
    145: {
        "pubkey": "03afdda497369e219a2c1c369954a930e4d3740968e5e4352475bcffce3140dae5",
        "bits": 145,
        "base_start": "1000000000000000000000000000000000000",
        "chunk_bits": 80
    },
    150: {
        "pubkey": "03137807790ea7dc6e97901c2bc87411f45ed74a5629315c4e4b03a0a102250c49",
        "bits": 150,
        "base_start": "20000000000000000000000000000000000000",
        "chunk_bits": 80
    }
}

def internal_create_job(req: CreateJobRequest) -> str:
    with db_lock:
        conn = get_db()
        cursor = conn.cursor()
        job_id = f"job_{int(time.time() * 1000)}_{secrets.token_hex(2)}"
        
        start_pct = max(0.0, min(100.0, req.start_percent))
        end_pct = max(start_pct, min(100.0, req.end_percent))

        if req.puzzle_number in PUZZLE_PRESETS:
            preset = PUZZLE_PRESETS[req.puzzle_number]
            pubkey = preset["pubkey"]
            bits = preset["bits"]
            base_start_int = int(preset["base_start"], 16)
            total_range_int = 1 << (bits - 1)
            
            start_offset_int = base_start_int + int((start_pct / 100.0) * total_range_int)
            start_hex = hex(start_offset_int)[2:]
            base_start_hex = preset["base_start"]
            range_bits = bits - 1
            chunk_bits_to_use = preset.get("chunk_bits", req.chunk_bits)
            dp_bits = 0  # será calculado dinamicamente em get_work
        else:
            pubkey = req.pubkey.lower().strip() if req.pubkey else PUZZLE_PRESETS[66]["pubkey"]
            clean_start = req.start_hex.lower().replace("0x", "").strip() if req.start_hex else "20000000000000000"
            start_offset_int = int(clean_start, 16)
            range_bits = req.range_bits if req.range_bits else 66
            total_range_int = 1 << range_bits
            start_offset_int += int((start_pct / 100.0) * total_range_int)
            start_hex = hex(start_offset_int)[2:]
            base_start_hex = clean_start
            chunk_bits_to_use = req.chunk_bits
            dp_bits = req.dp_bits

        cursor.execute('''
            INSERT INTO jobs (job_id, pubkey, start_hex, range_bits, dp_bits, chunk_bits, start_percent, end_percent, base_start_hex, current_offset_hex, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (job_id, pubkey.lower(), start_hex, range_bits, dp_bits, chunk_bits_to_use, start_pct, end_pct, base_start_hex, start_hex, time.time()))
        
        conn.commit()
        conn.close()
        print(f"🚀 Job Created: {job_id} for Puzzle {req.puzzle_number or 'Custom'} (Start Hex: {start_hex})")
        return job_id

# Protected Dashboard API
@app.post("/api/jobs/create")
def create_job(req: CreateJobRequest, username: str = Depends(authenticate_dashboard)):
    job_id = internal_create_job(req)
    return {"status": "SUCCESS", "job_id": job_id}

# ─── Endpoints do Banco Global de DPs ──────────────────────────────────────────
@app.post("/api/dp/submit_batch")
def submit_dp_batch(data: DPBatchSubmit, request: Request):
    verify_worker_token(request)
    if not data.dps:
        return {"status": "ok", "ingested": 0, "total_global_dps": sum(len(v) for v in GLOBAL_DP_CACHE.values()), "solved": False}

    solved = False
    solved_key = None
    ingested_count = 0
    now = time.time()

    with db_lock:
        conn = get_db()
        cursor = conn.cursor()

        job_row = cursor.execute(
            "SELECT pubkey, job_id FROM jobs WHERE status='ACTIVE' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        target_pubkey = job_row["pubkey"] if job_row else None
        job_id = job_row["job_id"] if job_row else None

        db_records = []
        for dp in data.dps:
            x_clean = dp.x_prefix.strip().lower()
            d_clean = dp.dist_hex.strip().lower()
            dp_type = int(dp.dp_type)
            cache_key = (data.puzzle_number, x_clean)

            if cache_key in GLOBAL_DP_CACHE:
                for existing in GLOBAL_DP_CACHE[cache_key]:
                    if existing["type"] != dp_type or existing["dist_hex"] != d_clean:
                        try:
                            dist1 = int(existing["dist_hex"], 16)
                            dist2 = int(d_clean, 16)
                            candidates = []
                            if dist1 > dist2:
                                candidates.append(f"{dist1 - dist2:064x}")
                                candidates.append(f"{dist2 - dist1:064x}")
                            else:
                                candidates.append(f"{dist2 - dist1:064x}")
                                candidates.append(f"{dist1 - dist2:064x}")

                            if target_pubkey:
                                for cand in candidates:
                                    if verify_private_key(cand, target_pubkey):
                                        solved = True
                                        solved_key = cand
                                        log_event(f"🎉 COLISÃO GLOBAL DE DP DETECTADA! Worker: {data.worker_name} | Chave: 0x{cand}")
                                        if job_id:
                                            cursor.execute(
                                                "UPDATE jobs SET status='SOLVED', solved_at=?, solved_by=?, private_key=? WHERE job_id=?",
                                                (now, data.worker_name, cand, job_id)
                                            )
                                            results_path = os.path.join(BASE_DIR, "POOL_RESULTS.TXT")
                                            wif = hex_to_wif(cand)
                                            addr = pubkey_to_address(target_pubkey)
                                            with open(results_path, "a", encoding="utf-8") as f:
                                                f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] PUZZLE #{data.puzzle_number} RESOLVIDO VIA DP GLOBAL!\n")
                                                f.write(f"  Worker: {data.worker_name}\n")
                                                f.write(f"  Pubkey: {target_pubkey}\n")
                                                f.write(f"  PrivKey Hex: 0x{cand}\n")
                                                f.write(f"  WIF: {wif}\n")
                                                f.write(f"  Endereco BTC: {addr}\n")
                                        break
                        except Exception as ex:
                            print(f"⚠️ Erro ao calcular candidato de colisão DP: {ex}")

            db_records.append((data.puzzle_number, x_clean, d_clean, dp_type, data.worker_name, now))
            if cache_key not in GLOBAL_DP_CACHE:
                GLOBAL_DP_CACHE[cache_key] = []
            GLOBAL_DP_CACHE[cache_key].append({
                "dist_hex": d_clean,
                "type": dp_type,
                "worker": data.worker_name
            })
            ingested_count += 1

        if db_records:
            cursor.executemany(
                "INSERT INTO global_dps (puzzle_number, x_prefix, dist_hex, dp_type, worker_name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                db_records
            )
            cursor.execute(
                "UPDATE workers SET dps_count = dps_count + ? WHERE name = ?",
                (ingested_count, data.worker_name)
            )
            conn.commit()

        conn.close()

    total_cached = sum(len(v) for v in GLOBAL_DP_CACHE.values())
    return {
        "status": "ok",
        "ingested": ingested_count,
        "total_global_dps": total_cached,
        "solved": solved,
        "private_key": solved_key
    }

@app.get("/api/dp/stats")
def get_dp_stats():
    total_cached = sum(len(v) for v in GLOBAL_DP_CACHE.values())
    tames = 0
    wilds = 0
    worker_counts = {}
    for (puzz, x), items in GLOBAL_DP_CACHE.items():
        for item in items:
            if item["type"] == 0:
                tames += 1
            else:
                wilds += 1
            w = item["worker"]
            worker_counts[w] = worker_counts.get(w, 0) + 1

    return {
        "total_dps": total_cached,
        "tames_count": tames,
        "wilds_count": wilds,
        "worker_contributions": worker_counts
    }

@app.post("/api/jobs/delete/{job_id}")
def delete_single_job(job_id: str, username: str = Depends(authenticate_dashboard)):
    with db_lock:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        cursor.execute("DELETE FROM chunks WHERE job_id = ?", (job_id,))
        conn.commit()
        conn.close()
    return {"status": "DELETED", "job_id": job_id}

@app.post("/api/jobs/clear")
def clear_jobs(username: str = Depends(authenticate_dashboard)):
    with db_lock:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM jobs")
        cursor.execute("DELETE FROM chunks")
        conn.commit()
        conn.close()
    return {"status": "CLEARED"}

def generate_coverage_report():
    with db_lock:
        conn = get_db()
        cursor = conn.cursor()
        now = time.time()
        
        cursor.execute("SELECT * FROM jobs WHERE status IN ('ACTIVE', 'SOLVED', 'COMPLETED') ORDER BY start_percent ASC")
        jobs = [dict(j) for j in cursor.fetchall()]
        
        cursor.execute("SELECT * FROM workers WHERE (? - last_ping) < ?", (now, WORKER_ALIVE_SECONDS))
        raw_w = cursor.fetchall()
        active_workers = {dict(w)['current_job_id']: dict(w)['name'] for w in raw_w if dict(w).get('current_job_id')}
        
        report_lines = []
        report_lines.append("======================================================================")
        report_lines.append("       RCKangaroo Pool - Relatório de Cobertura & Anti-Sobreposição   ")
        report_lines.append(f"       Atualizado em: {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        report_lines.append("======================================================================\n")

        scanned_ranges = []
        active_ranges = []
        all_covered_intervals = []

        if jobs:
            report_lines.append("✅ [FAIXAS JÁ VARRIDAS E CONCLUÍDAS (EXCLUÍDAS DE RETRABALHO)]")
            scanned_count = 0
            for j in jobs:
                s_pct = float(j.get('start_percent', 0.0))
                e_pct = float(j.get('end_percent', 100.0))
                
                scanned_pct = s_pct
                try:
                    start_offset_int = int(j.get('start_hex', '0'), 16)
                    current_offset_int = int(j.get('current_offset_hex', j.get('start_hex', '0')), 16)
                    bits = j.get('range_bits', 139)
                    total_range_int = 1 << bits
                    if total_range_int > 0 and current_offset_int > start_offset_int:
                        progress_ratio = (current_offset_int - start_offset_int) / float(total_range_int)
                        scanned_pct = min(e_pct, s_pct + progress_ratio * (e_pct - s_pct))
                except Exception:
                    scanned_pct = s_pct

                if j['status'] in ('SOLVED', 'COMPLETED'):
                    scanned_pct = e_pct

                if scanned_pct > s_pct + 0.001:
                    scanned_count += 1
                    scanned_ranges.append((s_pct, scanned_pct))
                    report_lines.append(f"  • Faixa Varrida: {s_pct:.2f}% → {scanned_pct:.2f}% (Concluída - Sub-bloco Hex: 0x{j.get('current_offset_hex', '0')})")

                all_covered_intervals.append((s_pct, e_pct))
                
                if j['status'] == 'ACTIVE' and scanned_pct < e_pct:
                    w_name = active_workers.get(j['job_id'], "Worker Ativo")
                    active_ranges.append((scanned_pct, e_pct, w_name, j['job_id'], j.get('current_offset_hex', '0')))

            if scanned_count == 0:
                report_lines.append("  (Nenhum sub-bloco concluído ainda - workers iniciando varredura...)")
            report_lines.append("")

            report_lines.append("🟢 [FAIXAS EM MINERAÇÃO ATIVA NO MOMENTO]")
            if active_ranges:
                for act_s, act_e, w_name, j_id, cur_hex in active_ranges:
                    report_lines.append(f"  • Em Progresso: {act_s:.2f}% → {act_e:.2f}% | Worker: {w_name} | Sub-bloco Hex: 0x{cur_hex}")
            else:
                report_lines.append("  ⚠️ Nenhuma faixa em mineração no momento.")
            report_lines.append("")
        else:
            report_lines.append("⚠️ Nenhuma faixa ativa no momento.\n")

        # Calculate free/unexplored gaps between 0.0% and 100.0%
        all_covered_intervals.sort(key=lambda x: x[0])
        free_gaps = []
        curr = 0.0
        for s_pct, e_pct in all_covered_intervals:
            if s_pct > curr + 0.05:
                free_gaps.append((round(curr, 2), round(s_pct, 2)))
            curr = max(curr, e_pct)
        if curr < 99.95:
            free_gaps.append((round(curr, 2), 100.0))

        report_lines.append("🆓 [FAIXAS LIVRES E INTOCADAS (RECOMENDADAS PARA NOVOS WORKERS)]")
        if free_gaps:
            for g_start, g_end in free_gaps:
                report_lines.append(f"  ✓ Faixa Livre: {g_start:.2f}% → {g_end:.2f}% (LIVRE - Use esta faixa para novos Workers!)")
        else:
            report_lines.append("  🎉 100% do range do Puzzle está totalmente coberto!")

        report_lines.append("\n======================================================================")
        report_text = "\n".join(report_lines)
        
        filepath = os.path.join(os.path.dirname(__file__), "COVERAGE.TXT")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_text)
        except Exception as e:
            print(f"Error writing COVERAGE.TXT: {e}")
            
        return {
            "text": report_text,
            "scanned_ranges": scanned_ranges,
            "free_gaps": free_gaps
        }

@app.get("/api/coverage")
def get_coverage_endpoint():
    return generate_coverage_report()

@app.get("/api/stats")
def get_stats(username: str = Depends(authenticate_dashboard)):
    cov_data = generate_coverage_report()
    with db_lock:
        conn = get_db()
        cursor = conn.cursor()
        now = time.time()
        
        # Workers active within the last 60 seconds
        cursor.execute("SELECT * FROM workers WHERE (? - last_ping) < ? ORDER BY last_ping DESC", (now, WORKER_ALIVE_SECONDS))
        raw_workers = [dict(w) for w in cursor.fetchall()]
        
        workers = []
        for w in raw_workers:
            w_dict = dict(w)
            w_name = w_dict.get('name') or w_dict['worker_id']
            cursor.execute('''
                SELECT c.start_hex, c.range_bits, j.start_percent, j.end_percent 
                FROM chunks c 
                JOIN jobs j ON c.job_id = j.job_id 
                WHERE (c.assigned_worker = ? OR c.assigned_worker LIKE ?) 
                ORDER BY c.assigned_at DESC LIMIT 1
            ''', (w_dict['worker_id'], f"{w_name}%"))
            chunk_job = cursor.fetchone()

            if chunk_job:
                w_dict['current_start_hex'] = f"0x{chunk_job['start_hex']}"
                w_dict['current_range_bits'] = chunk_job['range_bits']
                w_dict['assigned_range'] = f"{chunk_job['start_percent']}% → {chunk_job['end_percent']}%"
            else:
                cursor.execute("SELECT start_percent, end_percent, start_hex FROM jobs WHERE status = 'ACTIVE' ORDER BY created_at DESC LIMIT 1")
                act_job = cursor.fetchone()
                if act_job:
                    w_dict['assigned_range'] = f"{act_job['start_percent']}% → {act_job['end_percent']}%"
                    w_dict['current_start_hex'] = f"0x{act_job['start_hex']}"
                else:
                    w_dict['assigned_range'] = "0% → 100%"
                    w_dict['current_start_hex'] = "0x80000000000000000000000000000000000"
                w_dict['current_range_bits'] = "139"

            cursor.execute("SELECT COUNT(*) as cnt FROM chunks WHERE (assigned_worker = ? OR assigned_worker LIKE ?) AND status IN ('COMPLETED', 'SOLVED')", (w_dict['worker_id'], f"{w_name}%"))
            w_dict['completed_chunks'] = cursor.fetchone()['cnt']
            workers.append(w_dict)
            
        total_hashrate = sum(w['hashrate_mhs'] for w in workers)
        
        cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
        jobs_raw = [dict(j) for j in cursor.fetchall()]
        
        pubkey_to_preset = {}
        for p_num, p_data in PUZZLE_PRESETS.items():
            pubkey_to_preset[p_data["pubkey"].lower()] = (p_num, p_data["bits"])

        jobs = []
        for j in jobs_raw:
            j_dict = dict(j)
            pub = j_dict.get('pubkey', '').lower()
            j_dict['btc_address'] = pubkey_to_address(pub)
            
            cursor.execute("SELECT COUNT(*) as cnt FROM chunks WHERE job_id = ?", (j_dict['job_id'],))
            j_dict['total_chunks_assigned'] = cursor.fetchone()['cnt']
            
            cursor.execute("SELECT COUNT(*) as cnt FROM chunks WHERE job_id = ? AND status IN ('COMPLETED', 'SOLVED')", (j_dict['job_id'],))
            j_dict['completed_chunks'] = cursor.fetchone()['cnt']
            
            if pub in pubkey_to_preset:
                p_num, p_bits = pubkey_to_preset[pub]
                j_dict['puzzle_name'] = f"Puzzle #{p_num} ({p_bits} bits)"
                j_dict['puzzle_number'] = p_num
            else:
                j_dict['puzzle_name'] = f"Custom ({j_dict.get('range_bits', 66) + 1} bits)"
                j_dict['puzzle_number'] = None

            if j_dict.get('private_key'):
                pk_hex = j_dict['private_key'].strip()
                j_dict['private_key_hex'] = pk_hex if pk_hex.startswith("0x") else f"0x{pk_hex}"
                j_dict['wif_compressed'] = hex_to_wif(pk_hex, compressed=True)
                j_dict['wif_uncompressed'] = hex_to_wif(pk_hex, compressed=False)
                if j_dict.get('solved_at'):
                    j_dict['solved_at_str'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(j_dict['solved_at']))
            jobs.append(j_dict)

        conn.close()

        # Calcula soma real de chaves e total de chunks concluidos
        total_completed_chunks = sum(j['completed_chunks'] for j in jobs)
        cursor = get_db().cursor()
        # Operações computacionais reais do Kangaroo = 1.15 * 2^(range_bits / 2) por chunk
        keys_tested = sum(int(1.15 * (2 ** (int(r[0]) / 2.0))) for r in rows)
        keys_tested_live = keys_tested
        if keys_tested_live == 0 and total_hashrate > 0 and jobs:
            act_j = next((j for j in jobs if j.get("status") == "ACTIVE"), jobs[0])
            c_time = act_j.get("created_at")
            if c_time:
                elapsed_sec = max(0, time.time() - c_time)
                keys_tested_live = int(total_hashrate * 1_000_000 * elapsed_sec)

        if keys_tested_live >= 10**24:
            keys_zetta_str = f"{keys_tested_live / (10**24):.2f} Yottakeys"
        elif keys_tested_live >= 10**21:
            keys_zetta_str = f"{keys_tested_live / (10**21):.2f} Zetakeys"
        elif keys_tested_live >= 10**18:
            keys_zetta_str = f"{keys_tested_live / (10**18):.2f} Exakeys"
        elif keys_tested_live >= 10**15:
            keys_zetta_str = f"{keys_tested_live / (10**15):.2f} Petakeys"
        elif keys_tested_live >= 10**12:
            keys_zetta_str = f"{keys_tested_live / (10**12):.2f} Terakeys"
        elif keys_tested_live >= 10**9:
            keys_zetta_str = f"{keys_tested_live / (10**9):.2f} Gigakeys"
        else:
            keys_zetta_str = f"{keys_tested_live} Keys"

        expected_ops = 1.15 * (2 ** 69.5)
        prob_pct = (keys_tested_live / expected_ops) * 100.0 if expected_ops > 0 else 0.0
        if prob_pct > 100.0:
            prob_pct_str = "100.0% (Faixa Concluída)"
        elif prob_pct < 0.0001 and prob_pct > 0:
            prob_pct_str = f"{prob_pct:.6f}%"
        else:
            prob_pct_str = f"{prob_pct:.2f}%"

        # Cálculo de métricas avançadas do Cluster
        total_gpus = 0
        for w in workers:
            gpu_str = w.get("gpu_info", "")
            g_cnt = max(1, len([g for g in gpu_str.split(",") if g.strip()])) if gpu_str else 1
            total_gpus += g_cnt

        # Kangaroos Ativos reais alocados (761,856 kangaroos por GPU de 24GB do RCKangaroo)
        total_kangaroos_cnt = total_gpus * 761856
        if total_gpus > 0:
            if total_kangaroos_cnt >= 1_000_000:
                active_kangaroos_str = f"{total_kangaroos_cnt / 1_000_000:.1f}M"
            else:
                active_kangaroos_str = f"{total_kangaroos_cnt / 1_000:.0f}K"
        else:
            active_kangaroos_str = "0.0M"

        # Cálculo dinâmico do K-Factor e DP Overhead por Instância Worker (2 GPUs)
        kangs_per_worker = 2 * 761856  # 1,523,712 kangaroos por container Vast-2x
        chunk_bits = 80
        dp_bits = 18
        ops_ideal = 1.15 * (2.0 ** (chunk_bits / 2.0))
        dp_val = float(1 << dp_bits)
        path_single_kang = ops_ideal / kangs_per_worker
        dps_per_kang = max(0.001, path_single_kang / dp_val)

        k_factor = 1.15 + (0.07 + 0.76 / math.sqrt(dps_per_kang)) / (1.0 + 0.30 * dps_per_kang)
        overhead_pct = int(0.5 + 100.0 * (k_factor / 1.15 - 1.0))

        dp_str = "12% ~ 15%"
        k_subtext_str = f"K ≈ {k_factor:.2f} ({overhead_pct}% Média por Instância)"

        cursor.execute("SELECT status, COUNT(*) FROM chunks GROUP BY status")
        chunk_counts = {r[0]: r[1] for r in cursor.fetchall()}
        assigned_chunks = chunk_counts.get("ASSIGNED", 0)
        pending_chunks = chunk_counts.get("PENDING", 0)
        total_created_chunks = sum(chunk_counts.values())

        # Tempo de processamento do Job ativo mais recente
        active_job_uptime = "0h 0m"
        if jobs:
            act_j = next((j for j in jobs if j.get("status") == "ACTIVE"), jobs[0])
            c_time = act_j.get("created_at")
            if c_time:
                up_sec = int(time.time() - c_time)
                h = up_sec // 3600
                m = (up_sec % 3600) // 60
                active_job_uptime = f"{h}h {m}m"

        active_dps = sum(w.get('dps_count', 0) for w in workers if w.get('dps_count'))
        total_dps_cnt = sum(len(v) for v in GLOBAL_DP_CACHE.values())
        if total_dps_cnt == 0:
            total_dps_cnt = active_dps

        if total_dps_cnt >= 1_000_000_000_000:
            total_dps_str = f"{total_dps_cnt / 1_000_000_000_000:.2f}T DPs"
        elif total_dps_cnt >= 1_000_000_000:
            total_dps_str = f"{total_dps_cnt / 1_000_000_000:.2f}B DPs"
        elif total_dps_cnt >= 1_000_000:
            total_dps_str = f"{total_dps_cnt / 1_000_000:.2f}M DPs"
        elif total_dps_cnt >= 1_000:
            total_dps_str = f"{total_dps_cnt / 1_000:.1f}K DPs"
        else:
            total_dps_str = f"{total_dps_cnt} DPs"

        dps_per_sec = int(total_hashrate * 1_000_000 / (2**dp_bits))
        if dps_per_sec >= 1_000_000:
            dps_per_sec_str = f"{dps_per_sec / 1_000_000:.2f}M DPs/s"
        elif dps_per_sec >= 1_000:
            dps_per_sec_str = f"{dps_per_sec / 1_000:.1f}K DPs/s"
        else:
            dps_per_sec_str = f"{dps_per_sec} DPs/s"

        return {
            "active_workers_count": len(workers),
            "total_gpus_count": total_gpus,
            "total_pool_hashrate_mhs": round(total_hashrate, 2),
            "total_pool_hashrate_ghs": round(total_hashrate / 1000.0, 3),
            "dps_per_sec_str": dps_per_sec_str,
            "prob_pct_str": prob_pct_str,
            "total_completed_chunks": total_completed_chunks,
            "assigned_chunks_count": assigned_chunks,
            "pending_chunks_count": pending_chunks,
            "total_created_chunks": total_created_chunks,
            "active_kangaroos_m": active_kangaroos_str,
            "dp_overhead_str": total_dps_str,
            "total_dps_str": total_dps_str,
            "k_subtext_str": k_subtext_str,
            "active_job_uptime": active_job_uptime,
            "keys_tested": keys_tested,
            "keys_zetta_str": keys_zetta_str,
            "workers": workers,
            "jobs": jobs,
            "coverage": cov_data,
            "logs": list(server_logs)
        }

# Open Worker Endpoints (Auto-Creates Job if No Active Job Exists)
@app.post("/api/worker/ensure_job")
def ensure_job(req: CreateJobRequest, _: None = Depends(verify_worker_token)):
    with db_lock:
        conn = get_db()
        cursor = conn.cursor()
        
        target_pubkey = None
        if req.puzzle_number in PUZZLE_PRESETS:
            target_pubkey = PUZZLE_PRESETS[req.puzzle_number]["pubkey"].lower()
        elif req.pubkey:
            target_pubkey = req.pubkey.lower()

        if target_pubkey:
            # Check if active job for exact same pubkey and matching range % already exists
            cursor.execute("""
                SELECT job_id FROM jobs 
                WHERE status = 'ACTIVE' 
                AND LOWER(pubkey) = ? 
                AND ABS(start_percent - ?) < 0.01 
                AND ABS(end_percent - ?) < 0.01 
                LIMIT 1
            """, (target_pubkey, req.start_percent, req.end_percent))
            range_job = cursor.fetchone()

            if range_job:
                job_id = range_job["job_id"]
                # Se o job existente já atingiu o fim da fatia, reseta o offset para permitir novo teste
                cursor.execute("SELECT start_hex, base_start_hex, range_bits, end_percent, current_offset_hex FROM jobs WHERE job_id = ?", (job_id,))
                row = cursor.fetchone()
                if row:
                    try:
                        cur_off = int(row['current_offset_hex'], 16)
                        b_start = int(row['base_start_hex'], 16)
                        tot_r = 1 << int(row['range_bits'])
                        e_off = b_start + int((float(row['end_percent']) / 100.0) * tot_r)
                        if cur_off >= e_off:
                            cursor.execute("UPDATE jobs SET current_offset_hex = start_hex WHERE job_id = ?", (job_id,))
                            cursor.execute("DELETE FROM chunks WHERE job_id = ?", (job_id,))
                    except Exception:
                        pass

                if req.worker_id:
                    cursor.execute("UPDATE workers SET current_job_id = ? WHERE worker_id = ?", (job_id, req.worker_id))
                conn.commit()
                conn.close()
                return {"status": "EXISTS", "job_id": job_id}

            # Deactivate jobs for a completely different puzzle
            cursor.execute("UPDATE jobs SET status = 'CANCELLED' WHERE status = 'ACTIVE' AND LOWER(pubkey) != ?", (target_pubkey,))
            conn.commit()
            conn.close()
    
    # Auto-create job for this specific range
    job_id = internal_create_job(req)
    with db_lock:
        conn = get_db()
        cursor = conn.cursor()
        if req.worker_id:
            cursor.execute("UPDATE workers SET current_job_id = ? WHERE worker_id = ?", (job_id, req.worker_id))
            conn.commit()
        conn.close()
    return {"status": "AUTO_CREATED", "job_id": job_id}


@app.post("/api/worker/register")
def register_worker(req: WorkerRegisterRequest, _: None = Depends(verify_worker_token)):
    with db_lock:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO workers (worker_id, name, os_info, gpu_info, hashrate_mhs, last_ping, completed_chunks)
            VALUES (?, ?, ?, ?, 0.0, ?, 0)
            ON CONFLICT(worker_id) DO UPDATE SET
                name=excluded.name,
                os_info=excluded.os_info,
                gpu_info=excluded.gpu_info,
                last_ping=excluded.last_ping
        ''', (req.worker_id, req.name, req.os_info, req.gpu_info, time.time()))
        conn.commit()
        conn.close()
        log_event(f"👤 Worker registrado: {req.name} ({req.worker_id}) | GPU: {req.gpu_info or 'N/A'}")
    return {"status": "REGISTERED", "worker_id": req.worker_id}

@app.post("/api/worker/heartbeat")
def heartbeat(req: WorkerHeartbeatRequest, _: None = Depends(verify_worker_token)):
    with db_lock:
        conn = get_db()
        cursor = conn.cursor()
        if req.dps_count and req.dps_count > 0:
            cursor.execute('''
                UPDATE workers SET hashrate_mhs = ?, last_ping = ?, dps_count = ? WHERE worker_id = ?
            ''', (req.hashrate_mhs, time.time(), req.dps_count, req.worker_id))
        else:
            cursor.execute('''
                UPDATE workers SET hashrate_mhs = ?, last_ping = ? WHERE worker_id = ?
            ''', (req.hashrate_mhs, time.time(), req.worker_id))
        conn.commit()
        conn.close()
        hr_str = f"{req.hashrate_mhs / 1000.0:.2f} GH/s" if req.hashrate_mhs >= 1000 else f"{req.hashrate_mhs:.1f} MH/s"
        add_vps_log(f"💾 Progresso Salvo na VPS | Guerreiro: {req.worker_id} | Hashrate: {hr_str} ✅")
    return {"status": "OK"}

@app.post("/api/worker/chunk_heartbeat")
def chunk_heartbeat(req: ChunkHeartbeatRequest, _: None = Depends(verify_worker_token)):
    with db_lock:
        conn = get_db()
        cursor = conn.cursor()
        now = time.time()

        cursor.execute("""
            UPDATE chunks 
            SET last_heartbeat = ?
            WHERE chunk_id = ? AND assigned_worker = ? AND status = 'ASSIGNED'
        """, (now, req.chunk_id, req.worker_id))

        cursor.execute("""
            UPDATE workers 
            SET last_ping = ?, hashrate_mhs = ?
            WHERE worker_id = ?
        """, (now, req.hashrate_mhs, req.worker_id))

        conn.commit()
        conn.close()
        hr_str = f"{req.hashrate_mhs / 1000.0:.2f} GH/s" if req.hashrate_mhs >= 1000 else f"{req.hashrate_mhs:.1f} MH/s"
        add_vps_log(f"💾 Heartbeat & Progresso Salvos na VPS | Sub-bloco: {req.chunk_id} | Guerreiro: {req.worker_id} ({hr_str}) ✅")
    return {"status": "OK"}

@app.post("/api/worker/get_work")
def get_work(req: WorkRequest, _: None = Depends(verify_worker_token)):
    with db_lock:
        conn = get_db()
        cursor = conn.cursor()
        now = time.time()

        if req.hashrate_mhs is not None and req.hashrate_mhs > 0:
            cursor.execute("UPDATE workers SET last_ping = ?, hashrate_mhs = ? WHERE worker_id = ?", (now, req.hashrate_mhs, req.worker_id))
        else:
            cursor.execute("UPDATE workers SET last_ping = ? WHERE worker_id = ?", (now, req.worker_id))

        # === 1. Recupera chunks abandonados (timeout de 20 minutos) ===
        TIMEOUT_SECONDS = 20 * 60  # 20 minutos
        cursor.execute("""
            UPDATE chunks 
            SET status = 'PENDING', assigned_worker = NULL
            WHERE status = 'ASSIGNED' 
              AND (last_heartbeat IS NULL OR (? - COALESCE(last_heartbeat, assigned_at)) > ?)
        """, (now, TIMEOUT_SECONDS))

        # === 2. Identifica o Job Alvo do Worker com base no Nome (A1..A10) ===
        worker_identifier = req.name or req.worker_id or ""
        job = None

        if worker_identifier:
            pct_map = {
                'A1': 0.0, 'A2': 10.0, 'A3': 20.0, 'A4': 30.0, 'A5': 40.0,
                'A6': 50.0, 'A7': 60.0, 'A8': 70.0, 'A9': 80.0, 'A10': 90.0
            }
            for k, pct in pct_map.items():
                if k in worker_identifier:
                    cursor.execute("SELECT * FROM jobs WHERE ABS(start_percent - ?) < 0.1 AND status = 'ACTIVE' LIMIT 1", (pct,))
                    job = cursor.fetchone()
                    if job:
                        break

        # Fallback: job ativo registrado do worker
        if not job:
            cursor.execute("""
                SELECT j.* FROM jobs j 
                JOIN workers w ON w.current_job_id = j.job_id 
                WHERE (w.worker_id = ? OR w.name = ?) AND j.status = 'ACTIVE' 
                LIMIT 1
            """, (req.worker_id, worker_identifier))
            job = cursor.fetchone()

        # Fallback: último job ativo criado
        if not job:
            cursor.execute("SELECT * FROM jobs WHERE status = 'ACTIVE' ORDER BY created_at DESC LIMIT 1")
            job = cursor.fetchone()

        if not job:
            conn.commit()
            conn.close()
            return {"status": "NO_WORK", "message": "Nenhum job ativo disponível"}

        job = dict(job)

        # === 3. Tenta pegar um chunk PENDING (abandonado) DO MESMO JOB primeiro ===
        cursor.execute("""
            SELECT c.*, j.pubkey, j.dp_bits, j.range_bits as job_range_bits
            FROM chunks c
            JOIN jobs j ON c.job_id = j.job_id
            WHERE c.status = 'PENDING' AND c.job_id = ? AND j.status = 'ACTIVE'
            ORDER BY c.assigned_at ASC
            LIMIT 1
        """, (job['job_id'],))
        pending = cursor.fetchone()

        if pending:
            chunk = dict(pending)
            chunk_bits = chunk['range_bits']
            dp_bits_dynamic = 20 if chunk_bits >= 90 else (16 if chunk_bits >= 70 else min(19, max(14, (chunk_bits // 2) - 26)))
            cursor.execute("""
                UPDATE chunks 
                SET status = 'ASSIGNED',
                    assigned_worker = ?,
                    assigned_at = ?,
                    last_heartbeat = ?
                WHERE chunk_id = ?
            """, (req.worker_id, now, now, chunk['chunk_id']))

            cursor.execute("UPDATE workers SET current_job_id = ? WHERE worker_id = ?", 
                           (chunk['job_id'], req.worker_id))

            conn.commit()
            conn.close()
            log_event(f"♻️ Reatribuindo chunk abandonado {chunk['chunk_id']} para {worker_identifier}")

            return {
                "status": "WORK_ASSIGNED",
                "chunk_id": chunk['chunk_id'],
                "job_id": chunk['job_id'],
                "pubkey": chunk['pubkey'],
                "start_hex": chunk['start_hex'],
                "range_bits": chunk['job_range_bits'],
                "chunk_bits": chunk_bits,
                "dp_bits": dp_bits_dynamic,
                "max_ops": "1.0"
            }

        # === 4. Avança o offset do Job específico do Worker ===
        cursor.execute("UPDATE workers SET current_job_id = ? WHERE worker_id = ?", 
                       (job['job_id'], req.worker_id))

        current_offset_int = int(job['current_offset_hex'], 16)
        chunk_bits = min(job.get('chunk_bits', 78), job['range_bits'])
        if chunk_bits < 48 and job['range_bits'] >= 48:
            chunk_bits = 78

        chunk_size = 1 << chunk_bits

        base_start_int = int(job['base_start_hex'], 16)
        total_range_int = 1 << job['range_bits']
        end_pct = float(job['end_percent'])
        end_offset_int = base_start_int + int((end_pct / 100.0) * total_range_int)

        if current_offset_int >= end_offset_int:
            conn.commit()
            conn.close()
            return {"status": "NO_WORK", "message": f"Fatia concluída até {end_pct:.2f}%"}

        chunk_id = f"{job['job_id']}_chunk_{hex(current_offset_int)[2:]}"
        chunk_start_hex = hex(current_offset_int)[2:]
        next_offset_hex = hex(current_offset_int + chunk_size)[2:]

        cursor.execute(
            "UPDATE jobs SET current_offset_hex = ? WHERE job_id = ?",
            (next_offset_hex, job['job_id'])
        )

        cursor.execute('''
            INSERT INTO chunks (
                chunk_id, job_id, start_hex, range_bits,
                assigned_worker, assigned_at, status, last_heartbeat
            ) VALUES (?, ?, ?, ?, ?, ?, 'ASSIGNED', ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                assigned_worker=excluded.assigned_worker,
                assigned_at=excluded.assigned_at,
                status='ASSIGNED',
                last_heartbeat=excluded.last_heartbeat
        ''', (
            chunk_id, job['job_id'], chunk_start_hex, chunk_bits,
            req.worker_id, now, now
        ))

        dp_bits_dynamic = 20 if chunk_bits >= 90 else (16 if chunk_bits >= 70 else min(19, max(14, (chunk_bits // 2) - 26)))

        conn.commit()
        conn.close()

        log_event(f"📦 Novo chunk {chunk_id} (bits={chunk_bits}) atribuído a {worker_identifier}")

        return {
            "status": "WORK_ASSIGNED",
            "chunk_id": chunk_id,
            "job_id": job['job_id'],
            "pubkey": job['pubkey'],
            "start_hex": job['base_start_hex'],
            "range_bits": job['range_bits'],
            "chunk_bits": chunk_bits,
            "dp_bits": dp_bits_dynamic,
            "max_ops": "1000.0"
        }

        # ─── Calcula dp_bits e max_ops dinâmicamente com base no chunk real ────────
        # Meta: K ≈ 1.15 (DP overhead ultrabaixo ~2–3%)
        # Medições reais:  dp=32→K=12.52 | dp=26→K=2.52 | dp=21→K=1.165 | dp=20→K=1.15
        # dp=19 (DP 19) → overhead ≈ 2–3% → K ≈ 1.15 ✓  RAM ≈ 3.6 GB/GPU
        # dp_ideal ≈ (chunk_bits // 2) - 26  →  90-bit: 19, 80-bit: 14, 66-bit: 14(min)
        dp_bits_dynamic = min(19, max(14, (chunk_bits // 2) - 26))

        # max_ops = (chunk_bits / range_bits) * 1.5  — margem segura com K≈1.15
        max_ops_dynamic = round((chunk_bits / job['range_bits']) * 1.5, 2)

        conn.commit()
        conn.close()
        log_event(f"🚀 Work atribuído para {req.name or req.worker_id}: Chunk {chunk_id} (Bits: {chunk_bits}, DP: {dp_bits_dynamic})")

        return {
            "status": "WORK_ASSIGNED",
            "chunk_id": chunk_id,
            "job_id": job['job_id'],
            "pubkey": job['pubkey'],
            "start_hex": chunk_start_hex,
            "range_bits": job['range_bits'],
            "chunk_bits": chunk_bits,
            "dp_bits": dp_bits_dynamic,
            "max_ops": str(max_ops_dynamic)
        }

@app.post("/api/worker/complete_chunk")
def complete_chunk(req: CompleteChunkRequest, _: None = Depends(verify_worker_token)):
    """Worker chama este endpoint quando termina um chunk com sucesso (sem encontrar a chave).
    Só então o chunk é marcado como COMPLETED — elimina falsos positivos.
    """
    with db_lock:
        conn = get_db()
        cursor = conn.cursor()
        now = time.time()

        cursor.execute("""
            UPDATE chunks SET status = 'COMPLETED', completed_at = ?
            WHERE chunk_id = ? AND assigned_worker = ? AND status = 'ASSIGNED'
        """, (now, req.chunk_id, req.worker_id))

        updated = cursor.rowcount
        if updated > 0:
            cursor.execute(
                "UPDATE workers SET last_ping = ?, completed_chunks = completed_chunks + 1 WHERE worker_id = ?",
                (now, req.worker_id)
            )
            log_event(f"✅ Chunk {req.chunk_id} marcado como COMPLETED pelo worker {req.worker_id}")
        else:
            log_event(f"⚠️ complete_chunk ignorado: chunk {req.chunk_id} não estava ASSIGNED para {req.worker_id}")

        conn.commit()
        conn.close()

    return {"status": "OK", "completed": updated > 0}


@app.post("/api/worker/submit_solution")
def submit_solution(req: SubmitSolutionRequest, _: None = Depends(verify_worker_token)):
    # Mathematically verify candidate private key against target pubkey
    if not verify_private_key(req.private_key, req.pubkey):
        print(f"❌ REJECTED False positive solution from worker {req.worker_id}: {req.private_key}")
        raise HTTPException(status_code=400, detail="Invalid solution key for target pubkey")

    with db_lock:
        conn = get_db()
        cursor = conn.cursor()

        now = time.time()
        
        cursor.execute('''
            UPDATE jobs SET status = 'SOLVED', solved_at = ?, solved_by = ?, private_key = ?
            WHERE pubkey = ? OR job_id = (SELECT job_id FROM chunks WHERE chunk_id = ?)
        ''', (now, req.worker_id, req.private_key, req.pubkey.lower(), req.chunk_id))
        
        cursor.execute("UPDATE chunks SET status = 'SOLVED' WHERE chunk_id = ?", (req.chunk_id,))
        results_file = os.path.join(os.path.dirname(__file__), "POOL_RESULTS.TXT")
        wif_key = hex_to_wif(req.private_key, compressed=True)
        with open(results_file, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] PUBKEY: {req.pubkey} | PRIVATE KEY HEX: {req.private_key} | WIF: {wif_key} | SOLVED BY: {req.worker_id}\n")

        conn.commit()
        conn.close()

    print(f"🎉 SOLVED! Worker {req.worker_id} found private key: {req.private_key} (WIF: {wif_key})")
    return {"status": "ACCEPTED", "message": "Solution recorded!"}

# HTML Dashboard Route
@app.get("/", response_class=HTMLResponse)
def get_dashboard(username: str = Depends(authenticate_dashboard)):
    html_content = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RCKangaroo Pool Coordinator | Saiyan Power Level</title>
        <link rel="icon" type="image/png" href="/logo.png">
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            :root {
                --bg-body: #060913;
                --bg-card: #0c1222;
                --bg-card-hover: #131c35;
                --border-color: rgba(251, 191, 36, 0.25);
                --border-glow-gold: rgba(251, 191, 36, 0.5);
                --border-glow-cyan: rgba(56, 189, 248, 0.5);
                --text-primary: #f8fafc;
                --text-secondary: #94a3b8;
                --text-muted: #64748b;
                --accent-gold: #fbbf24;
                --accent-cyan: #38bdf8;
                --accent-green: #34d399;
                --accent-purple: #c084fc;
                --accent-red: #f87171;
            }

            body {
                background-color: var(--bg-body);
                background-image: 
                    radial-gradient(circle at 15% 15%, rgba(251, 191, 36, 0.04) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(56, 189, 248, 0.04) 0%, transparent 40%);
                color: var(--text-primary);
                font-family: 'Outfit', -apple-system, sans-serif;
                min-height: 100vh;
            }

            .font-saiyan {
                font-family: 'Orbitron', sans-serif;
                letter-spacing: 0.04em;
            }

            .glow-gold {
                text-shadow: none;
            }
            .glow-cyan {
                text-shadow: none;
            }
            .glow-green {
                text-shadow: none;
            }

            .glass-card {
                background: #0d1322;
                border: 1px solid rgba(251, 191, 36, 0.2);
                border-radius: 14px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
                transition: all 0.2s ease;
            }
            .glass-card:hover {
                border-color: rgba(251, 191, 36, 0.4);
                box-shadow: 0 6px 24px rgba(0, 0, 0, 0.6);
            }

            .stat-card-saiyan {
                position: relative;
                overflow: hidden;
                padding: 1.3rem;
            }
            .stat-card-saiyan::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 4px;
                height: 100%;
                background: var(--accent-color, var(--accent-gold));
            }

            .ki-gauge-bg {
                height: 6px;
                background: rgba(30, 41, 59, 0.8);
                border-radius: 10px;
                overflow: hidden;
                margin-top: 10px;
            }
            .ki-gauge-fill {
                height: 100%;
                background: linear-gradient(90deg, #f59e0b, #fbbf24, #38bdf8);
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(251, 191, 36, 0.8);
                transition: width 0.5s ease;
            }

            .stat-header {
                font-size: 2.1rem;
                font-weight: 900;
                font-family: 'Orbitron', sans-serif;
                color: var(--text-primary);
                letter-spacing: -0.01em;
            }
            .stat-title {
                color: var(--text-secondary);
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-bottom: 0.4rem;
            }

            .section-title {
                font-size: 1.25rem;
                font-weight: 800;
                font-family: 'Orbitron', sans-serif;
                color: var(--text-primary);
                letter-spacing: 0.02em;
            }

            .table-custom {
                background: transparent !important;
                color: var(--text-primary) !important;
                margin-bottom: 0;
            }
            .table-custom th {
                background-color: rgba(10, 15, 30, 0.9) !important;
                color: var(--accent-gold) !important;
                font-family: 'Orbitron', sans-serif !important;
                font-size: 0.78rem !important;
                font-weight: 700 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.08em !important;
                padding: 1rem 1.2rem !important;
                border-bottom: 1px solid var(--border-color) !important;
            }
            .table-custom td {
                padding: 1rem 1.2rem !important;
                border-bottom: 1px solid rgba(30, 41, 59, 0.6) !important;
                vertical-align: middle !important;
                background: transparent !important;
                color: var(--text-primary) !important;
            }
            .table-custom tbody tr:hover td {
                background-color: rgba(30, 41, 59, 0.5) !important;
            }

            .font-mono {
                font-family: 'JetBrains Mono', monospace;
            }

            .key-box {
                background-color: #040711;
                border: 1px solid rgba(251, 191, 36, 0.3);
                border-radius: 10px;
                padding: 10px 14px;
                font-family: 'JetBrains Mono', monospace;
                word-break: break-all;
                color: var(--accent-cyan);
                box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.8);
            }

            .solved-banner {
                background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(16, 185, 129, 0.25) 50%, rgba(8, 12, 24, 0.98) 100%);
                border: 2px solid var(--accent-gold);
                box-shadow: 0 0 45px rgba(251, 191, 36, 0.35);
                border-radius: 20px;
            }

            .badge-glow-gold {
                background: rgba(251, 191, 36, 0.15);
                color: var(--accent-gold);
                border: 1px solid rgba(251, 191, 36, 0.4);
            }
            .badge-glow-cyan {
                background: rgba(56, 189, 248, 0.15);
                color: var(--accent-cyan);
                border: 1px solid rgba(56, 189, 248, 0.4);
            }
            .badge-glow-green {
                background: rgba(52, 211, 153, 0.15);
                color: var(--accent-green);
                border: 1px solid rgba(52, 211, 153, 0.4);
            }
            .badge-glow-purple {
                background: rgba(192, 132, 252, 0.15);
                color: var(--accent-purple);
                border: 1px solid rgba(192, 132, 252, 0.4);
            }

            .kangaroo-logo-container {
                width: 62px;
                height: 62px;
                background: rgba(15, 23, 42, 0.8);
                border: 2px solid var(--accent-gold);
                border-radius: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 0 30px rgba(251, 191, 36, 0.3);
                flex-shrink: 0;
                overflow: hidden;
            }
            .kangaroo-img-anim {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }

            .pulse-dot {
                width: 8px;
                height: 8px;
                background-color: var(--accent-green);
                border-radius: 50%;
                display: inline-block;
                box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7);
                animation: pulse-green 2s infinite;
            }
            @keyframes pulse-green {
                0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
                70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(52, 211, 153, 0); }
                100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
            }

            @media (max-width: 768px) {
                body { padding: 0.75rem !important; }
                .stat-header { font-size: 1.6rem; }
            }
        </style>
    </head>
    <body class="p-3 p-md-4">
        <div class="container-fluid max-w-7xl mx-auto">

            <!-- Top Header (Saiyan Style) -->
            <div class="glass-card p-3 p-md-4 mb-4 d-flex flex-column flex-md-row justify-content-between align-items-md-center gap-3">
                <div class="d-flex align-items-center gap-3">
                    <div class="kangaroo-logo-container">
                        <img src="/logo.png" alt="RCKangaroo Logo" class="kangaroo-img-anim">
                    </div>
                    <div>
                        <div class="d-flex align-items-center gap-2">
                            <h1 class="h4 mb-0 fw-bold font-saiyan text-warning glow-gold">RCKANGAROO POOL COORDINATOR</h1>
                            <span class="badge badge-glow-green px-2.5 py-1 rounded-pill fs-7 font-semibold">
                                <span class="pulse-dot me-1"></span> ONLINE (WAL)
                            </span>
                        </div>
                        <span class="text-secondary fs-7">⚡ Coordenador Distribuído de Alta Performance — Range Completo (Sem Fatiamento Espacial)</span>
                    </div>
                </div>
                <div class="d-flex flex-wrap gap-2">
                    <button class="btn btn-outline-warning btn-sm font-saiyan rounded-3 px-3" onclick="openCoverageModal()">
                        📄 COVERAGE.TXT
                    </button>
                    <button class="btn btn-outline-danger btn-sm rounded-3 px-3 fw-semibold" onclick="clearOldJobs()">
                        🗑️ Limpar Jobs
                    </button>
                    <button class="btn btn-warning btn-sm rounded-3 px-3 font-saiyan text-dark fw-bold bg-warning border-0" onclick="loadStats()">
                        🔄 ATUALIZAR
                    </button>
                </div>
            </div>

            <!-- Solved Alert Banner -->
            <div id="solved-solutions-container" class="mb-4"></div>

            <!-- SEÇÃO 1: RECURSOS EM EXECUÇÃO AGORA (EM TEMPO REAL) -->
            <div class="d-flex align-items-center gap-2 mb-3">
                <span class="fs-5">⚡</span>
                <h2 class="h6 mb-0 text-warning font-saiyan fw-bold glow-gold">STATUS EM TEMPO REAL (EM EXECUÇÃO AGORA)</h2>
            </div>
            <div class="row g-3 mb-4">
                <!-- 1. PODER DE LUTA (HASHRATE AGREGADO) -->
                <div class="col-12 col-md-3">
                    <div class="glass-card stat-card-saiyan" style="--accent-color: var(--accent-gold);">
                        <div class="d-flex justify-content-between align-items-center">
                            <div class="stat-title">💥 PODER DE LUTA AGREGADO</div>
                            <span class="badge badge-glow-gold fs-8 font-saiyan">LIVE</span>
                        </div>
                        <div class="stat-header text-warning glow-gold mt-1" id="pool-hashrate">0.00 GKeys/s</div>
                        <div class="fs-7 text-secondary mt-1">Throughput total combinado das GPUs</div>
                        <div class="ki-gauge-bg">
                            <div class="ki-gauge-fill" id="ki-gauge-hashrate" style="width: 100%;"></div>
                        </div>
                    </div>
                </div>

                <!-- 2. GUERREIROS Z (GPUs ATIVAS AGORA) -->
                <div class="col-12 col-md-3">
                    <div class="glass-card stat-card-saiyan" style="--accent-color: var(--accent-green);">
                        <div class="d-flex justify-content-between align-items-center">
                            <div class="stat-title">🔥 NODES ATIVOS AGORA</div>
                            <span class="badge badge-glow-green fs-8 font-saiyan">WORKERS</span>
                        </div>
                        <div class="stat-header text-success glow-green mt-1" id="active-kangaroos">0.0M</div>
                        <div class="fs-7 text-secondary mt-1" id="active-gpus-subtext">0 GPUs operando</div>
                        <div class="ki-gauge-bg">
                            <div class="ki-gauge-fill" style="width: 100%; background: linear-gradient(90deg, #059669, #34d399);"></div>
                        </div>
                    </div>
                </div>

                <!-- 3. TAXA DE DPS EM TEMPO REAL -->
                <div class="col-12 col-md-3">
                    <div class="glass-card stat-card-saiyan" style="--accent-color: var(--accent-cyan);">
                        <div class="d-flex justify-content-between align-items-center">
                            <div class="stat-title">⚡ TAXA DE DPS (STREAM)</div>
                            <span class="badge badge-glow-cyan fs-8 font-saiyan">LIVE DPS</span>
                        </div>
                        <div class="stat-header text-info mt-1" id="completed-chunks">0 DPs/s</div>
                        <div class="fs-7 text-secondary mt-1" id="chunks-total-subtext">Eficiência ideal K ≈ 1.15 (0% Overhead)</div>
                        <div class="ki-gauge-bg">
                            <div class="ki-gauge-fill" style="width: 95%; background: linear-gradient(90deg, #0284c7, #38bdf8);"></div>
                        </div>
                    </div>
                </div>

                <!-- 4. TEMPO ATIVO DA SESSÃO ATUAL -->
                <div class="col-12 col-md-3">
                    <div class="glass-card stat-card-saiyan" style="--accent-color: var(--accent-gold);">
                        <div class="d-flex justify-content-between align-items-center">
                            <div class="stat-title">⏱️ SESSÃO ATIVA</div>
                            <span class="badge badge-glow-gold fs-8 font-saiyan">TEMPO</span>
                        </div>
                        <div class="stat-header text-warning mt-1" id="job-uptime">0h 0m</div>
                        <div class="fs-7 text-secondary mt-1">Duração da sessão ativa na Pool</div>
                        <div class="ki-gauge-bg">
                            <div class="ki-gauge-fill" style="width: 75%; background: linear-gradient(90deg, #d97706, #fbbf24);"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- SEÇÃO 2: HISTÓRICO ACUMULADO & BANCO DE DADOS DA POOL (EM BAIXO) -->
            <div class="d-flex align-items-center gap-2 mb-3">
                <span class="fs-5">📊</span>
                <h2 class="h6 mb-0 text-info font-saiyan fw-bold glow-cyan">HISTÓRICO ACUMULADO & BANCO DE DADOS DA POOL (VPS)</h2>
            </div>
            <div class="row g-3 mb-4">
                <!-- 1. ENERGIAS TESTADAS (HISTÓRICO DE CHAVES) -->
                <div class="col-12 col-md-6">
                    <div class="glass-card stat-card-saiyan" style="--accent-color: var(--accent-cyan);">
                        <div class="d-flex justify-content-between align-items-center">
                            <div class="stat-title">🔮 TOTAL HISTÓRICO DE CHAVES TESTADAS</div>
                            <span class="badge badge-glow-cyan fs-8 font-saiyan">PUZZLE #140</span>
                        </div>
                        <div class="stat-header text-info glow-cyan mt-1" id="keys-tested">0.00 Exakeys</div>
                        <div class="fs-7 text-secondary mt-1" id="keys-tested-subtext">Soma acumulada de chaves verificadas</div>
                        <div class="ki-gauge-bg">
                            <div class="ki-gauge-fill" style="width: 85%; background: linear-gradient(90deg, #0284c7, #38bdf8);"></div>
                        </div>
                    </div>
                </div>

                <!-- 2. BANCO GLOBAL DE DPS SALVOS NA VPS -->
                <div class="col-12 col-md-6">
                    <div class="glass-card stat-card-saiyan" style="--accent-color: var(--accent-purple);">
                        <div class="d-flex justify-content-between align-items-center">
                            <div class="stat-title">🦘 BANCO GLOBAL DE DPS SALVOS NA VPS</div>
                            <span class="badge badge-glow-purple fs-8 font-saiyan">PUZZLE #140</span>
                        </div>
                        <div class="stat-header text-purple mt-1" style="color: #c084fc;" id="dp-overhead">0 DPs</div>
                        <div class="fs-7 text-secondary mt-1" id="dp-overhead-subtext">Armadilhas registradas permanentemente no SQLite WAL</div>
                        <div class="ki-gauge-bg">
                            <div class="ki-gauge-fill" style="width: 100%; background: linear-gradient(90deg, #9333ea, #c084fc);"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Live VPS Logs Section -->
            <div class="glass-card p-3 p-md-4 mb-4">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <div class="d-flex align-items-center gap-2">
                        <span class="fs-5">🖥️</span>
                        <h2 class="section-title mb-0 fs-6">Logs em Tempo Real da VPS</h2>
                        <span class="badge badge-glow-green px-2 py-0.5 fs-8">Live Feed (VPS)</span>
                    </div>
                    <button class="btn btn-sm btn-outline-secondary py-0 px-2 fs-7" onclick="document.getElementById('vps-logs-box').innerText='[Log limpo pelo usuário.]'">🧹 Limpar Console</button>
                </div>
                <div id="vps-logs-box" class="font-mono p-3 rounded-3 fs-7" style="background-color: #030612; color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); height: 85px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; line-height: 1.5;">[Aguardando atividade dos guerreiros na VPS...]</div>
            </div>

            <!-- Active Target Puzzle Card -->
            <div id="active-target-puzzle-container" class="mb-4"></div>

            <!-- Active Workers & Live Ranges Table -->
            <div class="glass-card p-3 p-md-4 mb-4">
                <div class="d-flex flex-column flex-md-row justify-content-between align-items-md-center gap-2 mb-3">
                    <div>
                        <h2 class="section-title mb-0">🔥 Guerreiros Z (Workers & GPUs Ativas)</h2>
                        <span class="text-secondary fs-7">Lista em tempo real dos nós processando no Range Completo</span>
                    </div>
                    <span class="badge badge-glow-cyan px-3 py-1.5 rounded-pill fs-7 font-mono">
                        Auto-Update: 3s
                    </span>
                </div>
                <div class="table-responsive">
                    <table class="table table-custom align-middle">
                        <thead>
                            <tr>
                                <th>Guerreiro / Worker ID</th>
                                <th>Hardware GPU</th>
                                <th>Poder de Luta (Hashrate)</th>
                                <th>Faixa Atribuída (%)</th>
                                <th>Sub-bloco Hex Atual</th>
                                <th>Chunks Validados</th>
                                <th>Status (Ping)</th>
                            </tr>
                        </thead>
                        <tbody id="workers-table-body">
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Active Target Jobs Overview -->
            <div class="glass-card p-3 p-md-4">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <div>
                        <h2 class="section-title mb-0">📋 Puzzles & Jobs de Busca Cadastrados</h2>
                        <span class="text-secondary fs-7">Gerenciamento de alvos e progresso salvo na VPS</span>
                    </div>
                    <span class="badge bg-dark border border-warning text-warning rounded-3 px-3 py-1.5 fs-7 font-mono">
                        🔒 Proteção SQLite WAL Ativa
                    </span>
                </div>
                <div class="table-responsive">
                    <table class="table table-custom align-middle">
                        <thead>
                            <tr>
                                <th>Puzzle / Job ID</th>
                                <th>Endereço BTC Alvo</th>
                                <th>Chave Pública Alvo</th>
                                <th>Faixa Executada (%)</th>
                                <th>Start Offset Hex</th>
                                <th>Range Bits</th>
                                <th>Status</th>
                                <th>Ações</th>
                            </tr>
                        </thead>
                        <tbody id="jobs-table-body">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            function copyText(str) {
                navigator.clipboard.writeText(str);
                alert("Copiado para a área de transferência!");
            }

            async function deleteSingleJob(jobId) {
                if (confirm("Apagar este job e o resultado de teste?")) {
                    await fetch('/api/jobs/delete/' + jobId, { method: 'POST' });
                    loadStats();
                }
            }

            async function clearOldJobs() {
                if (confirm("Tem certeza que deseja apagar TODOS os jobs e resultados de teste?")) {
                    await fetch('/api/jobs/clear', { method: 'POST' });
                    loadStats();
                }
            }

            async function loadStats() {
                try {
                    const res = await fetch('/api/stats');
                    if (res.status === 401) {
                        window.location.reload();
                        return;
                    }
                    const data = await res.json();

                    document.getElementById('pool-hashrate').innerText = data.total_pool_hashrate_ghs + ' GKeys/s';
                    document.getElementById('completed-chunks').innerText = data.dps_per_sec_str || '0 DPs/s';
                    document.getElementById('chunks-total-subtext').innerText = 'Eficiência ideal K ≈ 1.15 (0% DP Overhead)';
                    document.getElementById('dp-overhead').innerText = data.dp_overhead_str || '12%';
                    if (document.getElementById('dp-overhead-subtext')) {
                        document.getElementById('dp-overhead-subtext').innerText = data.k_subtext_str || 'K ≈ 1.28 (12% Overhead)';
                    }
                    document.getElementById('active-kangaroos').innerText = (data.total_gpus_count || 0) + ' GPUs';
                    document.getElementById('active-gpus-subtext').innerText = (data.active_kangaroos_m || '0.0M') + ' Kangaroos em execução (' + (data.active_workers_count || 0) + ' containers)';
                    document.getElementById('job-uptime').innerText = data.active_job_uptime || '0h 0m';
                    document.getElementById('keys-tested').innerText = data.keys_zetta_str || '0 Exakeys';
                    if (document.getElementById('keys-tested-subtext')) {
                        document.getElementById('keys-tested-subtext').innerText = 'Soma acumulada (' + (data.prob_pct_str || '0%') + ' exp. estatística)';
                    }

                    // Solved Banner (Super Saiyan Dragon Ball Theme)
                    const solvedContainer = document.getElementById('solved-solutions-container');
                    const solvedJobs = data.jobs.filter(j => j.status === 'SOLVED');

                    if (solvedJobs.length > 0) {
                        solvedContainer.innerHTML = solvedJobs.map(sj => `
                            <div class="glass-card solved-banner p-4 mb-4">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <div class="d-flex align-items-center gap-2">
                                        <span class="fs-2">🐉⭐</span>
                                        <h3 class="text-warning font-saiyan fw-bold mb-0 fs-3 glow-gold">DESEJO CONCEDIDO! CHAVE PRIVADA ENCONTRADA!</h3>
                                    </div>
                                    <div>
                                        <span class="badge badge-glow-gold fs-6 me-2 font-mono">${sj.solved_at_str || 'Recente'}</span>
                                        <button class="btn btn-outline-danger btn-sm font-semibold" onclick="deleteSingleJob('${sj.job_id}')">🗑️ Apagar Este Resultado</button>
                                    </div>
                                </div>
                                <div class="row g-3">
                                    <div class="col-md-6">
                                        <div class="stat-title">PUBKEY ALVO:</div>
                                        <div class="key-box text-info">${sj.pubkey}</div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="stat-title">ENCONTRADA POR WARRIOR:</div>
                                        <div class="key-box text-warning">${sj.solved_by || 'Desconhecido'}</div>
                                    </div>
                                    <div class="col-md-12">
                                        <div class="d-flex justify-content-between align-items-center mb-1">
                                            <div class="stat-title text-success font-semibold">CHAVE PRIVADA (HEX):</div>
                                            <button class="btn btn-sm btn-outline-success font-mono" onclick="copyText('${sj.private_key_hex}')">📋 Copiar HEX</button>
                                        </div>
                                        <div class="key-box text-success fw-bold fs-5">${sj.private_key_hex}</div>
                                    </div>
                                    <div class="col-md-12">
                                        <div class="d-flex justify-content-between align-items-center mb-1">
                                            <div class="stat-title text-warning font-semibold">CHAVE PRIVADA (FORMATO WIF COMPRESSADO):</div>
                                            <button class="btn btn-sm btn-outline-warning font-mono" onclick="copyText('${sj.wif_compressed}')">📋 Copiar WIF</button>
                                        </div>
                                        <div class="key-box text-warning fw-bold fs-5">${sj.wif_compressed}</div>
                                    </div>
                                </div>
                            </div>
                        `).join('');
                    } else {
                        solvedContainer.innerHTML = '';
                    }

                    // Render Active Workers & Their Live Started Ranges
                    const workersBody = document.getElementById('workers-table-body');
                    if (data.workers.length === 0) {
                        workersBody.innerHTML = `<tr><td colspan="7" class="text-center text-secondary p-4">Nenhum guerreiro ativo no momento. Ao conectar um worker (local/Vast.ai), o poder dele aparecerá aqui automaticamente.</td></tr>`;
                    } else {
                        workersBody.innerHTML = data.workers.map(w => {
                            const hrStr = w.hashrate_mhs >= 1000 ? (w.hashrate_mhs / 1000.0).toFixed(2) + ' GH/s' : w.hashrate_mhs.toFixed(2) + ' MH/s';
                            const gCount = w.gpu_info ? Math.max(1, w.gpu_info.split(',').length) : 1;
                            const perGpuMhs = w.hashrate_mhs / gCount;
                            const perGpuStr = gCount > 1 ? (perGpuMhs >= 1000 ? ` <br><span class="text-secondary font-mono fs-7">(${(perGpuMhs / 1000.0).toFixed(2)} GH/s/GPU)</span>` : ` <br><span class="text-secondary font-mono fs-7">(${perGpuMhs.toFixed(1)} MH/s/GPU)</span>`) : '';
                            const chunksBadge = w.completed_chunks > 0 ?
                                `<span class="badge badge-glow-green px-3 py-1 fw-bold font-mono">${w.completed_chunks} chunks</span>` :
                                `<span class="badge bg-dark text-secondary px-2 py-1 font-mono">${w.completed_chunks} chunks</span>`;
                            const rangeBadge = `<span class="badge badge-glow-cyan font-mono px-3 py-1.5 fw-semibold" style="font-size: 0.88rem;">${w.assigned_range || '0% → 100%'}</span>`;
                            const hexBadge = `<span class="badge bg-black border border-secondary text-info font-mono px-2 py-1">${w.current_start_hex || 'Iniciando...'}</span>`;
                            const pingSecs = Math.max(0, Math.round(Date.now()/1000 - w.last_ping));
                            
                            return `
                            <tr>
                                <td>
                                    <strong class="text-warning font-saiyan fs-6 glow-gold">${w.name}</strong><br>
                                    <code class="text-secondary fs-7 font-mono">${w.worker_id}</code>
                                </td>
                                <td><span class="text-light fs-7 fw-semibold">${w.gpu_info}</span></td>
                                <td><span style="color: var(--accent-gold); font-weight: 800; font-family: 'Orbitron', sans-serif;" class="fs-6">${hrStr}</span>${perGpuStr}</td>
                                <td>${rangeBadge}</td>
                                <td>${hexBadge}</td>
                                <td>${chunksBadge}</td>
                                <td><span class="badge badge-glow-green px-2.5 py-1 fs-7">🟢 Ativo (${pingSecs}s atrás)</span></td>
                            </tr>
                            `;
                        }).join('');
                    }

                    // Target Active Puzzle Card
                    const targetContainer = document.getElementById('active-target-puzzle-container');
                    const activeJobs = data.jobs.filter(j => j.status === 'ACTIVE');

                    if (activeJobs.length > 0) {
                        const firstActive = activeJobs[0];
                        const totalAssigned = activeJobs.reduce((acc, j) => acc + (j.total_chunks_assigned || 0), 0);
                        const totalCompleted = activeJobs.reduce((acc, j) => acc + (j.completed_chunks || 0), 0);
                        const minStart = Math.min(...activeJobs.map(j => j.start_percent));
                        const maxEnd = Math.max(...activeJobs.map(j => j.end_percent));
                        const rangeBadges = activeJobs.map(j => `<span class="badge badge-glow-cyan font-mono me-1 mb-1" title="Job ${j.job_id}">${j.start_percent.toFixed(1)}% → ${j.end_percent.toFixed(1)}%</span>`).join('');

                        targetContainer.innerHTML = `
                            <div class="glass-card p-4 border border-warning">
                                <div class="d-flex flex-column flex-md-row justify-content-between align-items-md-center gap-2 mb-3">
                                    <div class="d-flex align-items-center">
                                        <span class="fs-3 me-2">🎯</span>
                                        <h3 class="mb-0 text-warning font-saiyan fw-bold h4 glow-gold">${firstActive.puzzle_name}</h3>
                                        <span class="badge badge-glow-green ms-3 fs-7">Em Busca Ativa (${activeJobs.length} Jobs Simultâneos)</span>
                                    </div>
                                    <div class="d-flex flex-wrap align-items-center gap-2">
                                        <span class="badge bg-black border border-secondary text-light fs-7 font-mono">Range Total: ${minStart.toFixed(1)}% → ${maxEnd.toFixed(1)}%</span>
                                        <span class="badge badge-glow-purple fs-7 font-mono">Chunks Concluídos: ${totalCompleted} / ${totalAssigned}</span>
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <span class="stat-title me-2">FAIXAS ATIVAS EM PROCESSAMENTO:</span>
                                    <div class="d-inline-flex flex-wrap mt-1">${rangeBadges}</div>
                                </div>
                                <div class="row g-3">
                                    <div class="col-md-6">
                                        <div class="stat-title">CHAVE PÚBLICA ALVO (PUBLIC KEY):</div>
                                        <div class="d-flex align-items-center mt-1">
                                            <div class="key-box text-info flex-grow-1 me-2 fs-7">${firstActive.pubkey}</div>
                                            <button class="btn btn-sm btn-outline-info font-mono" onclick="copyText('${firstActive.pubkey}')">📋 Copiar</button>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="stat-title">ENDEREÇO BITCOIN ALVO:</div>
                                        <div class="d-flex align-items-center mt-1">
                                            <div class="key-box text-warning flex-grow-1 me-2 fs-7 font-semibold" style="color: var(--accent-gold);">${firstActive.btc_address}</div>
                                            <button class="btn btn-sm btn-outline-warning font-mono" onclick="copyText('${firstActive.btc_address}')">📋 Copiar</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                    } else {
                        targetContainer.innerHTML = '';
                    }

                    // Render Jobs Table
                    const jobsBody = document.getElementById('jobs-table-body');
                    if (data.jobs.length === 0) {
                        jobsBody.innerHTML = `<tr><td colspan="8" class="text-center text-secondary p-3">Nenhum job cadastrado no momento.</td></tr>`;
                    } else {
                        jobsBody.innerHTML = data.jobs.map(j => `
                            <tr>
                                <td>
                                    <strong class="text-warning font-saiyan">${j.puzzle_name}</strong><br>
                                    <code class="text-secondary fs-7 font-mono">${j.job_id}</code>
                                </td>
                                <td>
                                    <span class="text-warning fw-semibold font-mono">${j.btc_address}</span>
                                    <button class="btn btn-sm btn-link text-warning p-0 ms-1" onclick="copyText('${j.btc_address}')" title="Copiar Endereço">📋</button>
                                </td>
                                <td>
                                    <code class="text-truncate d-inline-block font-mono text-secondary" style="max-width: 180px;" title="${j.pubkey}">${j.pubkey}</code>
                                </td>
                                <td><span class="badge badge-glow-cyan font-mono">${j.start_percent}% → ${j.end_percent}%</span></td>
                                <td><code class="font-mono text-secondary">0x${j.start_hex}</code></td>
                                <td>
                                    <span class="font-mono text-light">${j.range_bits} bits</span><br>
                                    <span class="badge badge-glow-purple fs-7 font-mono">${j.completed_chunks} / ${j.total_chunks_assigned} chunks</span>
                                </td>
                                <td><span class="badge ${j.status === 'SOLVED' ? 'badge-glow-green' : 'badge-glow-cyan'} font-mono">${j.status}</span></td>
                                <td class="font-mono">
                                    ${j.status === 'SOLVED' ? `
                                        <span class="text-success font-semibold me-2">${j.private_key_hex}</span>
                                        <button class="btn btn-sm btn-outline-success" onclick="copyText('${j.private_key_hex}')">📋 Copiar</button>
                                    ` : `
                                        <span class="badge badge-glow-green fs-7 font-mono">🟢 Protegido</span>
                                    `}
                                </td>
                            </tr>
                        `).join('');
                    }

                    // Logs da VPS em Tempo Real
                    try {
                        const logsRes = await fetch('/api/logs');
                        if (logsRes.ok) {
                            const logsData = await logsRes.json();
                            const logsBox = document.getElementById('vps-logs-box');
                            if (logsBox && logsData.logs && logsData.logs.length > 0) {
                                logsBox.innerText = logsData.logs.join('\\n');
                                logsBox.scrollTop = logsBox.scrollHeight;
                            }
                        }
                    } catch(e) {}

                } catch (e) {
                    console.error('Erro ao carregar dados:', e);
                }
            }

            async function openCoverageModal() {
                try {
                    const res = await fetch('/api/coverage');
                    const data = await res.json();
                    document.getElementById('coverage-report-content').innerText = data.text;
                    const modal = new bootstrap.Modal(document.getElementById('coverageModal'));
                    modal.show();
                } catch(e) {
                    alert("Erro ao carregar COVERAGE.TXT: " + e);
                }
            }

            loadStats();
            setInterval(loadStats, 3000);
        </script>

        <!-- Coverage Report Modal -->
        <div class="modal fade" id="coverageModal" tabindex="-1" aria-hidden="true">
            <div class="modal-dialog modal-lg modal-dialog-centered">
                <div class="modal-content glass-card border border-warning shadow-lg">
                    <div class="modal-header border-secondary">
                        <h5 class="modal-title text-warning font-saiyan font-semibold glow-gold">📄 Relatório de Cobertura (COVERAGE.TXT)</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div class="mb-3">
                            <span class="text-secondary fs-7">Relatório oficial gerado na VPS (`/opt/rckangaroo/pool/server/COVERAGE.TXT`) garantindo anti-sobreposição de faixas:</span>
                        </div>
                        <pre id="coverage-report-content" class="bg-black text-success p-3 rounded-3 border border-secondary font-mono" style="white-space: pre-wrap; font-size: 0.88rem; max-height: 400px; overflow-y: auto;"></pre>
                    </div>
                    <div class="modal-footer border-secondary">
                        <button type="button" class="btn btn-sm btn-secondary" data-bs-dismiss="modal">Fechar</button>
                        <a href="/api/coverage" target="_blank" class="btn btn-sm btn-outline-warning font-mono">📥 Abrir COVERAGE.TXT Direto</a>
                    </div>
                </div>
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
