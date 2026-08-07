import os
import sys
import time
import subprocess
import hashlib

sys.stdout.reconfigure(encoding='utf-8')

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root_dir, "x64", "Release", "RCKangaroo.exe"),
        os.path.join(root_dir, "rckangaroo"),
        os.path.join(root_dir, "RCKangaroo.exe"),
    ]
    bin_path = next((p for p in candidates if os.path.exists(p)), None)

    if not bin_path:
        print("❌ Binário RCKangaroo não encontrado para o teste!")
        sys.exit(1)

    print("==================================================================")
    print("🦘 RCKangaroo — TESTE DE COMPORTAMENTO DE ARQUIVO (-tames)")
    print("==================================================================")

    test_file = os.path.join(os.path.dirname(bin_path), "test_tames.dat")
    kill_file = os.path.join(os.path.dirname(bin_path), "test_tames_kill.dat")

    for f in [test_file, kill_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    range_val = "76"
    dp_val = "16"
    start_val = "10000000000000000000"

    # --- TESTE 1: RUN 1 (-max 0.01) ---
    print("\n▶️ [RUN 1] Executando com -max 0.01...")
    cmd1 = [bin_path, "-gpu", "0", "-dp", dp_val, "-range", range_val, "-start", start_val, "-tames", "test_tames.dat", "-max", "0.01"]
    res1 = subprocess.run(cmd1, cwd=os.path.dirname(bin_path), capture_output=True, text=True)
    if res1.stdout:
        print("[OUT1]:", res1.stdout.strip())
    if res1.stderr:
        print("[ERR1]:", res1.stderr.strip())

    if not os.path.exists(test_file):
        print("❌ Falha: Arquivo test_tames.dat não foi criado no RUN 1!")
        sys.exit(1)

    size1 = os.path.getsize(test_file)
    with open(test_file, "rb") as f:
        md5_1 = hashlib.md5(f.read()).hexdigest()

    print(f"   • Tamanho após RUN 1 (-max 0.1) : {size1:,} bytes")
    print(f"   • MD5 Hash após RUN 1            : {md5_1}")

    # --- TESTE 1: RUN 2 (-max 0.3) ---
    print("\n▶️ [RUN 2] Executando novamente o mesmo comando com -max 0.3 (arquivo já existente)...")
    cmd2 = [bin_path, "-gpu", "0", "-dp", dp_val, "-range", range_val, "-start", start_val, "-tames", "test_tames.dat", "-max", "0.3"]
    subprocess.run(cmd2, cwd=os.path.dirname(bin_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    size2 = os.path.getsize(test_file)
    with open(test_file, "rb") as f:
        md5_2 = hashlib.md5(f.read()).hexdigest()

    print(f"   • Tamanho após RUN 2 (-max 0.3) : {size2:,} bytes")
    print(f"   • MD5 Hash após RUN 2            : {md5_2}")

    print("\n📊 RESULTADO DO TESTE 1 (Acumulação vs Sobrescrita):")
    if size2 > size1:
        print(f"   ✅ O arquivo CRESCEU ({size1:,} -> {size2:,} bytes) -- INDÍCIO DE ACUMULAÇÃO!")
    elif size1 == size2:
        if md5_1 == md5_2:
            print("   ⚠️ O arquivo ficou IDÊNTICO (mesmo tamanho e hash) -- Modo de Geração SOBRESCREVE do zero!")
        else:
            print("   ⚠️ Mesmo tamanho, hash diferente -- Recriou do zero com número de DPs limitado.")
    else:
        print(f"   ⚠️ Tamanho reduziu ({size1:,} -> {size2:,} bytes) -- Arquivo foi recriado/sobrescrito do zero!")

    # --- TESTE 2: KILL TEST ---
    print("\n▶️ [TESTE 2] Executando com -max 10.0 e aplicando Interrupção (Kill) em 3 segundos...")
    cmd_kill = [bin_path, "-gpu", "0", "-dp", dp_val, "-range", range_val, "-start", start_val, "-tames", "test_tames_kill.dat", "-max", "10.0"]
    proc = subprocess.Popen(cmd_kill, cwd=os.path.dirname(bin_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except Exception:
        proc.kill()

    print("\n📊 RESULTADO DO TESTE 2 (Comportamento de Salvamento ao Interromper):")
    if os.path.exists(kill_file):
        kill_size = os.path.getsize(kill_file)
        print(f"   ✅ Arquivo EXISTE após o kill: {kill_size:,} bytes")
    else:
        print("   ⚠️ Arquivo NÃO EXISTE após o kill -- Confirma que só salva ao TERMINAR naturalmente ou no limite -max!")

    # Limpeza dos arquivos de teste
    for f in [test_file, kill_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

if __name__ == "__main__":
    main()
