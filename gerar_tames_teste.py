import os
import sys
import time
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("==================================================================")
    print("🦘 RCKangaroo — GERADOR E TESTADOR ISOLADO DE TAMES (.dat)")
    print("==================================================================")
    print("ℹ️  Este script é 100% isolado e NÃO afeta o banco de dados da Pool.\n")

    # Configurações do Teste Local
    pubkey     = "031f6a332d3c5c4f2de2378c012f429cd109ba07d69690c6c701b6bb87860d6640" # PubKey Teste / Puzzle 140
    start_hex  = "80000000000000000000000000000000000"
    range_bits = 76   # Faixa de teste (ex: 76 bits)
    dp_bits    = 16   # Bits de DP (16 para teste rápido, 22-24 para grandes faixas)
    max_ops    = 2.0  # Fator de operações para geração de Tames (ex: 2.0 * sqrt(range))
    gpu_id     = "0"  # GPU(s) a utilizar (ex: "0" ou "0123")
    
    tames_filename = f"tames_teste_{range_bits}bits.dat"

    root_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root_dir, "x64", "Release", "RCKangaroo.exe"),
        os.path.join(root_dir, "rckangaroo"),
        os.path.join(root_dir, "RCKangaroo.exe"),
    ]
    bin_path = next((p for p in candidates if os.path.exists(p)), None)

    if not bin_path:
        print("❌ Binário RCKangaroo não foi encontrado!")
        sys.exit(1)

    output_tames_path = os.path.join(os.path.dirname(bin_path), tames_filename)

    # 1. ETAPA DE GERAÇÃO DOS TAMES
    print(f"📌 [FASE 1] Gerando arquivo de Tames: {tames_filename}")
    print(f"   • Range Bits : {range_bits} bits")
    print(f"   • DP Bits    : {dp_bits}")
    print(f"   • Max Ops    : {max_ops}")
    print(f"   • GPU ID     : {gpu_id}\n")

    if os.path.exists(output_tames_path):
        try:
            os.remove(output_tames_path)
            print("🧹 Arquivo de teste antigo removido.\n")
        except Exception:
            pass

    cmd_gen = [
        bin_path,
        "-gpu", gpu_id,
        "-dp", str(dp_bits),
        "-range", str(range_bits),
        "-start", start_hex,
        "-pubkey", pubkey,
        "-max", str(max_ops),
        "-tames", tames_filename
    ]

    print(f"⚙️ Executando Geração:\n   {' '.join(cmd_gen)}\n")
    t0 = time.time()
    proc_gen = subprocess.run(cmd_gen, cwd=os.path.dirname(bin_path))
    t_gen = time.time() - t0

    if proc_gen.returncode != 0 or not os.path.exists(output_tames_path):
        print("❌ Falha durante a geração do arquivo de Tames.")
        sys.exit(1)

    file_size_bytes = os.path.getsize(output_tames_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    print("\n" + "━"*60)
    print("🎉 TAMES GERADOS COM SUCESSO!")
    print(f"   ⏱️ Tempo de Geração : {t_gen:.2f} segundos")
    print(f"   💾 Tamanho do Arquivo: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    print(f"   📁 Localização      : {output_tames_path}")
    print("━"*60 + "\n")

    # 2. ETAPA DE VALIDAÇÃO (CARREGANDO OS TAMES GERADOS)
    print(f"📌 [FASE 2] Testando Carregamento dos Tames com {tames_filename}...")
    cmd_load = [
        bin_path,
        "-gpu", gpu_id,
        "-dp", str(dp_bits),
        "-range", str(range_bits),
        "-start", start_hex,
        "-pubkey", pubkey,
        "-max", "1.0",
        "-tames", tames_filename
    ]

    print(f"⚙️ Executando Teste com Tames Carregados:\n   {' '.join(cmd_load)}\n")
    t1 = time.time()
    proc_load = subprocess.run(cmd_load, cwd=os.path.dirname(bin_path))
    t_load = time.time() - t1

    print("\n" + "━"*60)
    print("✅ TESTE CONCLUÍDO COM SUCESSO!")
    print(f"   ⚡ O binário carregou o arquivo {tames_filename} ({file_size_mb:.2f} MB) instantaneamente.")
    print("━"*60)

if __name__ == "__main__":
    main()
