import os
import sys

def check():
    print("🔍 INICIANDO DIAGNÓSTICO DO AMBIENTE...")
    print("-" * 50)
    
    print(f"🐍 Python Executável: {sys.executable}")
    print(f"🐍 Python Versão: {sys.version.split()[0]}")
    
    print("\n📦 PACOTES PIP (TENSORFLOW/ROCm) INSTALADOS:")
    os.system(f"{sys.executable} -m pip list | grep -iE 'tensor|rocm'")
    
    print("\n📂 VERIFICANDO INSTALAÇÃO DO ROCm (/opt/rocm):")
    if os.path.exists("/opt/rocm"):
        print("✅ Diretório /opt/rocm encontrado.")
        os.system("cat /opt/rocm/.info/version 2>/dev/null || echo 'Arquivo de versão não encontrado.'")
    else:
        print("❌ Diretório /opt/rocm NÃO encontrado!")

    print("\n🔎 BUSCANDO A BIBLIOTECA FÍSICA 'libhipsparselt':")
    os.system("find /opt/rocm /usr/lib /usr/local/lib /lib -name 'libhipsparselt.so*' 2>/dev/null")
    
    print("\n🔗 VERIFICANDO CACHE DO SISTEMA LINUX (ldconfig):")
    os.system("ldconfig -p | grep hipsparselt || echo '❌ Nenhuma referência no ldconfig'")
    
    print("\n🐧 VERSÃO DO LINUX/UBUNTU:")
    os.system("cat /etc/os-release | grep PRETTY_NAME")
    
    print("-" * 50)

if __name__ == "__main__":
    check()