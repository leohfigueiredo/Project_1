import sys
print("Python:", sys.version)
print()

libs = [
    ("numpy", "1.24.3"),
    ("tensorflow", "2.13.0"),
    ("torch", "2.0.1"),
    ("pandas", "1.5.3"),
    ("sklearn", "1.3.0"),
    ("jupyter", "1.0.0")
]

for lib, ver in libs:
    try:
        if lib == "sklearn":
            import sklearn
            v = sklearn.__version__
        else:
            exec(f"import {lib}")
            v = eval(f"{lib}.__version__")
        
        if v == ver:
            print(f"✅ {lib}=={v}")
        else:
            print(f"⚠️  {lib}=={v} (esperado: {ver})")
    except Exception as e:
        print(f"❌ {lib}: {str(e)[:50]}")

print("\n" + "="*50)
print("Para testar Jupyter:")
print("1. Crie novo arquivo .ipynb")
print("2. Selecione kernel 'Python (ML Windows)'")
print("3. Execute: import tensorflow as tf")