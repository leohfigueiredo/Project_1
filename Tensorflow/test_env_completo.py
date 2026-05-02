print("=== TESTE COMPLETO DO AMBIENTE ML ===")
print()

# 1. TensorFlow
print("1. TensorFlow:")
import tensorflow as tf
print(f"   Versão: {tf.__version__}")
print(f"   GPU disponível: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"   Dispositivos: {[d.device_type for d in tf.config.list_physical_devices()]}")

print()

# 2. PyTorch
print("2. PyTorch:")
import torch
print(f"   Versão: {torch.__version__}")
print(f"   ROCm/CUDA disponível: {torch.cuda.is_available()}")
print(f"   Backend: {'CPU' if not torch.cuda.is_available() else 'GPU (ROCm)'}")

print()

# 3. NumPy e dados
print("3. Bibliotecas de dados:")
import numpy as np
import pandas as pd
import sklearn
print(f"   NumPy: {np.__version__}")
print(f"   Pandas: {pd.__version__}")
print(f"   Scikit-learn: {sklearn.__version__}")

print()

# 4. Teste prático
print("4. Teste prático:")
# TensorFlow
x_tf = tf.constant([[1, 2], [3, 4]])
print(f"   TensorFlow tensor shape: {x_tf.shape}")

# PyTorch
x_pt = torch.tensor([[1, 2], [3, 4]])
print(f"   PyTorch tensor shape: {x_pt.shape}")

# NumPy
x_np = np.array([[1, 2], [3, 4]])
print(f"   NumPy array shape: {x_np.shape}")

print()
print("✅ Ambiente configurado e funcionando!")
print("   Verifique acima se as GPUs foram detectadas e estão retornando 'True'.")
print("   Lembre-se: ROCm tem suporte nativo no Linux (ou WSL2 para Windows).")