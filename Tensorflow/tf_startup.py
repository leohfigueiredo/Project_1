"""
═══════════════════════════════════════════════════════════════════════
TensorFlow Startup Configuration - AMD ROCm (GPU)
Otimizado para aceleração via Placas de Vídeo AMD (ROCm)
═══════════════════════════════════════════════════════════════════════

USO:
    # PRIMEIRA linha do seu notebook/script:
    import tf_startup
    
    # Depois importe o resto:
    import numpy as np
    import pandas as pd
    # ... seu código

IMPORTANTE: 
- Importe ANTES de qualquer outra coisa
- Se já usou TensorFlow, reinicie o kernel primeiro
═══════════════════════════════════════════════════════════════════════
"""

import os
import warnings

# ═════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES DE AMBIENTE (antes de importar TensorFlow)
# ═════════════════════════════════════════════════════════════════════

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# === HACK PARA GEEKOM A9 MAX (Radeon 780M - iGPU) ===
# Força o ROCm a aceitar a arquitetura da iGPU do seu Ryzen 9
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

# Ajuda o sistema a encontrar as bibliotecas do ROCm caso estejam na pasta padrão
os.environ['LD_LIBRARY_PATH'] = f"/opt/rocm/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
# Configurações críticas de PATH para evitar o erro "undefined symbol"
ROCM_PATH = "/opt/rocm/lib"
if ROCM_PATH not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = f"{ROCM_PATH}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Garante que kernels dinâmicos do TF encontrem as libs do ROCm 6.2
os.environ['TF_ROCM_AMDGPU_TARGETS'] = 'gfx1103'

# ═════════════════════════════════════════════════════════════════════
# IMPORTAR TENSORFLOW
# ═════════════════════════════════════════════════════════════════════

import sys

# Bloqueia a carga de versões conflitantes do protobuf/abseil se necessário
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

try:
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
except ImportError as e:
    raise ImportError("❌ ERRO CRÍTICO: TensorFlow não encontrado! Verifique se você selecionou o kernel correto (ex: ml_env_311) no canto superior direito do VS Code.") from e

# Configurar alocação dinâmica de memória na GPU (Memory Growth)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ AVISO: Nenhuma GPU (ROCm) detectada pelo TensorFlow. O treino será feito na CPU.")

# Mixed precision (float16) - ~30% mais rápido
mixed_precision.set_global_policy('mixed_float16')

# ═════════════════════════════════════════════════════════════════════
# FUNÇÕES HELPER OTIMIZADAS
# ═════════════════════════════════════════════════════════════════════

def create_fast_model(input_shape, num_classes, hidden_units=[512, 256]):
    """
    Cria modelo otimizado para GPU AMD (ROCm)
    
    Args:
        input_shape: tuple - Shape da entrada, ex: (784,)
        num_classes: int - Número de classes
        hidden_units: list - Neurônios por camada, ex: [512, 256]
    
    Returns:
        tf.keras.Sequential
    
    Exemplo:
        model = create_fast_model((784,), 10, [512, 256])
    """
    layers = [tf.keras.layers.InputLayer(input_shape=input_shape)]
    
    for units in hidden_units:
        layers.append(tf.keras.layers.Dense(
            units,
            activation='relu',
            kernel_initializer='he_normal'
        ))
    
    layers.append(tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        dtype='float32'  # Importante para mixed precision
    ))
    
    return tf.keras.Sequential(layers)


def get_fast_optimizer(learning_rate=0.01):
    """
    Retorna Adam (Excelente performance em GPU)
    
    Args:
        learning_rate: float - Taxa de aprendizado (padrão: 0.01)
    
    Returns:
        tf.keras.optimizers.SGD
    
    Exemplo:
        optimizer = get_fast_optimizer(0.01)
    """
    return tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )


def compile_fast(model, optimizer=None, loss='sparse_categorical_crossentropy', 
                 metrics=['accuracy']):
    """
    Compila modelo com configurações otimizadas
    
    Args:
        model: Modelo a compilar
        optimizer: Optimizer (None = SGD otimizado)
        loss: Função de perda
        metrics: Lista de métricas
    
    Exemplo:
        compile_fast(model)
    """
    if optimizer is None:
        optimizer = get_fast_optimizer()
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def optimize_dataset(X, y, batch_size=256, shuffle=True, cache=True):
    """
    Cria dataset otimizado com cache e prefetch
    
    Args:
        X: numpy array - Features
        y: numpy array - Labels
        batch_size: int - Tamanho do batch (Depende da VRAM da sua GPU, ex: 256-1024)
        shuffle: bool - Embaralhar dados
        cache: bool - Cachear em memória
    
    Returns:
        tf.data.Dataset
    
    Exemplo:
        train_ds = optimize_dataset(X_train, y_train, batch_size=256)
        model.fit(train_ds, epochs=10)
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(X), 10000))
    
    if cache:
        dataset = dataset.cache()
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ═════════════════════════════════════════════════════════════════════
# INFORMAÇÕES
# ═════════════════════════════════════════════════════════════════════

print("═" * 70)
print("✅ TensorFlow Otimizado - GPU AMD (ROCm)")
print("═" * 70)
print(f"  🚀 Performance: Aceleração via GPU Ativada")
print(f"  💻 GPUs Detectadas: {len(gpus)}")
print(f"  ⚡ Mixed Precision: float16")
print(f"  🔧 Memory Growth: Habilitado")
print(f"  📦 TensorFlow: {tf.__version__}")
print("═" * 70)
print("\n📝 FUNÇÕES DISPONÍVEIS:")
print("  • create_fast_model(input_shape, num_classes, hidden_units)")
print("  • get_fast_optimizer(learning_rate=0.01)")
print("  • compile_fast(model, optimizer=None)")
print("  • optimize_dataset(X, y, batch_size=256)")
print("\n💡 EXEMPLO RÁPIDO:")
print("""
model = create_fast_model((784,), 10)
compile_fast(model)
train_ds = optimize_dataset(X_train, y_train)
model.fit(train_ds, epochs=10)
""")
print("═" * 70)
print("✓ Pronto para usar! Importe seus dados e comece a treinar.\n")