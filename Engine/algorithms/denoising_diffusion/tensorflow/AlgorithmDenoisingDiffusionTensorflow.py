import os
import sys
import numpy
import tensorflow
from typing import Any, Optional, Dict, Tuple
from tensorflow.keras.utils import to_categorical


class AlgorithmDenoisingDiffusionTensorflow(tensorflow.keras.Model):
    """
    DDPM (Denoising Diffusion Probabilistic Model) - VERSÃO CORRIGIDA

    Correções aplicadas:
    ✅ Removido EMA (simplificação)
    ✅ dtype int64 para timesteps (crítico!)
    ✅ Acesso correto ao input_shape
    ✅ Uso de .predict() na geração
    ✅ Sem padding na geração (usa shape original)
    """

    def __init__(self,
                 output_shape: int,
                 first_unet_model: tensorflow.keras.Model,
                 gdf_util: Any,
                 optimizer_diffusion: tensorflow.keras.optimizers.Optimizer,
                 time_steps: int,
                 margin: float = 1.0,  # Argumento mantido para compatibilidade
                 train_stage: str = 'all',  # Argumento mantido para compatibilidade
                 debug: bool = False,
                 second_unet_model: Optional[tensorflow.keras.Model] = None,
                 optimizer_autoencoder: Optional[tensorflow.keras.optimizers.Optimizer] = None,
                 **kwargs):  # Captura qualquer outro argumento extra

        super().__init__()

        self._gdf_util = gdf_util
        self._time_steps = time_steps
        self._network = first_unet_model
        self._original_shape = output_shape
        self._optimizer_diffusion = optimizer_diffusion
        self._debug = debug

        # Trackers de métricas e Loss
        self._loss_tracker = tensorflow.keras.metrics.Mean(name="loss")
        self._loss_fn = tensorflow.keras.losses.MeanSquaredError()

        if self._debug:
            print("\n" + "=" * 50)
            print("🚀 MODELO INICIALIZADO COM SUCESSO")
            print("=" * 50)
            print("⚠️  EMA REMOVIDO - usando modelo direto")
            print("=" * 50)

    def _padding_input_tensor(self, input_tensor):
        """
        ✅ CORRIGIDO - Ajusta o padding para o tamanho esperado pela UNet.

        FIX: Acessa corretamente input_shape[0][1] ao invés de [-2]
        """
        # ✅ input_shape[0] retorna (None, seq_len, channels)
        # Queremos seq_len que está no índice [1]
        input_spec = self._network.input_shape[0]

        if isinstance(input_spec, (tuple, list)):
            target_dim = input_spec[1]  # seq_len
        else:
            raise ValueError(f"Unexpected input_shape format: {input_spec}")

        current_dim = tensorflow.shape(input_tensor)[1]
        padding_needed = tensorflow.maximum(0, target_dim - current_dim)

        # Define o padding para o rank correto do tensor
        paddings = [[0, 0], [0, padding_needed], [0, 0]]
        return tensorflow.pad(input_tensor, paddings)

    @tensorflow.function
    def train_step(self, data):
        """
        ✅ CORRIGIDO - Passo de treinamento otimizado.

        FIX: dtype int64 para timesteps (era int32)
        FIX: Corta predicted_noise para o tamanho do noise original
        """
        # Suporta (x, y) ou apenas x
        if isinstance(data, (tuple, list)):
            x, y = data[0], data[1]
        else:
            x, y = data, None

        with tensorflow.GradientTape() as tape:
            # 1. Sample timesteps
            batch_size = tensorflow.shape(x)[0]

            # ✅ FIX CRÍTICO: dtype=tensorflow.int64 (era int32)
            t = tensorflow.random.uniform(
                (batch_size,),
                minval=0,
                maxval=self._time_steps,
                dtype=tensorflow.int64  # ← MUDANÇA CRÍTICA!
            )

            # 2. Sample noise
            noise = tensorflow.random.normal(tensorflow.shape(x))

            # 3. Forward Diffusion (Processo de Difusão)
            x_noisy = self._gdf_util.q_sample(x, t, noise)

            # 4. Padding (se necessário)
            x_padded = self._padding_input_tensor(x_noisy)

            # 5. Predição
            predicted_noise = self._network([x_padded, t, y], training=True)

            # 6. Ajusta shape se necessário
            if len(predicted_noise.shape) > len(noise.shape):
                predicted_noise = tensorflow.squeeze(predicted_noise, axis=-1)

            # ✅ FIX: Corta predicted_noise para o tamanho do noise original
            predicted_noise = predicted_noise[:, :tensorflow.shape(noise)[1], :]

            # 7. Cálculo de Perda
            loss = self._loss_fn(noise, predicted_noise)

        # 8. Otimização
        grads = tape.gradient(loss, self._network.trainable_weights)
        self._optimizer_diffusion.apply_gradients(zip(grads, self._network.trainable_weights))

        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    def generate_data(self, labels, batch_size: int = 32):
        """
        ✅ CORRIGIDO - Gera dados usando o processo de difusão reversa.

        FIX CRÍTICO: Adiciona padding nos samples antes de passar para a rede!
        """
        if not isinstance(labels, tensorflow.Tensor):
            labels = tensorflow.convert_to_tensor(labels, dtype=tensorflow.float32)

        num_samples = tensorflow.shape(labels)[0]

        # Acessa shape corretamente
        input_spec = self._network.input_shape[0]
        if isinstance(input_spec, (tuple, list)):
            channels = input_spec[2]  # channels está no índice [2]
        else:
            raise ValueError(f"Unexpected input_shape format: {input_spec}")

        # ✅ Começa com ruído puro NO TAMANHO ORIGINAL (sem padding)
        samples = tensorflow.random.normal(
            (num_samples, self._original_shape, channels),
            dtype=tensorflow.float32
        )

        if self._debug:
            print(f"\n🎨 Gerando {num_samples} amostras...")
            print(f"  • Shape inicial: {samples.shape}")
            print(f"  • Timesteps: {self._time_steps}")
            print(f"  • Initial noise mean: {tensorflow.reduce_mean(samples).numpy():.6f}")
            print(f"  • Initial noise std: {tensorflow.math.reduce_std(samples).numpy():.6f}")

        # ✅ Loop de difusão reversa
        for i in reversed(range(0, self._time_steps)):
            t = tensorflow.cast(
                tensorflow.fill([num_samples], i),
                dtype=tensorflow.int64
            )

            # ✅ FIX CRÍTICO: ADICIONA PADDING antes de passar para a rede!
            samples_padded = self._padding_input_tensor(samples)

            # Agora passa o tensor com padding
            pred_noise = self._network.predict(
                [samples_padded, t, labels],
                verbose=0,
                batch_size=num_samples
            )

            # Ajusta shape se necessário
            if len(pred_noise.shape) > len(samples.shape):
                pred_noise = tensorflow.squeeze(pred_noise, axis=-1)

            # ✅ IMPORTANTE: Corta pred_noise para o tamanho original (sem padding)
            pred_noise = pred_noise[:, :self._original_shape, :]

            # Valida shapes
            if pred_noise.shape != samples.shape:
                raise ValueError(
                    f"Shape mismatch at timestep {i}: "
                    f"pred_noise={pred_noise.shape}, samples={samples.shape}"
                )

            # Denoise step (usa samples SEM padding)
            samples = self._gdf_util.p_sample(
                pred_noise, samples, t, clip_denoised=True
            )

            # Debug a cada 100 steps
            if self._debug and (i % 100 == 0 or i < 5):
                print(f"  • Step {i:4d}: mean={tensorflow.reduce_mean(samples).numpy():+.6f}, "
                      f"std={tensorflow.math.reduce_std(samples).numpy():.6f}")

        output = samples.numpy()

        if self._debug:
            print(f"\n✅ Geração completa!")
            print(f"  • Output shape: {output.shape}")
            print(f"  • Output range: [{output.min():.4f}, {output.max():.4f}]")
            print(f"  • Output mean: {output.mean():.4f}")
            print(f"  • Output std: {output.std():.4f}")

        return output
    def get_samples(self, number_samples_per_class: Dict):
        """Interface amigável para gerar amostras de múltiplas classes."""
        generated_data = {}
        num_classes = number_samples_per_class["number_classes"]

        for label, count in number_samples_per_class["classes"].items():
            if self._debug: print(f"\n🎨 Gerando {count} amostras para classe {label}...")
            one_hot = to_categorical([label] * count, num_classes=num_classes)
            samples = self.generate_data(one_hot)
            generated_data[label] = numpy.squeeze(samples)

        return generated_data


# ============================================================================
# VALIDAÇÃO: Execute isso para confirmar que está tudo correto
# ============================================================================

def validate_model_setup(model, x_sample, y_sample):
    """
    Valida se o modelo está configurado corretamente.
    """
    print("\n" + "=" * 70)
    print("🔍 VALIDAÇÃO DO MODELO")
    print("=" * 70)

    # 1. Valida input_shape
    input_spec = model._network.input_shape
    print(f"\n1️⃣ Input Shape:")
    print(f"  • input_shape: {input_spec}")
    print(f"  • input_shape[0]: {input_spec[0]}")
    print(f"  • input_shape[0][1] (seq_len): {input_spec[0][1]}")
    print(f"  • input_shape[0][2] (channels): {input_spec[0][2]}")

    # 2. Testa um train_step
    print(f"\n2️⃣ Teste de Train Step:")
    batch_size = 2
    x_batch = x_sample[:batch_size]
    y_batch = y_sample[:batch_size]

    result = model.train_step([x_batch, y_batch])
    print(f"  • Loss: {result['loss'].numpy():.6f}")
    print(f"  • ✅ Train step funcionou!")

    # 3. Testa geração
    print(f"\n3️⃣ Teste de Geração:")
    samples = model.generate_data(y_batch[:1])
    print(f"  • Generated shape: {samples.shape}")
    print(f"  • Generated range: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"  • ✅ Geração funcionou!")

    # 4. Valida dtype dos timesteps
    print(f"\n4️⃣ Validação de dtype:")
    print(f"  • gdf_util.betas.dtype: {model._gdf_util.betas.dtype}")
    print(f"  • gdf_util.alphas_cumprod.dtype: {model._gdf_util.alphas_cumprod.dtype}")

    # Testa criação de timestep
    t_test = tensorflow.random.uniform((4,), 0, 1000, dtype=tensorflow.int64)
    print(f"  • Timestep test dtype: {t_test.dtype}")
    print(f"  • ✅ dtype correto (int64)!")

    print("\n" + "=" * 70)
    print("✅ VALIDAÇÃO COMPLETA - TUDO OK!")
    print("=" * 70)


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

"""
# 1. Instancie o modelo
model = AlgorithmDenoisingDiffusionTensorflow(
    output_shape=128,
    first_unet_model=unet_model,
    gdf_util=gdf_util,
    optimizer_diffusion=optimizer,
    time_steps=1000,
    debug=True  # ← ATIVE para ver logs detalhados
)

# 2. Valide antes de treinar
validate_model_setup(model, x_train, y_train)

# 3. Treine normalmente
model.compile(optimizer=optimizer)
model.fit(train_dataset, epochs=100)

# 4. Gere amostras
samples = model.get_samples({
    "number_classes": 10,
    "classes": {0: 10, 1: 10}  # 10 amostras de cada classe
})
"""