import os
import sys
import numpy
import tensorflow
from typing import Any, Optional, Dict, Tuple
from tensorflow.keras.utils import to_categorical


class AlgorithmDenoisingDiffusionTensorflow(tensorflow.keras.Model):
    """
    DDPM (Denoising Diffusion Probabilistic Model) - VERSÃO FINAL ESTABILIZADA
    Correção: Adicionado suporte a argumentos legados e inicialização de variáveis.
    """

    def __init__(self,
                 output_shape: int,
                 first_unet_model: tensorflow.keras.Model,
                 gdf_util: Any,
                 optimizer_diffusion: tensorflow.keras.optimizers.Optimizer,
                 time_steps: int,
                 ema: float = 0.999,
                 margin: float = 1.0,  # Argumento mantido para compatibilidade
                 train_stage: str = 'all',  # Argumento mantido para compatibilidade
                 debug: bool = False,
                 second_unet_model: Optional[tensorflow.keras.Model] = None,
                 optimizer_autoencoder: Optional[tensorflow.keras.optimizers.Optimizer] = None,
                 # Corrigindo o erro de TypeError
                 **kwargs):  # Captura qualquer outro argumento extra

        super().__init__()

        self._ema = ema
        self._gdf_util = gdf_util
        self._time_steps = time_steps
        self._network = first_unet_model
        self._original_shape = output_shape
        self._optimizer_diffusion = optimizer_diffusion
        self._debug = debug

        # Trackers de métricas e Loss
        self._loss_tracker = tensorflow.keras.metrics.Mean(name="loss")
        self._loss_fn = tensorflow.keras.losses.MeanSquaredError()

        # ✅ Variáveis de estado compatíveis com Graph Mode (tf.function)
        self._training_steps = tensorflow.Variable(0, trainable=False, dtype=tensorflow.int64)
        self._ema_initialized = tensorflow.Variable(False, trainable=False, dtype=tensorflow.bool)

        # ✅ Gerenciamento do Modelo EMA
        if second_unet_model is None:
            if self._debug: print("🔧 Criando clone para modelo EMA...")
            self._second_unet_model = tensorflow.keras.models.clone_model(first_unet_model)
        else:
            self._second_unet_model = second_unet_model

        self._second_unet_model.trainable = False

        if self._debug:
            print("\n" + "=" * 50)
            print("🚀 MODELO INICIALIZADO COM SUCESSO")
            print("=" * 50)

    def _initialize_ema_weights(self):
        """Sincroniza os pesos iniciais."""
        if not self._ema_initialized:
            if self._debug: print("🔄 Sincronizando pesos EMA...")
            for w_train, w_ema in zip(self._network.trainable_weights,
                                      self._second_unet_model.trainable_weights):
                w_ema.assign(w_train)
            self._ema_initialized.assign(True)

    @tensorflow.function
    def update_ema_weights(self):
        """Atualização exponencial dos pesos."""
        for w, ew in zip(self._network.trainable_weights,
                         self._second_unet_model.trainable_weights):
            ew.assign(self._ema * ew + (1.0 - self._ema) * w)
        self._training_steps.assign_add(1)

    def _padding_input_tensor(self, input_tensor):
        """Ajusta o padding para o tamanho esperado pela UNet."""
        target_dim = self._network.input_shape[0][-2]
        current_dim = tensorflow.shape(input_tensor)[1]
        padding_needed = tensorflow.maximum(0, target_dim - current_dim)

        # Define o padding para o rank correto do tensor
        paddings = [[0, 0], [0, padding_needed], [0, 0]]
        return tensorflow.pad(input_tensor, paddings)

    @tensorflow.function
    def train_step(self, data):
        """Passo de treinamento otimizado."""
        # Suporta (x, y) ou apenas x
        if isinstance(data, (tuple, list)):
            x, y = data[0], data[1]
        else:
            x, y = data, None

        # Inicialização Lazy do EMA
        self._initialize_ema_weights()

        with tensorflow.GradientTape() as tape:
            # 1. Padding e Ruído
            x_padded = self._padding_input_tensor(x)
            batch_size = tensorflow.shape(x)[0]
            t = tensorflow.random.uniform((batch_size,), 0, self._time_steps, dtype=tensorflow.int32)
            noise = tensorflow.random.normal(tensorflow.shape(x_padded))

            # 2. Forward Diffusion (Processo de Difusão)

            x_noisy = self._gdf_util.q_sample(x_padded, t, noise)

            # 3. Predição
            predicted_noise = self._network([x_noisy, t, y], training=True)

            if len(predicted_noise.shape) > len(noise.shape):
                predicted_noise = tensorflow.squeeze(predicted_noise, axis=-1)

            # 4. Cálculo de Perda
            loss = self._loss_fn(noise, predicted_noise)

        # 5. Otimização
        grads = tape.gradient(loss, self._network.trainable_weights)
        self._optimizer_diffusion.apply_gradients(zip(grads, self._network.trainable_weights))

        # 6. Update EMA
        self.update_ema_weights()

        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}

    def generate_data(self, labels, batch_size: int = 32):
        """Gera dados usando o processo de difusão reversa com o modelo EMA."""
        if not isinstance(labels, tensorflow.Tensor):
            labels = tensorflow.convert_to_tensor(labels, dtype=tensorflow.float32)

        target_dim = self._network.input_shape[0][-2]
        channels = self._network.input_shape[0][-1]

        # Começa com ruído puro
        img = tensorflow.random.normal((tensorflow.shape(labels)[0], target_dim, channels))

        for i in reversed(range(0, self._time_steps)):
            t = tensorflow.fill((tensorflow.shape(labels)[0],), i)
            # Predição com o modelo EMA (mais estável)
            pred_noise = self._second_unet_model([img, t, labels], training=False)

            img = self._gdf_util.p_sample(
                pred_noise, img, t, clip_denoised=True
            )

        # Corta para o tamanho original (removendo padding)
        return img[:, :self._original_shape, :].numpy()

    def get_samples(self, number_samples_per_class: Dict):
        """Interface amigável para gerar amostras de múltiplas classes."""
        generated_data = {}
        num_classes = number_samples_per_class["number_classes"]

        for label, count in number_samples_per_class["classes"].items():
            if self._debug: print(f"🎨 Gerando {count} amostras para classe {label}...")
            one_hot = to_categorical([label] * count, num_classes=num_classes)
            samples = self.generate_data(one_hot)
            generated_data[label] = numpy.squeeze(samples)

        return generated_data