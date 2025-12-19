
import os
import numpy as np
from PIL import Image
os.environ["ML_FRAMEWORK"] = "tensorflow"

from Engine.architectures.adversarial.AdversarialModel import AdversarialModel
from Engine.models.Adversarial import Adversarial

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Concatenate, Conv2D, Reshape, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

IMAGE_SIZE = (64, 64)
INPUT_SHAPE = (64, 64, 3)
DATASET_DIR = "./50k"
MAX_SAMPLES = 4200
N_CLASSES = 10
BATCH_LIMIT = 4200
LATENT_DIMENSION = 128

OUTPUT_DIR = "./output_mnist"
os.makedirs(OUTPUT_DIR, exist_ok=True)

x_real_samples, y_real_samples = [], []

image_files = sorted(os.listdir(DATASET_DIR))

for img_name in image_files:
    img_path = os.path.join(DATASET_DIR, img_name)

    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)

        img = np.asarray(img, dtype=np.float32) / 255.0
        label = np.random.randint(0, N_CLASSES)

        x_real_samples.append(img)
        y_real_samples.append(label)

    except Exception:
        continue

    if len(x_real_samples) >= MAX_SAMPLES:
        break


x_real_samples = np.array(x_real_samples, dtype=np.float32)
y_real_samples = np.array(y_real_samples, dtype=np.int32)

class MyClass(AdversarialModel):

    @staticmethod
    def get_discriminator():
        image_shape = (64, 64, 3)
        number_classes = 10
        last_layer_activation = "sigmoid"
        dense_layer_sizes = (256, 128)
        dropout_rate = 0.3
        initializer_mean = 0.0
        initializer_deviation = 0.02
        dataset_type = None
        """
        Build and return a 2D convolutional discriminator model.

        Input format: image_shape (64x64x3)
        """

        if dataset_type is None:
            import numpy as np
            dataset_type = np.float32

        initialization = RandomNormal(mean=initializer_mean, stddev=initializer_deviation)
        image_input = Input(shape=image_shape, dtype=dataset_type, name="image_input")
        label_input = Input(shape=(number_classes,), dtype=dataset_type, name="label_input")
        label_embedding = Dense(image_shape[0] * image_shape[1], kernel_initializer=initialization)(label_input)
        label_embedding = LeakyReLU(alpha=0.2)(label_embedding)
        label_embedding = Reshape((image_shape[0], image_shape[1], 1))(label_embedding)

        x = Concatenate(axis=-1)([image_input, label_embedding])

        x = Conv2D(
            32,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=initialization
        )(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=initialization
        )(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(
            128,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=initialization
        )(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(
            256,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=initialization
        )(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate)(x)

        # ------------------------------------------------------------------
        # Classification head
        # ------------------------------------------------------------------
        x = Flatten()(x)

        for units in dense_layer_sizes:
            x = Dense(units, kernel_initializer=initialization)(x)
            x = Dropout(dropout_rate)(x)
            x = LeakyReLU(alpha=0.2)(x)

        x = Dense(1, kernel_initializer=initialization)(x)

        if last_layer_activation == "sigmoid":
            from tensorflow.keras.activations import sigmoid
            validity = sigmoid(x)
        else:
            validity = x

        # ------------------------------------------------------------------
        # Final model
        # ------------------------------------------------------------------
        model = Model(
            inputs=[image_input, label_input],
            outputs=validity,
            name="Discriminator_64x64x3"
        )

        return model

    @staticmethod
    def get_generator():
        """
        Builds and returns a 2D convolutional generator model.

        Output format: 64x64x3
        """
        latent_dimension =  LATENT_DIMENSION
        number_classes =  10
        activation_function="relu"
        last_layer_activation="tanh"
        initializer_mean=0.0
        initializer_deviation=0.02
        dataset_type=None

        from tensorflow.keras.layers import (
            Input, Dense, Concatenate, Reshape,
            Conv2DTranspose, BatchNormalization
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.initializers import RandomNormal
        from tensorflow.keras.layers import ReLU, LeakyReLU
        import numpy as np

        if dataset_type is None:
            dataset_type = np.float32

        initialization = RandomNormal(
            mean=initializer_mean,
            stddev=initializer_deviation
        )

        # ------------------------------------------------------------------
        # Inputs
        # ------------------------------------------------------------------
        latent_input = Input(
            shape=(latent_dimension,),
            dtype=dataset_type,
            name="latent_input"
        )

        label_input = Input(
            shape=(number_classes,),
            dtype=dataset_type,
            name="label_input"
        )

        # ------------------------------------------------------------------
        # Conditional latent embedding
        # ------------------------------------------------------------------
        x = Concatenate()([latent_input, label_input])

        x = Dense(
            4 * 4 * 256,
            kernel_initializer=initialization
        )(x)

        if activation_function == "leaky_relu":
            x = LeakyReLU(alpha=0.2)(x)
        else:
            x = ReLU()(x)

        x = Reshape((4, 4, 256))(x)

        # ------------------------------------------------------------------
        # Upsampling blocks: 4 → 8 → 16 → 32 → 64
        # ------------------------------------------------------------------
        x = Conv2DTranspose(
            128, kernel_size=4, strides=2, padding="same",
            kernel_initializer=initialization
        )(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2DTranspose(
            64, kernel_size=4, strides=2, padding="same",
            kernel_initializer=initialization
        )(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2DTranspose(
            32, kernel_size=4, strides=2, padding="same",
            kernel_initializer=initialization
        )(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2DTranspose(
            16, kernel_size=4, strides=2, padding="same",
            kernel_initializer=initialization
        )(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # ------------------------------------------------------------------
        # Output layer (64x64x3)
        # ------------------------------------------------------------------
        generator_output = Conv2DTranspose(
            3,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=initialization
        )(x)

        if last_layer_activation == "tanh":
            from tensorflow.keras.activations import tanh
            generator_output = tanh(generator_output)
        elif last_layer_activation == "sigmoid":
            from tensorflow.keras.activations import sigmoid
            generator_output = sigmoid(generator_output)

        generator_model = Model(
            inputs=[latent_input, label_input],
            outputs=generator_output,
            name="Generator_64x64x3"
        )

        return generator_model


number_samples_per_class = {
    "number_classes": N_CLASSES,
    "classes": {i: 8 for i in range(N_CLASSES)}
}

# =====================
# Criar Modelo com Configuração Correta
# =====================

models = {
    "adversarial": Adversarial(
        number_classes=N_CLASSES,
        latent_dimension=LATENT_DIMENSION,  # CRÍTICO: Mesmo valor!
        model=MyClass(),
        number_samples_per_class=number_samples_per_class
    ),
}


try:
    import matplotlib.pyplot as plt

    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    print("⚠️ Matplotlib não disponível - pulando visualização")

for model_name, model in models.items():
    print(f"\n{'=' * 60}")
    print(f"🔧 Modelo: {model_name}")
    print(f"{'=' * 60}")

    # Treinamento
    print(f"⏳ Treinando {model_name}...")
    model.fit_model(
        input_shape=INPUT_SHAPE,
        x_real_samples=x_real_samples,
        y_real_samples=y_real_samples,
        flatten=True
    )
    print(f"✓ Treinamento concluído")

    # Geração
    print(f"🎨 Gerando amostras sintéticas...")
    synthetic_samples = model.get_samples(number_samples_per_class)
    print(f"✓ Geradas {synthetic_samples.shape[0]} amostras")

    # Visualização e Salvamento
    if matplotlib_available:
        n = min(36, synthetic_samples.shape[0])
        cols = 6
        rows = n // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        fig.suptitle(f"MNIST Sintético - {model_name}", fontsize=16, fontweight='bold')

        idx = 0
        for i in range(rows):
            for j in range(cols):
                axes[i, j].imshow(
                    synthetic_samples[idx].squeeze(),
                    cmap="gray"
                )
                axes[i, j].axis("off")
                idx += 1

        plt.tight_layout()

        # Salvar imagem
        output_path = os.path.join(OUTPUT_DIR, f"{model_name}_samples.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"💾 Imagem salva: {output_path}")

    # Salvar amostras como arrays NumPy
    samples_path = os.path.join(OUTPUT_DIR, f"{model_name}_samples.npy")
    np.save(samples_path, synthetic_samples)
    print(f"💾 Arrays salvos: {samples_path}")

print(f"\n{'=' * 60}")
print(f"✅ Processamento completo!")
print(f"📁 Resultados salvos em: {OUTPUT_DIR}")
print(f"{'=' * 60}")