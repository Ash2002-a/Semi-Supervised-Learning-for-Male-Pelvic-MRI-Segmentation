import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model

#Downsampling path for unet
def downsampling_path(input_shape=(224, 224, 16, 1)):
    inputs = Input(shape=input_shape)

    # Downsampling Path
    c1 = layers.Conv3D(32, (3, 3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv3D(32, (3, 3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling3D((2, 2, 2))(c1)

    c2 = layers.Conv3D(64, (3, 3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv3D(64, (3, 3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling3D((2, 2, 2))(c2)

    c3 = layers.Conv3D(128, (3, 3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv3D(128, (3, 3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling3D((2, 2, 2))(c3)

    c4 = layers.Conv3D(256, (3, 3, 2), activation='relu',
                       padding='same', kernel_initializer='he_normal')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv3D(256, (3, 3, 2), activation='relu',
                       padding='same', kernel_initializer='he_normal')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling3D((2, 2, 2))(c4)

    c5 = layers.Conv3D(512, (3, 3, 1), activation='relu',
                       padding='same', kernel_initializer='he_normal')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv3D(512, (3, 3, 1), activation='relu',
                       padding='same', kernel_initializer='he_normal')(c5)
    c5 = layers.BatchNormalization()(c5)
    p5 = layers.MaxPooling3D((2, 2, 1))(c5)

    # Bottleneck
    b = layers.Conv3D(1024, (3, 3, 1), activation='relu',
                      padding='same', kernel_initializer='he_normal')(p5)
    b = layers.BatchNormalization()(b)
    b = layers.Conv3D(1024, (3, 3, 1), activation='relu',
                      padding='same', kernel_initializer='he_normal')(b)
    b = layers.BatchNormalization()(b)

    return Model(inputs, [b, [c1, c2, c3, c4, c5]], name="DownsamplingPath")

#upsampling path for unet
def upsampling_path(skip_connections, bottleneck, n_classes=9):
    # Unpack skip connections
    c1, c2, c3, c4, c5 = skip_connections

    # Upsampling Path
    u5 = layers.Conv3DTranspose(512, (2, 2, 1), strides=(
        2, 2, 1), padding='same')(bottleneck)
    u5 = layers.concatenate([u5, c5], axis=-1)
    c5u = layers.Conv3D(512, (3, 3, 1), activation='relu',
                        padding='same', kernel_initializer='he_normal')(u5)
    c5u = layers.BatchNormalization()(c5u)
    c5u = layers.Conv3D(512, (3, 3, 1), activation='relu',
                        padding='same', kernel_initializer='he_normal')(c5u)
    c5u = layers.BatchNormalization()(c5u)

    u4 = layers.Conv3DTranspose(
        256, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5u)
    u4 = layers.concatenate([u4, c4], axis=-1)
    c6 = layers.Conv3D(256, (3, 3, 2), activation='relu',
                       padding='same', kernel_initializer='he_normal')(u4)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv3D(256, (3, 3, 2), activation='relu',
                       padding='same', kernel_initializer='he_normal')(c6)
    c6 = layers.BatchNormalization()(c6)

    u3 = layers.Conv3DTranspose(
        128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u3 = layers.concatenate([u3, c3], axis=-1)
    c7 = layers.Conv3D(128, (3, 3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(u3)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv3D(128, (3, 3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(c7)
    c7 = layers.BatchNormalization()(c7)

    u2 = layers.Conv3DTranspose(
        64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u2 = layers.concatenate([u2, c2], axis=-1)
    c8 = layers.Conv3D(64, (3, 3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(u2)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv3D(64, (3, 3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(c8)
    c8 = layers.BatchNormalization()(c8)

    u1 = layers.Conv3DTranspose(
        32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u1 = layers.concatenate([u1, c1], axis=-1)
    c9 = layers.Conv3D(32, (3, 3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(u1)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv3D(32, (3, 3, 3), activation='relu',
                       padding='same', kernel_initializer='he_normal')(c9)
    c9 = layers.BatchNormalization()(c9)

    # Output layer
    outputs = layers.Conv3D(n_classes, (1, 1, 1), activation='softmax')(c9)

    return Model([bottleneck] + skip_connections, outputs, name="UpsamplingPath")

#generator upsampling path
def generator(latent_dim=100, output_shape=(224, 224, 16, 1)):
    inputs = Input(shape=(latent_dim,))

    # Fully connected layer to reshape the latent vector
    x = layers.Dense(7 * 7 * 512, activation='relu')(inputs)
    # Starting small spatial dimensions (7x7x1)
    x = layers.Reshape((7, 7, 1, 512))(x)

    # Transposed convolutional layers to upsample
    x = layers.Conv3DTranspose(256, (4, 4, 2), strides=(
        2, 2, 1), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)  # -> 14 x 14 x 1 x 256

    x = layers.Conv3DTranspose(128, (4, 4, 2), strides=(
        2, 2, 2), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)  # -> 28 x 28 x 2 x 128

    x = layers.Conv3DTranspose(64, (4, 4, 2), strides=(
        2, 2, 2), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)  # -> 56 x 56 x 4 x 64

    x = layers.Conv3DTranspose(32, (4, 4, 2), strides=(
        2, 2, 2), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)  # -> 112 x 112 x 8 x 32

    # Final layer to generate the desired output volume
    x = layers.Conv3DTranspose(output_shape[-1], (4, 4, 2), strides=(
        2, 2, 2), padding='same', activation='sigmoid', kernel_initializer='he_normal')(x)
    # -> 224 x 224 x 16 x output_shape[-1]

    return Model(inputs, x, name="Generator")

#discriminator function
def discriminator(shared_downsampling):
    bottleneck_input = shared_downsampling.input
    bottleneck, _ = shared_downsampling(bottleneck_input)

    # Just one small conv block
    x = layers.Conv3D(8, (3, 3, 3), activation='relu',
                      padding='same', kernel_initializer='he_normal')(bottleneck)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Global pooling + small dense
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(4, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.4)(x)

    # Output
    adversarial_output = layers.Dense(
        1, activation='sigmoid', name="real_or_fake")(x)

    return tf.keras.Model(inputs=bottleneck_input, outputs=adversarial_output, name="Discriminator")

#weighted cross-entropy loss function
def weighted_categorical_crossentropy(inv_class_weights):
    # Convert the class weights into a tensor.
    class_weights = tf.constant(inv_class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # Squeeze last dimension to remove unnecessary channel.
        y_true = tf.squeeze(y_true, axis=-1)
        # Convert y_true to one-hot encoding with depth 9.
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=9)
        # Clip predictions to avoid numerical instabilities
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # Compute the weighted categorical cross-entropy loss:
        # For each pixel: loss = -sum( class_weights[c] * y_true[c] * log(y_pred[c]) )
        loss = -tf.reduce_sum(y_true * class_weights *
                              tf.math.log(y_pred), axis=-1)
        return loss
    return loss

#focal loss function
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        # Squeeze last dimension to remove the unnecessary channel (e.g., from (batch, 224, 224, 1) to (batch, 224, 224))
        y_true = tf.squeeze(y_true, axis=-1)
        # Convert to one-hot encoding
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=9)
        # Clip predictions to avoid numerical instability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
        # Compute focal loss
        loss = -y_true * alpha * \
            tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
        # Reduce the loss across all classes for each pixel
        return tf.reduce_sum(loss, axis=-1)

    return loss
