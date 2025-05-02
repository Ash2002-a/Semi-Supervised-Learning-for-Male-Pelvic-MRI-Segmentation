import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import LambdaCallback
from data import get_labeled_dataset, get_unlabeled_dataset

#Define training function for unet
def train_unet(model, imgs_lab_train, masks_lab_train,
               val_dataset, epochs=50, train_bs=6, min_grad=1e-7,
               stop_patience=10, reducLR_patience=5):
    labeled_dataset = (get_labeled_dataset(
        imgs_lab_train, masks_lab_train, train_bs).repeat())

    # EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=stop_patience,
        restore_best_weights=True,
        verbose=1
    )

    num_train_batches = len(imgs_lab_train) // train_bs

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=reducLR_patience,
        verbose=1,
        min_lr=min_grad  # optional
    )

    # Log when the LR is reduced
    lr_history = []
    prev_lr = [model.optimizer.learning_rate.numpy()]

    lr_logger = LambdaCallback(
        on_epoch_end=lambda epoch, logs: (
            lr_history.append(
                epoch + 1) if model.optimizer.learning_rate.numpy() < prev_lr[0] else None,
            prev_lr.__setitem__(0, model.optimizer.learning_rate.numpy())
        )
    )

    # Fit the model
    history = model.fit(
        labeled_dataset,
        validation_data=val_dataset,
        steps_per_epoch=num_train_batches,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stopping, reduce_lr, lr_logger]
    )

    return history, lr_history

#define training subroutines for unet, discriminator and generator in GAN.
#Manually computes and updates gradients
@tf.function
def train_unet_step(unet, x_labeled, y_labeled, unet_optimizer, seg_loss_fn):
    with tf.GradientTape() as tape:
        y_pred = unet(x_labeled, training=True)
        loss = tf.reduce_mean(seg_loss_fn(y_labeled, y_pred))
    gradients = tape.gradient(loss, unet.trainable_weights)
    unet_optimizer.apply_gradients(zip(gradients, unet.trainable_weights))
    return loss


@tf.function
def train_discriminator_step(discriminator, generator, latent_dim, disc_optimizer, x_unlabeled):
    # Generate fake images from the generator.
    latent_vectors = tf.random.normal(
        shape=(tf.shape(x_unlabeled)[0], latent_dim))
    fake_images = generator(latent_vectors, training=True)

    with tf.GradientTape() as tape:
        # Feed raw images to discriminator
        d_real_preds = discriminator(x_unlabeled, training=True)
        d_fake_preds = discriminator(fake_images, training=True)

        # Label smoothing for real images
        real_labels = tf.ones_like(d_real_preds) * 0.9
        fake_labels = tf.zeros_like(d_fake_preds)

        combined_preds = tf.concat([d_real_preds, d_fake_preds], axis=0)
        combined_labels = tf.concat([real_labels, fake_labels], axis=0)

        d_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                combined_labels, combined_preds, from_logits=False)
        )

    gradients = tape.gradient(d_loss, discriminator.trainable_weights)
    disc_optimizer.apply_gradients(
        zip(gradients, discriminator.trainable_weights))
    return d_loss


@tf.function
def train_generator_step(generator, discriminator, latent_dim, gen_optimizer, batch_size):
    latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    with tf.GradientTape() as tape:
        fake_images = generator(latent_vectors, training=True)
        d_fake_preds = discriminator(fake_images, training=False)
        g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
            tf.ones_like(d_fake_preds), d_fake_preds, from_logits=False))
    gradients = tape.gradient(g_loss, generator.trainable_weights)
    gen_optimizer.apply_gradients(zip(gradients, generator.trainable_weights))
    return g_loss

#Training function for GAN. Cycles unet trainig, discriminator training, and generator training
#batch-wise in each epoch
def train_semi_supervised_Gan(unet_Gan, discriminator_Gan, generator_Gan, labeled_imgs, labeled_masks,
                              unlabeled_imgs, validation_dataset, epochs, unet_loss,
                              latent_dim=100, lab_batch_size=6, unlab_batch_size=3,
                              patience=20, lr_patience=10, pre_train_epochs=0,
                              unetLR=0.001, discLR=0.000125, genLR=0.00025):
    history = {
        "unet_loss": [],
        "discriminator_loss": [],
        "generator_loss": [],
        "val_unet_loss": [],
        "lr_history": []
    }

    num_batches = len(labeled_imgs) // lab_batch_size

    best_val_loss = float("inf")
    best_unet_weights = None
    wait = 0      # for early stopping
    lr_pat = lr_patience
    lr_wait = 0   # for LR reduction

    # Define optimizers and loss function
    unet_optimizer = Adam(learning_rate=unetLR)
    disc_optimizer = Adam(learning_rate=discLR, clipnorm=1.0)
    gen_optimizer = Adam(learning_rate=genLR)
    seg_loss_fn = unet_loss

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Create iterators for labeled and unlabeled data
        labeled_dataset_iter = iter(
            get_labeled_dataset(labeled_imgs, labeled_masks,
                                lab_batch_size, epoch)
            .repeat()
            .take(num_batches)
        )
        unlabeled_dataset_iter = iter(
            get_unlabeled_dataset(unlabeled_imgs, unlab_batch_size, epoch)
            .repeat()
            .take(num_batches)
        )

        epoch_unet_loss = 0.0
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        for batch in range(num_batches):
            print(f"  Batch {batch+1}/{num_batches}")

            # 1. Train U-Net on labeled data
            x_labeled, y_labeled = next(labeled_dataset_iter)
            loss_unet = train_unet_step(
                unet_Gan, x_labeled, y_labeled, unet_optimizer, seg_loss_fn)
            epoch_unet_loss += loss_unet

            # 2. If past pre-training epochs, update discriminator & generator using unlabeled data
            if epoch >= pre_train_epochs:
                x_unlabeled = next(unlabeled_dataset_iter)
                loss_disc = train_discriminator_step(
                    discriminator_Gan, generator_Gan, latent_dim, disc_optimizer, x_unlabeled)
                epoch_d_loss += loss_disc

                loss_gen = train_generator_step(generator_Gan, discriminator_Gan,
                                                latent_dim, gen_optimizer, unlab_batch_size)
                epoch_g_loss += loss_gen

        # Average losses over the batches
        avg_unet_loss = epoch_unet_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches if epoch >= pre_train_epochs else 0.0
        avg_g_loss = epoch_g_loss / num_batches if epoch >= pre_train_epochs else 0.0

        # Evaluate validation loss using unet_newGan (segmentation performance)
        val_results = unet_Gan.evaluate(validation_dataset, verbose=1)
        avg_val_unet_loss = val_results[0]

        # Log losses
        history["unet_loss"].append(avg_unet_loss.numpy())
        history["discriminator_loss"].append(
            avg_d_loss.numpy() if isinstance(avg_d_loss, tf.Tensor) else avg_d_loss)
        history["generator_loss"].append(
            avg_g_loss.numpy() if isinstance(avg_g_loss, tf.Tensor) else avg_g_loss)
        history["val_unet_loss"].append(avg_val_unet_loss)

        print(f"Epoch {epoch+1} Summary:")
        print(f"  - U-Net Loss: {avg_unet_loss:.4f}")
        print(f"  - Discriminator Loss: {avg_d_loss:.4f}")
        print(f"  - Generator Loss: {avg_g_loss:.4f}")
        print(f"  - Validation U-Net Loss: {avg_val_unet_loss:.4f}")

        # Early stopping and LR reduction logic based on validation loss
        if avg_val_unet_loss < best_val_loss:
            best_val_loss = avg_val_unet_loss
            best_unet_weights = unet_Gan.get_weights()
            if lr_wait > 4:
                lr_patience -= 1
            wait = 0
            lr_wait = 0
        else:
            wait += 1
            lr_wait += 1
            old_lr = unet_optimizer.learning_rate.numpy()
            if lr_wait >= lr_patience and old_lr > 2 * 1e-7:
                history["lr_history"].append(epoch)
                new_lr = old_lr * 0.5
                unet_optimizer.learning_rate.assign(new_lr)
                print(
                    f"\n*** Reducing U-Net LR from {old_lr} to {new_lr} ***\n")
                lr_wait = 0
            if wait >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

        gc.collect()

        #Restore best weights based on validation dataset
    if best_unet_weights is not None:
        unet_Gan.set_weights(best_unet_weights)
        print("Restored best U-Net weights from lowest validation loss epoch.")

    return history
