from data import load_data, create_static_set, add_channel
from sklearn.model_selection import train_test_split
from models import downsampling_path, upsampling_path, \
    weighted_categorical_crossentropy, focal_loss, \
    discriminator, generator
import numpy as np
from train import train_unet, train_semi_supervised_Gan
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.optimizers import Adam
import gc


random_seed = 42

# Load data as lists of numpy arrays
# Each array has size 224x224xK
# where K is the z-dimension of the individual volume
imgs, masks = load_data()

# Split data into labeled and unlabeled sets
imgs_unlab, imgs_lab, masks_unlab, masks_lab = train_test_split(
    imgs, masks, test_size=0.3, random_state=random_seed)

# Split labeled data into training, validation, and test sets
imgs_lab_train, imgs_lab_test, masks_lab_train, masks_lab_test = train_test_split(
    imgs_lab, masks_lab, test_size=0.2, random_state=random_seed)

imgs_lab_train, imgs_lab_val, masks_lab_train, masks_lab_val = train_test_split(
    imgs_lab_train, masks_lab_train, test_size=0.15, random_state=random_seed)

# Add channel dimension to all sets
imgs_lab_train = add_channel(imgs_lab_train)
masks_lab_train = add_channel(masks_lab_train)
imgs_lab_val = add_channel(imgs_lab_val)
masks_lab_val = add_channel(masks_lab_val)
imgs_lab_test = add_channel(imgs_lab_test)
masks_lab_test = add_channel(masks_lab_test)
imgs_unlab = add_channel(imgs_unlab)

print(len(imgs_unlab))
print(len(imgs_lab_train))
print(len(imgs_lab_val))
print(len(imgs_lab_test))

# Generate static test and validation sets
test_dataset = create_static_set(
    imgs_lab_test, masks_lab_test, seed=35, window_size=16)
val_dataset = create_static_set(
    imgs_lab_val, masks_lab_val, seed=36, window_size=16)

for x, y in val_dataset.take(1):
    print(x.shape, y.shape)

# Get the frequencies in the labelled training set:
all_labels = np.concatenate([volume.flatten()
                             for volume in masks_lab_train]).astype(int)
class_counts = np.bincount(all_labels, minlength=9)
del all_labels
gc.collect()
total_labels = np.sum(class_counts)
class_fractions = class_counts / total_labels


inverse_class_weights = 1 / class_fractions
inverse_class_weights /= np.sum(inverse_class_weights)

# Define the weightsd categorical crossentropy
loss_fn = weighted_categorical_crossentropy(inverse_class_weights)

# Now define the architecture for the unet model
input_shape = (224, 224, 16, 1)
n_classes = 9

downsampling = downsampling_path(input_shape)
bottleneck, skip_connections = downsampling.output
upsampling = upsampling_path(skip_connections, bottleneck, n_classes=n_classes)
inputs = downsampling.input
outputs = upsampling([bottleneck] + skip_connections)
unet = Model(inputs=inputs, outputs=outputs, name="3D_U-Net")
lr = 0.001
unet.compile(optimizer=Adam(learning_rate=lr),
             loss=loss_fn,
             metrics=['accuracy'])

# Define the data on which to train
train_fraction = 0.6
split_idx = int(train_fraction * len(imgs_lab_train))
imgs_lab_train_reduced = imgs_lab_train[:split_idx]
masks_lab_train_reduced = masks_lab_train[:split_idx]

# training batch size
train_bs = 6

history, lr_hist = train_unet(unet, imgs_lab_train_reduced, masks_lab_train_reduced,
                              val_dataset, epochs=50, train_bs=train_bs, min_grad=1e-7,
                              stop_patience=10, reducLR_patience=5)

# evaluate model and get precision report
eval_model(unet, test_dataset)
# plot the loss graph
plot_loss(history, lr_hist, is_GAN=False)
# plot the image, true mask, and prediction for an image/volume in test_dataset
compare_truth_pred(model=unet, volume=0, slice_=8, test_dataset=test_dataset)

'''
# Define the fraction of unlabelled images to use
unlab_fraction = 0.6
split_idx = int(unlab_fraction * len(imgs_unlab))
imgs_unlab_reduced = imgs_unlab[:split_idx]


# Build the GAN
# Define the discriminator with the U-Net as the first part
disc_input = Input(shape=input_shape)
segmentation_output = unet(disc_input)  # Use the pre-trained U-Net
# Collapse so that output from unet can go into downsampling for discriminator
collapsed_output = layers.Conv3D(
    1, kernel_size=1, activation='sigmoid', name="collapse_conv")(segmentation_output)

# Add the additional discriminator layers
discriminator_branch_Gan = discriminator(downsampling)
disc_output = discriminator_branch_Gan(collapsed_output)
discriminator_GAN = Model(
    inputs=disc_input, outputs=disc_output, name="Discriminator_GAN")

# Define the generator which takes in latent vector and outputs image
latent_dim = 256
generator_GAN = generator(latent_dim=latent_dim, output_shape=input_shape)

# ulabelled images batch size
unlab_bs = 3

# Train the GAN using the existing unet weights
#NOTE HERE THAT LOSS FOR UNET IS CHANGED TO A FOCAL LOSS
history = train_semi_supervised_Gan(unet, discriminator_GAN, generator_GAN, imgs_lab_train_reduced,
                                    masks_lab_train_reduced, imgs_unlab_reduced,
                                    val_dataset, epochs=50, unet_loss=focal_loss(),
                                    latent_dim=latent_dim, lab_batch_size=train_bs, unlab_batch_size=unlab_bs,
                                    patience=20, lr_patience=10, pre_train_epochs=0,
                                    unetLR=0.001, discLR=0.000125, genLR=0.00025)

# evaluate model and get precision report
eval_model(unet, test_dataset)
# plot the loss graph
plot_loss(history, lr_hist, is_GAN=True)
# plot the image, true mask, and prediction for an image/volume in test_dataset
compare_truth_pred(model=unet, volume=0, slice_=8, test_dataset=test_dataset)
'''
