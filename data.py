import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import tensorflow as tf
import gc


def load_data():
    """
    Loads NIfTI images and masks from the 'data' folder
    Returns:
       imgs: list of 3D volumes (numpy arrays)
       masks: list of corresponding 3D mask volumes (numpy arrays)
    """
    drive_path = 'data'

    # List all files in the data directory
    all_files = [os.path.join(drive_path, f) for f in os.listdir(drive_path)]

    # Create lists to store image and mask file paths.
    img_paths = []
    mask_paths = []

    # Iterate through all files and categorize them based on filename.
    for filename in all_files:
        if filename.endswith('_img.nii'):
            # Construct the corresponding mask filename.
            mask_name = filename.replace('_img.nii', '') + '_mask.nii'
            img_paths.append(filename)
            mask_paths.append(mask_name)

    # Load the files into lists.
    imgs = [nib.load(p).get_fdata() for p in img_paths]
    masks = [nib.load(p).get_fdata() for p in mask_paths]

    print(
        f"Loaded {len(imgs)} Images and Masks and converted to Numpy Arrays \n Reshaping Slices...")

    # Rescale volumes: resize the x-y dimensions to 224x224 and normalize each slice.
    new_imgs = []
    new_masks = []
    for i in range(len(imgs)):
        img = imgs[i]
        mask = masks[i]
        x_dI, x_dM = img.shape[0], mask.shape[0]
        y_dI, y_dM = img.shape[1], mask.shape[1]
        z_dI, z_dM = img.shape[2], mask.shape[2]

        # Check that image and mask dimensions match and that there are enough slices (require > 16)
        if (x_dI == x_dM) and (x_dI == y_dI) and (x_dM == y_dM) and (z_dI == z_dM) and (z_dI > 15):
            # Resize each slice using SciPy's zoom.
            img_resized = np.stack([zoom(img[:, :, z], (224 / x_dI, 224 / y_dI), order=3)
                                    for z in range(z_dI)], axis=2)
            mask_resized = np.stack([zoom(mask[:, :, z], (224 / x_dI, 224 / y_dI), order=0)
                                     for z in range(z_dI)], axis=2)

            # Normalize each slice of the image to have values in [0,1].
            for z in range(img_resized.shape[2]):
                max_value = img_resized[:, :, z].max()
                if max_value > 0:
                    img_resized[:, :, z] = img_resized[:, :, z] / max_value

            new_imgs.append(img_resized)
            new_masks.append(mask_resized)

    # Replace the old lists with the new processed ones.
    imgs, masks = new_imgs, new_masks
    del new_imgs, new_masks
    gc.collect()

    print(f"Reshaped and Kept {len(imgs)} images.")
    return imgs, masks

# Generate labelled data tensorflow datasets such that a random set of 16
# slices can be sampled from each volume for each epoch during training


def labeled_data_gen(imgs, masks, window_size=16, seed=None):
    num_volumes = len(imgs)
    rng = np.random.default_rng(seed)
    while True:
        # Shuffle the order of volumes (one epoch)
        volume_indices = np.random.permutation(num_volumes)

        for i in volume_indices:
            img = imgs[i]
            mask = masks[i]
            H, W, Z, C = img.shape

            # Compute max possible start index
            max_start = Z - window_size
            # Pick a random start
            start_z = rng.integers(0, max_start + 1)
            sub_img = img[:, :, start_z:start_z + window_size, :]
            sub_mask = mask[:, :, start_z:start_z + window_size, :]

            yield tf.convert_to_tensor(sub_img, dtype=tf.float32), tf.convert_to_tensor(sub_mask, dtype=tf.int32)

# Same for unlabelled data
def unlabeled_data_gen(imgs, window_size=16, seed=None):
    rng = np.random.default_rng(seed)
    num_volumes = len(imgs)
    while True:
        # Shuffle the order of volumes (one epoch)
        volume_indices = np.random.permutation(num_volumes)

        for i in volume_indices:
            img = imgs[i]
            H, W, Z, C = img.shape

            # Compute max possible start index
            max_start = Z - window_size
            # Pick a random start
            start_z = rng.integers(0, max_start + 1)
            sub_img = img[:, :, start_z:start_z + window_size, :]

            yield tf.convert_to_tensor(sub_img, dtype=tf.float32)

# Getter functions to create a datasets for training.
# This function dynamically generates random sets of 16 slices (window_size)
# from each labeled image and its corresponding mask during an epoch.
# The labeled_data_gen function is used to create the dataset on the fly.


def get_labeled_dataset(imgs, masks, batch_size, window_size=16, seed=None):
    dataset = tf.data.Dataset.from_generator(
        lambda: labeled_data_gen(imgs, masks, window_size, seed),
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 16, 1), dtype=tf.float32),  # Image
            tf.TensorSpec(shape=(224, 224, 16, 1), dtype=tf.int32)     # Mask
        )
    )
    return dataset.shuffle(buffer_size=50).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_unlabeled_dataset(imgs, batch_size, window_size=16, seed=None):
    dataset = tf.data.Dataset.from_generator(
        lambda: unlabeled_data_gen(imgs, window_size, seed),
        output_signature=tf.TensorSpec(
            shape=(224, 224, 16, 1), dtype=tf.float32)
    )
    return dataset.shuffle(buffer_size=50).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# function to create a random dataset which is fixed
def create_static_set(imgs, masks, seed=35, window_size=16):
    # Pre-generate all slices for the test dataset
    static_vols = []
    static_masks = []

    for img, mask in zip(imgs, masks):
        H, W, Z, C = img.shape
        max_start = Z - window_size

        rng = np.random.default_rng(seed)
        start_z = rng.integers(0, max_start + 1)

        # Generate slices for each test volume
        sub_img = img[:, :, start_z:start_z + window_size, :]
        sub_mask = mask[:, :, start_z:start_z + window_size, :]
        static_vols.append(sub_img)
        static_masks.append(sub_mask)

    # Convert to tensors
    static_vols = tf.convert_to_tensor(np.array(static_vols), dtype=tf.float32)
    static_masks = tf.convert_to_tensor(np.array(static_masks), dtype=tf.int32)

    # Create a static dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (static_vols, static_masks)).batch(1).prefetch(tf.data.AUTOTUNE)

    return dataset


def add_channel(volumes):
    return [volume[..., np.newaxis] for volume in volumes]
