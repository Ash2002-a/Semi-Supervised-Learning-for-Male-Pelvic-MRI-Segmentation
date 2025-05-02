import matplotlib.pyplot as plt
import numpy as np


def eval_model(model, test_dataset):
    # Evaluate on the test set
    test_results = model.evaluate(test_dataset, verbose=1)
    print(
        f"Test Loss: {test_results[0]:.4f}, Test Accuracy: {test_results[1]:.4f}")

    from sklearn.metrics import classification_report

    labels = ["BG", "bladder", "bone", "OI", "TZ", "CG", "rectum", "SV", "NB"]

    # Predict on the test dataset
    y_true = []
    y_pred = []

    for x_batch, y_batch in test_dataset:
        # Predict the segmentation mask
        preds = unet.predict(x_batch, verbose=0)

        # Convert predictions to discrete class labels
        # (since the model output is [batch, H, W, D, num_classes])
        preds = np.argmax(preds, axis=-1)

        if y_batch.ndim == 5 and y_batch.shape[-1] == 1:
            y_batch = np.squeeze(y_batch, axis=-1)

        # Append to lists
        y_true.append(y_batch.flatten())
        y_pred.append(preds.flatten())

    # Concatenate all batches
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Generate classification report
    report = classification_report(
        y_true, y_pred, target_names=labels, zero_division=0)
    print(report)


def plot_loss(model_history, lr_history, is_GAN=False):

    if not is_GAN:
        # Retrieve validation loss history (assumes training started at epoch 1)
        val_loss = model_history.history['val_loss']

    else:
        val_loss = model_history['val_unet_loss']

    epochs = range(1, len(val_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, np.log(val_loss), label='Validation Loss',
             marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('(Log) Validation Loss')
    plt.title('Validation Loss Over Epochs')

    if is_GAN:
        gen_loss = model_history["generator_loss"]
        disc_loss = model_history["discriminator_loss"]
        plt.plot(epochs, np.log(gen_loss), label='Generator Loss',
                 marker='o', linestyle='-')
        plt.plot(epochs, np.log(disc_loss),
                 label='Discriminator Loss', marker='o', linestyle='-')

    for ep in lr_history:
        plt.axvline(ep, color='red', linestyle='--', alpha=0.7)
        plt.text(ep + 0.5, max(np.log(val_loss)) * 0.95,
                 'LR Reduced', color='red', rotation=90, va='top')

    plt.legend()
    plt.show()


def compare_truth_pred(model, volume, slice_, test_dataset):
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    # Example: 9 classes labeled 0..8
    NUM_CLASSES = 9

    labels = ["BG", "bladder", "bone", "OI", "TZ", "CG", "rectum", "SV", "NB"]

    # Define a discrete colormap with 9 distinct colors
    colors_list = [
        "#000000",  # class BG
        "#1f77b4",  # class bladder
        "#ff7f0e",  # class bone
        "#2ca02c",  # class OI
        "#d62728",  # class TZ
        "#9467bd",  # class CG
        "#8c564b",  # class rectum
        "#e377c2",  # class SV
        "#7f7f7f"   # class NB
    ]
    cmap = mcolors.ListedColormap(colors_list[:NUM_CLASSES])

    # Create legend handles for each label using a patch with corresponding color.
    legend_handles = [mpatches.Patch(color=colors_list[i], label=labels[i])
                      for i in range(NUM_CLASSES)]

    # Visualize predictions and ground truth
    x_batch, y_batch = test_dataset.skip(volume).take(1)
    # Predict the segmentation mask
    y_pred = unet.predict(x_batch)
    # Convert from probabilities to discrete class labels
    y_pred_class = np.argmax(y_pred, axis=-1)

    # Create a new figure with a specified figsize
    fig = plt.figure(figsize=(12, 6))

    # 1) Input image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Input Image")
    # Example: show slice index 8, channel 0
    ax1.imshow(x_batch[0, :, :, slice_, 0], cmap="gray")
    ax1.axis('off')

    # 2) Ground Truth
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Ground Truth")
    # If y_batch is shape (1, H, W, D, 1), squeeze out the last dimension
    if y_batch.shape[-1] == 1:
        y_batch_squeezed = np.squeeze(y_batch[0, :, :, slice_, :], axis=-1)
    else:
        y_batch_squeezed = y_batch[0, :, :, slice_, 0]
    ax2.imshow(y_batch_squeezed, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
    ax2.axis('off')

    # 3) Predicted
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Predicted")
    y_pred_slice = y_pred_class[0, :, :, slice_]
    ax3.imshow(y_pred_slice, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
    ax3.axis('off')

    # Add a legend beneath the subplots
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=NUM_CLASSES, bbox_to_anchor=(0.5, 0.1), fontsize='large')

    # Adjust the layout to make space for the legend
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
