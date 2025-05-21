import random
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

# --- Configuration ---
DATA_DIR = 'data/processed'
OUTPUT_IMAGE = 'slice_visualization.png'
MODALITIES = ['FLAIR', 'T1', 'T1ce', 'T2']

# --- Functions ---

# Select a random .npz file from the directory
def choose_random_case(data_dir):
    cases = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    return os.path.join(data_dir, random.choice(cases))

# Load images and labels from a .npz file
def load_npz(path):
    data = np.load(path)
    # images: (Z, C, H, W), labels: (Z, C, H, W)
    return data['images'], data['labels']

# Return a random slice (image, mask) and its z-index
def get_random_slice(images, labels):
    z = random.randint(0, images.shape[0] - 1)
    img = images[z]                 # (C, H, W)
    mask = np.argmax(labels[z], 0)  # collapse one-hot channels → (H, W)
    return img, mask, z

# Display each modality and the segmentation mask, add dimensions to titles
def plot_modalities_and_mask(img, mask, output_path, z_index, case_name):
    _, H, W = img.shape
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for i in range(4):
        axs[i].imshow(img[i], cmap='gray')
        axs[i].set_title(f"{MODALITIES[i]} ({H}×{W})")
        axs[i].axis('off')

    axs[4].imshow(mask, cmap='jet')
    axs[4].set_title(f"Mask ({H}×{W})")
    axs[4].axis('off')

    fig.suptitle(f'Case: {case_name} | Slice Z = {z_index}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Open the image using the default Linux viewer
def open_image(path):
    subprocess.run(['xdg-open', path])

# --- Main Execution ---
def main():
    case_path = choose_random_case(DATA_DIR)
    case_name = os.path.basename(case_path).replace('.npz', '')
    images, labels = load_npz(case_path)
    img, mask, z = get_random_slice(images, labels)
    plot_modalities_and_mask(img, mask, OUTPUT_IMAGE, z, case_name)
    open_image(OUTPUT_IMAGE)

if __name__ == "__main__":
    main()
