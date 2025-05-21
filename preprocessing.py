import os
import numpy as np
import nibabel as nib
from keras.utils import to_categorical
from tqdm import tqdm

# --- Configuration ---
DATASET_PATH = 'your_path/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
OUTPUT_PATH = 'data/processed'
MODALITIES = ['flair', 't1', 't1ce', 't2']

os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- Utility functions ---

# Loads images from FLAIR, T1, T1ce, and T2 modalities
def load_modalities(case_path, case_id):
    return np.stack([
        nib.load(os.path.join(case_path, f"{case_id}_{mod}.nii")).get_fdata()
        for mod in MODALITIES
    ])

# Loads the segmentation mask and remaps class 4 to 3
def load_segmentation(case_path, case_id):
    seg_path = os.path.join(case_path, f"{case_id}_seg.nii")
    label = nib.load(seg_path).get_fdata()
    label[label == 4] = 3
    return to_categorical(label, num_classes=4)  # (H, W, Z, C)

# Normalizes slice-by-slice and channel-by-channel using z-score
def standardize(image):
    standardized = np.zeros_like(image)
    for c in range(image.shape[0]):
        for z in range(image.shape[3]):
            slice_ = image[c, :, :, z]
            mean, std = np.mean(slice_), np.std(slice_)
            standardized[c, :, :, z] = (slice_ - mean) / std if std != 0 else np.zeros_like(slice_)
    return standardized

# Saves the preprocessed case to a .npz file
def save_case(image, label, output_dir, case_id):
    images = image.astype(np.float32).transpose(3, 0, 1, 2)  # (C, H, W, Z) → (Z, C, H, W)
    labels = label.astype(np.uint8).transpose(2, 3, 0, 1)    # (H, W, Z, C) → (Z, C, H, W)
    np.savez_compressed(
        os.path.join(output_dir, f"{case_id}.npz"),
        images=images,
        labels=labels
    )

# Processing pipeline for a single case
def process_case(case_folder):
    case_path = os.path.join(DATASET_PATH, case_folder)
    image = load_modalities(case_path, case_folder)
    label = load_segmentation(case_path, case_folder)
    image = standardize(image)
    save_case(image, label, OUTPUT_PATH, case_folder)

# --- Main processing ---
def main():
    cases = [c for c in os.listdir(DATASET_PATH) if c.startswith('BraTS20_Training')]
    for case in tqdm(cases, desc="Processing cases", ncols=100, unit="case"):
        process_case(case)

if __name__ == "__main__":
    main()
