import os
import numpy as np
import nibabel as nib
from keras.utils import to_categorical
from tqdm import tqdm

# --- Configuración ---
DATASET_PATH = 'your_path/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
OUTPUT_PATH = 'data/processed'
MODALITIES = ['flair', 't1', 't1ce', 't2']

os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- Funciones utilitarias ---

def load_modalities(case_path, case_id):
    """Carga las imágenes de las modalidades FLAIR, T1, T1ce, T2"""
    return np.stack([
        nib.load(os.path.join(case_path, f"{case_id}_{mod}.nii")).get_fdata()
        for mod in MODALITIES
    ])

def load_segmentation(case_path, case_id):
    """Carga la máscara de segmentación y ajusta clase 4 a 3"""
    seg_path = os.path.join(case_path, f"{case_id}_seg.nii")
    label = nib.load(seg_path).get_fdata()
    label[label == 4] = 3
    return to_categorical(label, num_classes=4)  # (H, W, Z, C)

def standardize(image):
    """Normaliza canal por canal y corte por corte usando z-score"""
    standardized = np.zeros_like(image)
    for c in range(image.shape[0]):
        for z in range(image.shape[3]):
            slice_ = image[c, :, :, z]
            mean, std = np.mean(slice_), np.std(slice_)
            standardized[c, :, :, z] = (slice_ - mean) / std if std != 0 else np.zeros_like(slice_)
    return standardized

def save_case(image, label, output_dir, case_id):
    """Guarda el caso preprocesado como archivo .npz"""
    images = image.astype(np.float32).transpose(3, 0, 1, 2)  # (C, H, W, Z) → (Z, C, H, W)
    labels = label.astype(np.uint8).transpose(2, 0, 1, 3)    # (H, W, Z, C) → (Z, H, W, C)
    np.savez_compressed(
        os.path.join(output_dir, f"{case_id}.npz"),
        images=images,
        labels=labels
    )

def process_case(case_folder):
    """Pipeline de procesamiento de un solo caso"""
    case_path = os.path.join(DATASET_PATH, case_folder)
    image = load_modalities(case_path, case_folder)
    label = load_segmentation(case_path, case_folder)
    image = standardize(image)
    save_case(image, label, OUTPUT_PATH, case_folder)

# --- Procesamiento principal ---
def main():
    cases = [c for c in os.listdir(DATASET_PATH) if c.startswith('BraTS20_Training')]
    for case in tqdm(cases, desc="Procesando casos", ncols=100, unit="caso"):
        process_case(case)

if __name__ == "__main__":
    main()
