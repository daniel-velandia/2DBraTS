import os
import io
import zipfile
import tempfile
from pathlib import Path
from functools import lru_cache
from typing import List, Tuple, Optional

import gradio as gr
import numpy as np
import nibabel as nib
import tensorflow as tf
from matplotlib import pyplot as plt

# --- Configuration and constants ---
MODEL_PATH: str = "brain_tumor_segmentation.keras"
MODALITIES: List[str] = ['flair.nii', 't1.nii', 't1ce.nii', 't2.nii']
FIGSIZE: Tuple[int, int] = (8, 4)
GALLERY_COLUMNS: int = 3
BATCH_SIZE: int = 8

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

# Reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_model() -> tf.keras.Model:
    """Load cached segmentation model from disk"""
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = lru_cache(maxsize=1)(load_model)()


def standardize_slices(slices: np.ndarray) -> np.ndarray:
    """Apply Z-score normalization per slice (input shape: Z, H, W, C)"""
    means = np.mean(slices, axis=(1, 2), keepdims=True)
    stds = np.std(slices, axis=(1, 2), keepdims=True)
    return (slices - means) / (stds + 1e-6)


def load_modalities_from_dir(directory: str, modalities: List[str]) -> np.ndarray:
    """Load and stack NIfTI modalities into 4D array (H, W, Z, C)"""
    vols = []
    for mod in modalities:
        path = next(Path(directory).glob(f"*{mod}"), None)
        if path is None:
            raise gr.Error(f"Falta la modalidad {mod} en el ZIP")
        vols.append(nib.load(str(path)).get_fdata())
    
    if len({v.shape for v in vols}) != 1:
        raise gr.Error("Formas inconsistentes entre modalidades")
    
    return np.stack(vols, axis=-1)


def load_and_preprocess(zip_file: io.BytesIO) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ZIP and return raw/normalized volumes"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file.name) as z:
            z.extractall(tmpdir)
        vol4 = load_modalities_from_dir(tmpdir, MODALITIES)
    
    raw = vol4.transpose(2, 0, 1, 3)  # (Z, H, W, C)
    norm = standardize_slices(raw)
    return raw, norm


def slice_to_image(orig_slice: np.ndarray, mask: np.ndarray, z_pos: int) -> np.ndarray:
    """Overlay RGB mask on grayscale slice with Z position label"""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    
    # Original slice
    axes[0].imshow(orig_slice[..., 0], cmap='gray')
    axes[0].axis('off')
    axes[0].set_title(f"Slice original (Z={z_pos})")
    
    # Mask overlay
    rgb = np.zeros(orig_slice.shape[:2] + (3,), dtype=float)
    rgb[mask == 1] = [0, 1, 0]  # Verde
    rgb[mask == 2] = [1, 0, 0]  # Rojo
    rgb[mask == 3] = [0, 0, 1]  # Azul
    
    axes[1].imshow(orig_slice[..., 0], cmap='gray')
    axes[1].imshow(rgb, alpha=0.4)
    axes[1].axis('off')
    axes[1].set_title(f"Segmentación (Z={z_pos})")
    
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def predict(
    zip_file: io.BytesIO, 
    progress: gr.Progress = gr.Progress()
) -> Tuple[List[np.ndarray], Optional[str]]:
    """Full prediction pipeline with fractional progress"""
    try:
        raw_slices, norm_slices = load_and_preprocess(zip_file)
    except gr.Error as e:
        return [], None
    
    total_slices = len(norm_slices)
    all_masks = []
    gallery_images = []
    
    for start in range(0, total_slices, BATCH_SIZE):
        batch = norm_slices[start:start + BATCH_SIZE]
        preds = model.predict(batch, verbose=0)
        
        for i, pred in enumerate(preds):
            mask = np.argmax(pred, axis=-1)
            all_masks.append(mask)
            gallery_images.append(
                slice_to_image(raw_slices[start + i], mask, start + i)
            )
        
        # Update progress
        progress_fraction = min((start + BATCH_SIZE) / total_slices, 1.0)
        progress(progress_fraction, "Procesando cortes")
    
    # Save final mask
    mask_volume = np.stack(all_masks, axis=0).astype(np.uint8)
    nifti_img = nib.Nifti1Image(mask_volume, affine=np.eye(4))
    
    with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp_nii:
        nib.save(nifti_img, tmp_nii.name)
        tmp_path = tmp_nii.name
    
    return gallery_images, tmp_path


def create_interface() -> gr.Interface:
    """Create Gradio interface with queueing"""
    interface = gr.Interface(
        fn=predict,
        inputs=gr.File(label="Subir archivo ZIP con escanes (FLAIR, T1, T1ce, T2)"),
        outputs=[
            gr.Gallery(label="Resultados de segmentación", columns=GALLERY_COLUMNS),
            gr.File(label="Descargar máscara en formato NIfTI")
        ],
        title="Segmentación Automática de Tumores Cerebrales",
        examples=[["./example.zip"]] if Path("./example.zip").exists() else None
    )
    return interface


if __name__ == '__main__':
    demo = create_interface()
    demo.queue()  # Enable request queueing
    demo.launch(server_name='0.0.0.0', server_port=7860)