import os
import io
import zipfile
import tempfile
from pathlib import Path
from functools import lru_cache
from typing import List, Tuple

import gradio as gr
import numpy as np
import nibabel as nib
import tensorflow as tf
from matplotlib import pyplot as plt

# --- Configuración y constantes ---
MODEL_PATH: str = "brain_tumor_segmentation.keras"
MODALITIES: List[str] = ['flair.nii', 't1.nii', 't1ce.nii', 't2.nii']
FIGSIZE: Tuple[int, int] = (8, 4)
GALLERY_COLUMNS: int = 3
BATCH_SIZE: int = 8

# Reducir logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_model() -> tf.keras.Model:
    """
    Carga el modelo de segmentación guardado en disco (cached).
    """
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = lru_cache(maxsize=1)(load_model)()


def standardize_slices(slices: np.ndarray) -> np.ndarray:
    """
    Aplica Z-score por slice.
    Input:  (Z, H, W, C)
    Output: mismo shape, dtype=float32
    """
    means = np.mean(slices, axis=(1, 2), keepdims=True)
    stds = np.std(slices, axis=(1, 2), keepdims=True)
    return (slices - means) / (stds + 1e-6)


def load_modalities_from_dir(directory: str, modalities: List[str]) -> np.ndarray:
    """
    Carga y apila modalidades NIfTI en un array 4D (H, W, Z, C).
    """
    vols = []
    for mod in modalities:
        path = next(Path(directory).glob(f"*{mod}"), None)
        if path is None:
            raise gr.Error(f"Falta la modalidad {mod} en el ZIP")
        vols.append(nib.load(str(path)).get_fdata())
    shapes = {v.shape for v in vols}
    if len(shapes) != 1:
        raise gr.Error(f"Formas inconsistentes: {shapes}")
    return np.stack(vols, axis=-1)


def load_and_preprocess(zip_file) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae un ZIP con 4 archivos NIfTI y devuelve los volúmenes raw y normalizados:
    - raw: (Z, H, W, C)
    - norm: (Z, H, W, C) Z-score
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file.name) as z:
            z.extractall(tmpdir)
        vol4 = load_modalities_from_dir(tmpdir, MODALITIES)

    raw = vol4.transpose(2, 0, 1, 3)
    norm = standardize_slices(raw)
    return raw, norm


def slice_to_image(orig_slice: np.ndarray, mask: np.ndarray, slice_index: int) -> np.ndarray:
    """
    Superpone la máscara RGB sobre el slice en escala de grises y añade título con índice Z.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    axes[0].imshow(orig_slice[..., 0], cmap='gray')
    axes[0].axis('off')
    axes[0].set_title(f"Slice {slice_index}")

    rgb = np.zeros(orig_slice.shape[:2] + (3,), dtype=float)
    rgb[mask == 1] = [0, 1, 0]
    rgb[mask == 2] = [1, 0, 0]
    rgb[mask == 3] = [0, 0, 1]

    axes[1].imshow(orig_slice[..., 0], cmap='gray')
    axes[1].imshow(rgb, alpha=0.4)
    axes[1].axis('off')
    axes[1].set_title(f"Mask overlay {slice_index}")

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def predict(zip_file, progress=gr.Progress()) -> Tuple[List[np.ndarray], gr.File]:
    """
    Pipeline completo: lectura, predicción en batches, render de imágenes, y generación de NIfTI de la máscara.
    """
    try:
        raw_slices, norm_slices = load_and_preprocess(zip_file)
    except gr.Error as e:
        return [], gr.update(error=str(e))

    total = len(norm_slices)
    all_masks = []
    results = []
    for start in progress.tqdm(range(0, total, BATCH_SIZE), desc="Procesando slices"):
        batch = norm_slices[start:start + BATCH_SIZE]
        preds = model.predict(batch, verbose=0)
        for i, pred in enumerate(preds):
            mask = np.argmax(pred, axis=-1)
            all_masks.append(mask)
            results.append(slice_to_image(raw_slices[start + i], mask, start + i))

    # Convertir máscara completa a NIfTI
    mask_volume = np.stack(all_masks, axis=0)  # (Z, H, W)
    nifti_img = nib.Nifti1Image(mask_volume.astype(np.uint8), affine=np.eye(4))
    tmp_nii = tempfile.NamedTemporaryFile(suffix='.nii', delete=False)
    nib.save(nifti_img, tmp_nii.name)

    return results, tmp_nii.name


def create_interface():
    interface = gr.Interface(
        fn=predict,
        inputs=gr.File(label="Subir ZIP con scans (FLAIR, T1, T1ce, T2)"),
        outputs=[
            gr.Gallery(label="Resultados", columns=GALLERY_COLUMNS),
            gr.File(label="Descargar máscara NIfTI")
        ],
        title="Segmentación de Tumores Cerebrales",
        flagging_mode="never",
        examples=[["./example.zip"]] if Path("./example.zip").exists() else None
    )
    return interface


if __name__ == '__main__':
    create_interface().launch(server_name='0.0.0.0', server_port=7860)
