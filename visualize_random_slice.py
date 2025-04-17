import random
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

# --- Configuración ---
DATA_DIR = 'data/processed'
OUTPUT_IMAGE = 'slice_visualization.png'
MODALITIES = ['FLAIR', 'T1', 'T1ce', 'T2']

# --- Funciones ---

def choose_random_case(data_dir):
    """Selecciona un archivo .npz aleatorio del directorio"""
    cases = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    return os.path.join(data_dir, random.choice(cases))

def load_npz(path):
    """Carga imágenes y etiquetas desde archivo .npz"""
    data = np.load(path)
    return data['images'], data['labels']  # (Z, C, H, W), (Z, H, W, 4)

def get_random_slice(images, labels):
    """Devuelve un corte aleatorio (img, mask)"""
    z = random.randint(0, images.shape[0] - 1)
    img = images[z]           # (C, H, W)
    mask = np.argmax(labels[z], axis=-1)  # (H, W)
    return img, mask, z

def plot_modalities_and_mask(img, mask, output_path, z_index, case_name):
    """Muestra los canales y la máscara, con dimensiones en el título"""
    # img shape: (C, H, W), mask shape: (H, W)
    _, H, W = img.shape
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for i in range(4):
        axs[i].imshow(img[i], cmap='gray')
        # Añadir dims al título
        axs[i].set_title(f"{MODALITIES[i]} ({H}×{W})")
        axs[i].axis('off')

    axs[4].imshow(mask, cmap='jet')
    axs[4].set_title(f"Mask ({H}×{W})")
    axs[4].axis('off')

    fig.suptitle(f'Caso: {case_name} | Corte Z = {z_index}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def open_image(path):
    """Abre la imagen usando el visor predeterminado de Linux"""
    subprocess.run(['xdg-open', path])

# --- Ejecución Principal ---
def main():
    case_path = choose_random_case(DATA_DIR)
    case_name = os.path.basename(case_path).replace('.npz', '')
    images, labels = load_npz(case_path)
    img, mask, z = get_random_slice(images, labels)
    plot_modalities_and_mask(img, mask, OUTPUT_IMAGE, z, case_name)
    open_image(OUTPUT_IMAGE)

if __name__ == "__main__":
    main()
