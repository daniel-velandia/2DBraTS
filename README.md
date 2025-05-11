# Brain Tumor Segmentation Web App

Esta aplicación, desarrollada con Gradio para la interfaz web, utiliza un modelo CNN basado en U-Net entrenado para segmentar tumores cerebrales en imágenes de resonancia magnética (MRI) 2D. El modelo se entrenó usando el dataset **BraTS 2020** (MICCAI Brain Tumor Segmentation Challenge) disponible en Kaggle: [BraTS 2020 Dataset Training/Validation](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation?select=BraTS2020_TrainingData).

---

## 📋 Contenido

- [`app.py`](#apppy): Script principal que define la interfaz y la lógica de carga, preprocesamiento, predicción y visualización.
- [`brain_tumor_segmentation.keras`](#modelo): Modelo entrenado en formato Keras.
- [`requirements.txt`](#requirements): Lista de dependencias necesarias.
- [`example.zip`](#examplezip): Archivo ZIP de ejemplo (BraTS) para pruebas rápidas.
- [`README.md`](#este-archivo): Documentación y guía de uso.

---

## 🚀 Características

- **Arquitectura U-Net**: red de segmentación con codificador y decodificador, capas de convolución, normalización y activación LeakyReLU.
- **Dataset BraTS 2020**: entrenado sobre los volúmenes FLAIR, T1, T1ce y T2 del dataset oficial.
- **Carga de datos**: acepta un archivo ZIP que contenga los 4 escaneos NIfTI (`FLAIR`, `T1`, `T1ce`, `T2`).
- **Preprocesamiento**: normalización Z-score por slice y canal.
- **Predicción por batch**: envía lotes de slices al modelo para aprovechar la GPU y reducir overhead.
- **Visualización**: muestra cada slice junto con la máscara segmentada superpuesta, indicando su índice Z.
- **Descarga de máscara**: permite obtener la máscara completa en un archivo NIfTI (`.nii`).

---

## ⚙️ Requisitos

- Python 3.8 o superior
- GPU recomendada (CUDA + cuDNN) para acelerar la inferencia

Dependencias principales:

```text
numpy
tensorflow
gradio
nibabel
matplotlib
```

Puedes encontrar la lista completa en [`requirements.txt`](requirements.txt).

---

## 📥 Instalación

1. **Clona este repositorio**

   ```bash
   git clone https://github.com/daniel-velandia/2DBraTS.git
   cd 2DBraTS
   ```

2. **Crea y activa un entorno virtual** (opcional pero recomendado)

   ```bash
   python -m venv venv
   # Linux/Mac
   source venv/bin/activate
   # Windows
   venv\\Scripts\\activate
   ```

3. **Instala las dependencias**

   ```bash
   pip install -r requirements.txt
   ```

4. **Coloca tu modelo Keras** en la raíz del proyecto con el nombre `brain_tumor_segmentation.keras`.

---

## 🏃‍♂️ Uso

Ejecuta la aplicación con:

```bash
python app.py
```

- Abre tu navegador en `http://0.0.0.0:7860`.
- Sube un ZIP con los archivos NIfTI (`*_flair.nii`, `*_t1.nii`, `*_t1ce.nii`, `*_t2.nii`), o usa `example.zip` para pruebas.
- Espera a que termine la barra de progreso.
- Revisa la galería de resultados (slice vs. máscara overlay).
- Descarga la máscara completa en `.nii` usando el botón correspondiente.

---

## 🔧 Estructura de Archivos

```plaintext
.
├── app.py
├── brain_tumor_segmentation.keras
├── example.zip
├── requirements.txt
└── README.md
```

- **app.py**: Contiene toda la lógica de Gradio.
- **brain_tumor_segmentation.keras**: Modelo guardado.
- **example.zip**: ZIP con ejemplos de BraTS para testing.
- **requirements.txt**: Dependencias.
- **README.md**: Este archivo.

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [`LICENSE`](LICENSE) para más detalles.

---

**¡Gracias por utilizar la aplicación!**
