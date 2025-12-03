# Medical Diagnosis Assistant — Guía de instalación y ejecución

Este submódulo contiene un prototipo modular para un Sistema de Diagnóstico Médico basado
en un Grafo de Conocimiento y un motor de embeddings (ItôE). Este README unifica instrucciones
de descarga, instalación del entorno, inicialización de modelos de Hugging Face y ejecución.

**Estructura relevante**

- `main.py`: demo/runner que orquesta NER + motor de inferencia.
- `src/`: código fuente principal (`inference_engine.py`, `itoe_loader.py`, `itoe_inference.py`, ...).
- `itoe_metadata.json`: metadata del grafo.
- `itoe_model/` y `itoe_model.pth`: artefactos locales del motor ItôE (si existen).
- `tests/`: pruebas rápidas (ej. `tests/test_node_extractor.py`).

**Requisitos**

- **Python**: 3.9+ (3.11 probado).
- **Git**: para clonar el repo.
- Opcional: `conda` o `virtualenv` para entornos aislados.

**Descargar el proyecto**

```bash
# (si aún no tienes el repo)
git clone <URL-del-repositorio>
cd <ruta-al-repo>/Python/Medical_Diagnosis_Assistant
```

Reemplaza `<URL-del-repositorio>` por la URL real (por ejemplo, la de GitHub o tu remoto).

**Crear e instalar el entorno**

```bash
# 1) Crear y activar un entorno virtual (venv)
python -m venv .venv
source .venv/bin/activate

# 2) Actualizar pip e instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# Si prefieres conda:
# conda create -n diag_asist python=3.11 -y
# conda activate diag_asist
# pip install -r requirements.txt
```

Nota sobre PyTorch y GPU: si necesitas soporte CUDA instala la rueda adecuada desde https://pytorch.org
en lugar del `torch` genérico que puede aparecer en `requirements.txt`.

**Configurar credenciales de Hugging Face (si usarás modelos privados o la API)**

```bash
# Opción A: login interactivo (recomendado para desarrolladores)
pip install huggingface_hub
huggingface-cli login

# Opción B: usar variable de entorno o .env (NO commitear .env)
export HUGGINGFACE_API_TOKEN="hf_xxx"
cat > .env <<'ENV'
# HUGGINGFACE_API_TOKEN=hf_xxx
# HUGGINGFACE_MODEL=d4data/biomedical-ner-all
# USE_LOCAL_ITOE=1
ENV
```

**Ejecución en Google Colab — guía paso a paso (rápida y directa)**

Sigue estas celdas exactamente (copiar/pegar en Colab). Estas instrucciones usan como ejemplo
el repositorio público https://github.com/ernestoelo/Medical_Diagnosis.git y el subdirectorio
`Python/Medical_Diagnosis_Assistant`.

1) Crear notebook y activar Drive (recomendado)

```python
# 1. Montar Google Drive (para persistir modelos y evitar descargas repetidas)
from google.colab import drive
drive.mount('/content/drive')

# Crear carpeta en Drive (opcional)
!mkdir -p /content/drive/MyDrive/MedicalDiagnosis/models
```

2) Clonar el repositorio y cambiar al subdirectorio

```bash
# 2. Clonar repo (reemplaza si tu repo es otro)
git clone https://github.com/ernestoelo/Medical_Diagnosis.git repo
cd repo/Python/Medical_Diagnosis_Assistant
ls -la
```

3) Instalar dependencias (CPU/GPU)

```bash
# 3. Instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt

# Si quieres usar GPU en Colab y necesitas una versión específica de PyTorch,
# instala la rueda adecuada desde https://pytorch.org (ajusta CUDA según el runtime):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4) Autenticarse en Hugging Face (si usarás modelos privados)

```python
from huggingface_hub import login
login()  # Pega tu token cuando lo solicite (más seguro que ponerlo en una celda)
```

Alternativa (no recomendada en notebooks compartidos):

```python
import os
os.environ['HUGGINGFACE_API_TOKEN'] = 'hf_...'
```

5) (Recomendado) Descargar modelos a Drive para caché

```python
from huggingface_hub import snapshot_download
repo_id = 'd4data/biomedical-ner-all'
path_local = snapshot_download(repo_id, cache_dir='/content/drive/MyDrive/MedicalDiagnosis/models')
print('Modelo descargado en', path_local)
```

6) (Opcional) Colocar `itoe_model/` en Drive

Si tienes un artefacto `itoe_model/` o `itoe_model.pth`, súbelo a Drive y cópialo dentro del submódulo:

```bash
# copiar desde Drive al workspace de Colab (si hiciste upload manual)
cp -r /content/drive/MyDrive/MedicalDiagnosis/models/itoe_model ./ || true
ls -la itoe_model || true
```

7) Ejecutar el demo (runner)

```bash
# Ejecutar demo (asegúrate de estar en repo/Python/Medical_Diagnosis_Assistant)
export HUGGINGFACE_API_TOKEN="$HUGGINGFACE_API_TOKEN"
PYTHONPATH=. python main.py
```

Notas importantes

- Activa GPU en Colab: `Runtime -> Change runtime type -> GPU` para acelerar inferencia.
- Usa Drive para almacenar modelos grandes y evitar volver a descargarlos.
- No pegues tokens en celdas que guardes públicamente; usa `huggingface_hub.login()`.

Problemas comunes

- Si `torch` da errores: instala la rueda compatible con la versión CUDA del runtime de Colab.
- Si las descargas tardan mucho, guarda los modelos en Drive y usa `snapshot_download(..., cache_dir=...)`.

**Inicializar / descargar un modelo desde Hugging Face**

Puedes usar `transformers` o `huggingface_hub` para descargar y cargar modelos. Ejemplos:

```python
# Ejemplo: descargar y cargar un modelo de Transformers (NER o clasificación)
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_NAME = "d4data/biomedical-ner-all"  # reemplaza por el id del modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# Si el modelo es privado, asegúrate de tener HUGGINGFACE_API_TOKEN en env vars
```

O, si prefieres descargar los archivos localmente (útil para despliegues offline):

```python
from huggingface_hub import snapshot_download

repo_id = "d4data/biomedical-ner-all"
path_local = snapshot_download(repo_id)
# path_local contendrá los pesos y la config descargados
```

**Uso del modelo ItôE local**

El proyecto incluye utilidades para cargar artefactos locales de ItôE en `src/itoe_loader.py`.
Por defecto la lógica en `src/itoe_loader.py` intenta convertir el contenido de `itoe_model/`
en un `state_dict` compatible con PyTorch (por ejemplo `itoe_model.pth`). Para inicializar localmente:

```bash
# Asegúrate que `itoe_model/` o `itoe_model.pth` estén en la raíz del submódulo
ls -la itoe_model*  # comprobar presencia
```

Si `main.py` detecta `itoe_model/` intentará usarlo primero; si no, intentará usar el modelo
de Hugging Face configurado en `.env` o las variables de entorno.

**Ejecutar pruebas y demo**

```bash
# Ejecutar pruebas rápidas
PYTHONPATH=. python -m pytest tests/test_node_extractor.py -q

# Ejecutar el demo/runner
PYTHONPATH=. python main.py
```

Revisa `main.py` para ver parámetros disponibles; suele respetar variables de entorno como
`HUGGINGFACE_MODEL`, `HUGGINGFACE_API_TOKEN` y flags para forzar uso local (`USE_LOCAL_ITOE`).

**Notas y recomendaciones**

- **Ficheros importantes**: `src/itoe_loader.py` (carga y conversión), `src/itoe_inference.py` (modelo), `main.py` (entrypoint).
- **Privacidad**: no comites tokens; usa `.env` o variables de entorno y añade `.env` a `.gitignore`.
- **Rendimiento**: para inferencia intensiva usa GPU (instala la rueda de `torch` con CUDA adecuada).

**Solución de problemas**

- Error de importación `torch`: instala la versión de PyTorch adecuada para tu sistema/CUDA.
- Descargas lentas: la primera ejecución descarga pesos de Transformers y puede tardar varios minutos.
- Fallo al cargar ItôE: revisa `src/itoe_loader.py` y los logs; puede requerir adaptar la conversión si el artefacto es propietario.

**Contacto / contribuciones**
Si quieres, puedo:

- Añadir un script `scripts/download_model.sh` para automatizar la descarga de modelos HF.
- Crear una GitHub Action que ejecute las pruebas básicas en push.

---

Disclaimer: este software es experimental y no debe usarse como sustituto de consejo médico.
