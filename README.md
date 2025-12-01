# Medical Diagnosis Assistant — Modular Prototype

Este repositorio contiene un prototipo modular para un Sistema de Diagnóstico Médico
basado en Grafos de Conocimiento y un motor de embeddings estocásticos (ItôE).

Estructura relevante:
- `main.py`: runner demo que usa NER + motor de inferencia.
- `src/`: módulos fuente
  - `inference_engine.py`: punto de entrada para obtener diagnósticos. Intenta usar un modelo ItôE local, luego la API de Hugging Face, y finalmente un fallback basado en reglas.
  - `itoe_loader.py`: utilidades para cargar el modelo ItôE desde la carpeta `itoe_model/` y convertirlo a un `state_dict` de PyTorch (mejor esfuerzo).
  - `itoe_inference.py`: definición del modelo de inferencia y función para cargarlo desde `itoe_model/`.
- `itoe_metadata.json`: metadata del grafo (entidades, relaciones, dimensión).
- `itoe_model/`: directorio con los datos del modelo (formato personalizado).
- `Med_KG.ipynb` y `MedDiag.ipynb`: notebooks para construir el KG y usar el motor ItôE (notebooks adaptados para Colab).

Quick start (usando el entorno `diag_asist` que ya tienes):

```bash
# activar conda env
conda activate diag_asist

# ejecutar demo
python main.py
```

Notas importantes
- Si `itoe_model/` fue generado en un formato personalizado, `src/itoe_loader.py` intenta convertirlo automáticamente a parámetros cargables por PyTorch. Si la conversión falla, el repo proporciona un fallback determinista.
- Para usar la API de Hugging Face, configura `.env` con `HUGGINGFACE_API_TOKEN`, `HUGGINGFACE_MODEL` y `HUGGINGFACE_API_ENDPOINT`.

Próximos pasos sugeridos
- Verificar la correcta conversión de `itoe_model/` a `itoe_model.pth` (el script `src/itoe_loader.py` ya intenta esto, pero puede requerir ajustes si el formato es propietario).
- Crear notebooks tutoriales para cada fase (KG construction, ItôE training, Inference demo).

Disclaimer: Este software es experimental y no debe ser usado como consejo médico.
