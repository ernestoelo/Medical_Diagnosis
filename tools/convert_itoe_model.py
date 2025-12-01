"""Conversion helper: intenta cargar el modelo ItôE desde `itoe_model/` y guarda
un `itoe_model.pth` (state_dict de PyTorch) en la raíz del proyecto.

USO:
    python tools/convert_itoe_model.py

Este script intentará cargar de forma permisiva (puede usar `torch.load` con
`weights_only=False` y `pickle.load`) para recuperar los arrays y mapearlos a
los parámetros esperados. Usa esto solo si confías en la fuente de los archivos.
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.itoe_loader import load_itoe_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='itoe_model', help='Directorio con el modelo ItôE')
    parser.add_argument('--metadata', default='itoe_metadata.json', help='Ruta al metadata JSON')
    parser.add_argument('--out', default='itoe_model.pth', help='Archivo de salida .pth')
    parser.add_argument('--unsafe', action='store_true', help='Permitir cargas inseguras (pickle/weights_only=False)')
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    metadata = os.path.abspath(args.metadata)
    out_file = os.path.abspath(args.out)

    print(f"Intentando convertir modelo en: {model_dir}")
    if not os.path.exists(model_dir):
        print(f"Error: no existe {model_dir}")
        return 2

    # load_itoe_state_dict may attempt unsafe operations; user signaled consent by running this script
    try:
        state = load_itoe_state_dict(model_dir, metadata_path=metadata)
    except Exception as e:
        print(f"Fallo al intentar construir state_dict: {e}")
        return 3

    # Save state_dict as .pth
    try:
        torch.save(state, out_file)
        print(f"Guardado exitoso: {out_file}")
    except Exception as e:
        print(f"Error guardando {out_file}: {e}")
        return 4

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
