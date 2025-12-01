import os
import pickle
from typing import Optional, Dict, Any
import torch
import numpy as np


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    import json
    with open(metadata_path, 'r') as f:
        return json.load(f)


def _try_pickle_load(path: str):
    with open(path, 'rb') as f:
        # Try permissive torch.load first (may require weights_only=False for legacy dumps)
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            # Older PyTorch may not accept weights_only; fall back to default torch.load
            try:
                return torch.load(path, map_location='cpu')
            except Exception:
                pass
        except Exception:
            # If torch.load with weights_only=False failed, try default torch.load next
            try:
                return torch.load(path, map_location='cpu')
            except Exception:
                pass

        # Finally attempt pickle.load as last resort (unsafe)
        f.seek(0)
        try:
            return pickle.load(f)
        except Exception:
            # If pickle failed due to persistent ids referencing storage files,
            # attempt a custom Unpickler that knows how to load the storages
            # from the `model_dir/data` folder. We cannot know `model_dir`
            # here, so the caller may call `try_custom_unpickle` directly.
            raise


def try_custom_unpickle(pickle_path: str, model_dir: str):
    """Attempt unpickling with a custom persistent_load that reads
    storage files from `model_dir/data/` and reconstructs tensors.
    Returns the unpickled object on success or raises.
    """
    class StorageUnpickler(pickle.Unpickler):
        def __init__(self, f, model_dir):
            super().__init__(f)
            self.model_dir = model_dir

        def persistent_load(self, pid):
            # Expected format seen in the dump: ('storage', <class 'torch.FloatStorage'>, '0', 'cuda:0', size)
            if not isinstance(pid, tuple):
                raise pickle.UnpicklingError('Unsupported persistent id format')
            tag = pid[0]
            if tag != 'storage':
                raise pickle.UnpicklingError(f'Unsupported persistent id tag: {tag}')

            _, storage_cls, fname, device, size = pid

            # Map storage class name to numpy dtype
            cls_name = None
            try:
                cls_name = storage_cls.__name__
            except Exception:
                cls_name = str(storage_cls)

            dtype_map = {
                'FloatStorage': np.float32,
                'DoubleStorage': np.float64,
                'LongStorage': np.int64,
                'IntStorage': np.int32,
                'ShortStorage': np.int16,
                'ByteStorage': np.uint8,
                'HalfStorage': np.float16,
            }

            np_dtype = dtype_map.get(cls_name, None)
            if np_dtype is None:
                # Try to infer from name
                if 'float' in cls_name.lower():
                    np_dtype = np.float32
                elif 'double' in cls_name.lower():
                    np_dtype = np.float64
                elif 'long' in cls_name.lower():
                    np_dtype = np.int64
                else:
                    raise pickle.UnpicklingError(f'Unknown storage class: {cls_name}')

            data_file = os.path.join(self.model_dir, 'data', str(fname))
            if not os.path.exists(data_file):
                # Try alternative dot-data folder
                alt = os.path.join(self.model_dir, '.data', str(fname))
                if os.path.exists(alt):
                    data_file = alt
                else:
                    raise FileNotFoundError(f'Storage file not found: {data_file}')

            # Read raw bytes and interpret as numpy array
            with open(data_file, 'rb') as df:
                raw = df.read()

            # Create numpy array from raw bytes. Use count=size to avoid overrun.
            arr = np.frombuffer(raw, dtype=np_dtype, count=size)

            # Convert to torch tensor (cpu) and return its storage object
            try:
                t = torch.from_numpy(arr).contiguous()
            except Exception:
                # Fallback: create Python list then tensor (slower)
                t = torch.tensor(arr.tolist())

            return t.storage()

    with open(pickle_path, 'rb') as f:
        unpickler = StorageUnpickler(f, model_dir)
        return unpickler.load()
def build_state_dict_from_dump(dump: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
    """
    Tries to map a generic dump (dict or tuple/list) to a PyTorch state_dict
    with keys expected by ItoE_Inference: ent_drift.weight, ent_diff.weight,
    rel_drift.weight, rel_diff.weight.
    """
    state = {}

    # If dump is already a dict with matching keys, map directly
    if isinstance(dump, dict):
        # common possible keys
        keymap = {
            'ent_drift': 'ent_drift.weight',
            'ent_diff': 'ent_diff.weight',
            'rel_drift': 'rel_drift.weight',
            'rel_diff': 'rel_diff.weight',
        }
        for k, tk in keymap.items():
            if k in dump:
                state[tk] = torch.as_tensor(dump[k])

        # If dump contains names like 'ent_weights' or 'drift_ent'
        if not state:
            # try flexible naming
            for k, v in dump.items():
                name = k.lower()
                if 'ent' in name and 'drift' in name and state.get('ent_drift.weight') is None:
                    state['ent_drift.weight'] = torch.as_tensor(v)
                if 'ent' in name and ('diff' in name or 'sigma' in name) and state.get('ent_diff.weight') is None:
                    state['ent_diff.weight'] = torch.as_tensor(v)
                if 'rel' in name and 'drift' in name and state.get('rel_drift.weight') is None:
                    state['rel_drift.weight'] = torch.as_tensor(v)
                if 'rel' in name and ('diff' in name or 'sigma' in name) and state.get('rel_diff.weight') is None:
                    state['rel_diff.weight'] = torch.as_tensor(v)

        # If dump contains arrays of shapes, attempt to infer by shapes using metadata
        if not state and metadata is not None:
            ent_count = len(metadata.get('ent2id', {}))
            rel_count = len(metadata.get('rel2id', {}))
            dim = metadata.get('dim')
            for v in dump.values():
                try:
                    arr = torch.as_tensor(v)
                except Exception:
                    continue
                if arr.dim() == 2 and arr.shape[0] == ent_count and arr.shape[1] == dim:
                    # If no ent_drift yet, set it
                    if 'ent_drift.weight' not in state:
                        state['ent_drift.weight'] = arr
                    elif 'ent_diff.weight' not in state:
                        state['ent_diff.weight'] = arr
                if arr.dim() == 2 and arr.shape[0] == rel_count and arr.shape[1] == dim:
                    if 'rel_drift.weight' not in state:
                        state['rel_drift.weight'] = arr
                    elif 'rel_diff.weight' not in state:
                        state['rel_diff.weight'] = arr

    # If dump is a list/tuple try to assign by order (best-effort)
    if not state and isinstance(dump, (list, tuple)):
        tensors = [torch.as_tensor(x) for x in dump]
        for t in tensors:
            if t.dim() == 2:
                # heuristics based on sizes
                if metadata is not None:
                    ent_count = len(metadata.get('ent2id', {}))
                    rel_count = len(metadata.get('rel2id', {}))
                    dim = metadata.get('dim')
                    if t.shape[0] == ent_count and t.shape[1] == dim and 'ent_drift.weight' not in state:
                        state['ent_drift.weight'] = t
                        continue
                    if t.shape[0] == ent_count and t.shape[1] == dim and 'ent_diff.weight' not in state:
                        state['ent_diff.weight'] = t
                        continue
                    if t.shape[0] == rel_count and t.shape[1] == dim and 'rel_drift.weight' not in state:
                        state['rel_drift.weight'] = t
                        continue
                    if t.shape[0] == rel_count and t.shape[1] == dim and 'rel_diff.weight' not in state:
                        state['rel_diff.weight'] = t
                        continue
                # fallback: fill in order
                if 'ent_drift.weight' not in state:
                    state['ent_drift.weight'] = t
                elif 'ent_diff.weight' not in state:
                    state['ent_diff.weight'] = t
                elif 'rel_drift.weight' not in state:
                    state['rel_drift.weight'] = t
                elif 'rel_diff.weight' not in state:
                    state['rel_diff.weight'] = t

    return state


def load_itoe_state_dict(model_dir: str, metadata_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """
    Load the custom itoe_model directory and return a PyTorch state_dict
    compatible with ItoE_Inference. Raises RuntimeError if not possible.
    """
    if metadata_path is None:
        metadata_path = os.path.join(os.path.dirname(model_dir), 'itoe_metadata.json')
    metadata = None
    if os.path.exists(metadata_path):
        metadata = load_metadata(metadata_path)

    # Common file candidates
    candidates = [
        os.path.join(model_dir, 'data.pkl'),
        os.path.join(model_dir, 'data', '0'),
        os.path.join(model_dir, 'data.pkl.gz'),
        os.path.join(model_dir, 'data/0'),
    ]

    found = None
    for c in candidates:
        if os.path.exists(c):
            found = c
            break

    if not found:
        # Try plain data.pkl in project root
        alt = os.path.join(model_dir, '..', 'data.pkl')
        if os.path.exists(alt):
            found = alt

    if not found:
        raise RuntimeError(f"No se encontró un archivo legible dentro de '{model_dir}'. Buscados: {candidates}")

    try:
        dump = _try_pickle_load(found)
    except Exception as e:
        # If regular loading fails (e.g. due to persistent id storage refs),
        # attempt the custom unpickler that knows how to read `model_dir/data/*`.
        try:
            dump = try_custom_unpickle(found, model_dir)
        except Exception:
            # Re-raise original exception for debugging
            raise
    state = build_state_dict_from_dump(dump, metadata)

    # Validate state contains required keys
    required = ['ent_drift.weight', 'ent_diff.weight', 'rel_drift.weight', 'rel_diff.weight']
    missing = [k for k in required if k not in state]
    if missing:
        raise RuntimeError(f"No se pudieron mapear todos los parámetros al state_dict. Faltan: {missing}")

    return state
