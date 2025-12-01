import torch
import torch.nn as nn
from typing import Optional
import os
from .itoe_loader import load_itoe_state_dict


class ItoE_Inference(nn.Module):
    def __init__(self, num_ent: int, num_rel: int, dim: int):
        super().__init__()
        self.ent_drift = nn.Embedding(num_ent, dim)
        self.ent_diff = nn.Embedding(num_ent, dim)
        self.rel_drift = nn.Embedding(num_rel, dim)
        self.rel_diff = nn.Embedding(num_rel, dim)

    def calcular_energia(self, h_idx: torch.Tensor, r_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        h_mu = self.ent_drift(h_idx)
        t_mu = self.ent_drift(t_idx)
        r_mu = self.rel_drift(r_idx)

        h_sig = torch.abs(self.ent_diff(h_idx)) + 1e-6
        t_sig = torch.abs(self.ent_diff(t_idx)) + 1e-6
        r_sig = torch.abs(self.rel_diff(r_idx)) + 1e-6

        pred_mu = h_mu + r_mu
        pred_sig = h_sig + r_sig + 1e-6

        trace = (pred_sig / t_sig).sum(dim=-1)
        diff = ((t_mu - pred_mu) ** 2 / t_sig).sum(dim=-1)
        log_det = (torch.log(t_sig) - torch.log(pred_sig)).sum(dim=-1)

        return 0.5 * (trace + diff + log_det)


def load_itoe_model(model_dir: str, metadata_path: Optional[str] = None, map_location: str = 'cpu') -> ItoE_Inference:
    """
    Loads the itoe model from a custom directory using the loader and returns
    an initialized ItoE_Inference with weights loaded.
    """
    # load metadata to determine sizes
    import json
    if metadata_path is None:
        metadata_path = os.path.join(os.path.dirname(model_dir), 'itoe_metadata.json')

    if not os.path.exists(metadata_path):
        raise RuntimeError(f"Metadata not found at {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    num_ent = len(metadata.get('ent2id', {}))
    num_rel = len(metadata.get('rel2id', {}))
    dim = metadata.get('dim')

    model = ItoE_Inference(num_ent, num_rel, dim)

    # Try to build state_dict from folder
    state = load_itoe_state_dict(model_dir, metadata_path=metadata_path)

    # Ensure tensors are float32
    state = {k: v.to(torch.float32) for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.eval()
    return model
