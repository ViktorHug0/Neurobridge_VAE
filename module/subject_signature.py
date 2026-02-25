from __future__ import annotations

from typing import Dict

import torch


# Relative band-power signatures provided by user (Delta, Theta, Alpha, Beta, Gamma).
SUBJECT_SIGNATURES: Dict[int, tuple[float, float, float, float, float]] = {
    1: (0.268095, 0.185404, 0.289399, 0.181020, 0.076082),
    2: (0.238498, 0.133932, 0.177798, 0.294832, 0.154940),
    3: (0.278524, 0.165583, 0.166074, 0.257245, 0.132574),
    4: (0.343162, 0.184536, 0.164946, 0.230716, 0.076640),
    5: (0.338304, 0.118610, 0.113033, 0.270022, 0.160031),
    6: (0.307150, 0.126358, 0.116187, 0.278085, 0.172220),
    7: (0.224826, 0.142246, 0.253193, 0.284403, 0.095332),
    8: (0.226231, 0.188022, 0.152403, 0.256631, 0.176713),
    9: (0.266861, 0.183055, 0.133934, 0.264177, 0.151973),
    10: (0.257376, 0.138021, 0.169078, 0.273924, 0.161601),
}


def get_subject_signatures(
    subject_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Map subject IDs (1..10) to a [B, 5] signature tensor."""
    if subject_ids.ndim != 1:
        subject_ids = subject_ids.view(-1)

    out = []
    for sid in subject_ids.detach().cpu().tolist():
        sid_int = int(sid)
        if sid_int not in SUBJECT_SIGNATURES:
            raise KeyError(f"Missing subject signature for subject_id={sid_int}.")
        out.append(SUBJECT_SIGNATURES[sid_int])
    return torch.tensor(out, device=device, dtype=dtype)

