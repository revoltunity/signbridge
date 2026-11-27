"""Export the trained SignBridge PyTorch checkpoint to browser-readable float32 weights."""
import json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "signbridge_gru.pt"
OUT = HERE.parents[1] / "web" / "public" / "models"
OUT.mkdir(parents=True, exist_ok=True)

checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
state = checkpoint["model_state_dict"]
manifest = {
    "num_classes": int(checkpoint["num_classes"]),
    "input_shape": checkpoint["input_shape"],
    "parameters": {}
}
arrays = []
offset = 0
for name, tensor in state.items():
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    flat = array.reshape(-1)
    manifest["parameters"][name] = {
        "shape": list(array.shape),
        "offset": offset,
        "length": int(flat.size)
    }
    arrays.append(flat)
    offset += flat.size

np.concatenate(arrays).astype("<f4", copy=False).tofile(OUT / "signbridge_gru_weights.bin")
(OUT / "signbridge_gru_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
print("Browser weights exported:", OUT)
