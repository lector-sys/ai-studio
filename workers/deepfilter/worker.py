"""DeepFilterNet3 speech enhancement (CPU): in.wav (48 kHz float) -> out.wav, same length.
usage: worker.py in.wav out.wav [--limit dB]   (setup: docs/LOCAL_WORKERS.md, deepfilter section)"""
import argparse
import json
import sys
import types

import soundfile as sf
import torch

try:  # df 0.5.6 imports a torchaudio module newer torchaudio removed; it only needs the name
    import torchaudio.backend.common  # noqa: F401
except ImportError:
    m = types.ModuleType("torchaudio.backend.common")
    m.AudioMetaData = object
    sys.modules["torchaudio.backend"] = types.ModuleType("torchaudio.backend")
    sys.modules["torchaudio.backend.common"] = m

from df import __version__ as df_version  # noqa: E402
from df.enhance import enhance, init_df  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--limit", type=float, default=None, help="max attenuation in dB (None = unlimited)")
a = ap.parse_args()
torch.manual_seed(0)
model, state, _ = init_df(log_level="ERROR", log_file=None)
x, r = sf.read(a.src, dtype="float32", always_2d=True)
if r != state.sr():
    sys.exit(f"needs {state.sr()} Hz input, got {r}")
y = enhance(model, state, torch.from_numpy(x.T.copy()), atten_lim_db=a.limit).numpy().T[:len(x)]
sf.write(a.dst, y, r, subtype="FLOAT")
print(json.dumps({"model": "DeepFilterNet3", "deepfilternet": df_version, "torch": torch.__version__,
                  "limit_db": a.limit, "samples": len(x), "channels": x.shape[1]}))
