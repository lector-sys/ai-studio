"""Named model registry: ops take `model=<name>` (downloaded once into the Hugging Face cache) or a
path to any compatible .onnx file, so backends swap without changing the op contract."""
from pathlib import Path

# name -> (hf repo, file, extra files from the same repo, what it is)
MODELS = {
    "yunet": ("opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx", [], "face detector (YuNet), 0.2 MB"),
    "gfpgan_1.4": ("facefusion/models-3.0.0", "gfpgan_1.4.onnx", [], "face restoration, 512 px (GFPGAN v1.4), 340 MB"),
    "real_esrgan_x2": ("facefusion/models-3.0.0", "real_esrgan_x2.onnx", [], "super-resolution x2 (Real-ESRGAN), 70 MB"),
    "real_esrgan_x4": ("facefusion/models-3.0.0", "real_esrgan_x4_fp16.onnx", [], "super-resolution x4 (Real-ESRGAN), 36 MB"),
    "rife_4.9": ("yuvraj108c/rife-onnx", "rife49_ensemble_True_scale_1_sim.onnx", [], "frame interpolation (RIFE 4.9), 21 MB"),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "onnx/model.onnx", ["tokenizer.json"],
               "sentence embeddings for semantic search (all-MiniLM-L6-v2), 90 MB"),
}


def path(model) -> str:
    """Resolve a registry name (downloading on first use) or an existing .onnx path."""
    if Path(str(model)).is_file():
        return str(model)
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}: give a .onnx path or one of {sorted(MODELS)}")
    from huggingface_hub import hf_hub_download
    repo, f, extra, _ = MODELS[model]
    for e in extra:
        hf_hub_download(repo, e)
    return hf_hub_download(repo, f)


def sibling(model, name) -> str:
    """An extra file shipped with a registry model (e.g. its tokenizer)."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(MODELS[model][0], name)


def listing() -> dict:
    from huggingface_hub import try_to_load_from_cache
    return {k: {"about": d, "repo": r, "file": f,
                "downloaded": isinstance(try_to_load_from_cache(r, f), str)} for k, (r, f, _, d) in MODELS.items()}
