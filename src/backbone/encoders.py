"""Frozen encoders. Each sample is processed independently and deterministically."""
import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from PIL import Image
from tqdm import tqdm

from .common import ROOT, device_for, load_pack, save_json, sha256, save_npy
from .data import assert_training


class FrozenEncoders:
    def __init__(self, device="mps"):
        torch.set_num_threads(4)
        os.environ.setdefault("TORCH_HOME", str(ROOT / "models/torch"))
        registry = json.loads((ROOT / "models/registry.json").read_text())
        for name, expected in registry["files"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Frozen pretrained file changed: {name}")
        from ._facenet.inception_resnet_v1 import InceptionResnetV1
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.device = device_for(device)
        self.face = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        model_path = ROOT / "models/wavlm-base-plus-sv"
        self.processor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)
        self.voice = WavLMForXVector.from_pretrained(model_path, local_files_only=True,
                                                   attn_implementation="eager").eval().to(self.device)
        for model in (self.face, self.voice):
            model.requires_grad_(False)

    @torch.inference_mode()
    def face_embedding(self, path):
        return self.face_embeddings([path])[0]

    @torch.inference_mode()
    def face_embeddings(self, paths):
        # Official data already provides face crops. No flip/multicrop augmentation.
        arrays = []
        for path in paths:
            im = Image.open(path).convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)
            arrays.append((np.asarray(im, dtype=np.float32).copy() - 127.5) / 128.0)
        x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(self.device)
        return self.face(x).cpu().numpy()

    @torch.inference_mode()
    def voice_embedding(self, path):
        return self.voice_embeddings([path])[0]

    @torch.inference_mode()
    def voice_embeddings(self, paths):
        waves = []
        lengths = []
        for path in paths:
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(audio.mean(axis=1))
            if rate != 16000:
                wav = torchaudio.functional.resample(wav, rate, 16000)
            if len(wav) < 4000:
                raise ValueError(f"Clip too short for normal WavLM inference: {path}")
            waves.append(wav.numpy())
            lengths.append(math.ceil(len(wav) / 16000) * 16000)
        if len(set(lengths)) != 1:
            raise ValueError("Training-only batches must share the same independently determined padding length")
        # Full clip once; standard zero-padding + attention mask to the next whole
        # second, determined only from this clip's own length. Finite shape choices
        # avoid compiling thousands of distinct MPS graphs. No truncation or pooling.
        values = self.processor(waves, sampling_rate=16000, return_tensors="pt",
                                padding="max_length", max_length=lengths[0])
        values = {k: v.to(self.device) for k, v in values.items()}
        embedding = self.voice(**values).embeddings
        return torch.nn.functional.normalize(embedding, dim=-1).cpu().numpy()


def extract_training(pack_path, out, device="mps"):
    source = assert_training(pack_path)
    pack = load_pack(pack_path)
    out = Path(out)
    cache = out.parent / (out.stem + "_cache")
    cache.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(pack_path)
    registry_hash = sha256(ROOT / "models/registry.json")
    cache_meta = cache / "provenance.json"
    expected = {"source_sha256": source_hash, "encoder_registry_sha256": registry_hash,
                "preprocessing": "rgb_resize160_fixed_standardization_wavlm_full_clip_rightpad1s_v2"}
    if cache_meta.exists() and json.loads(cache_meta.read_text()) != expected:
        raise ValueError("Feature cache provenance differs; use a new output path")
    save_json(cache_meta, expected)
    encoders = FrozenEncoders(device)
    # Training data may be batched by length. Padding is fixed independently for
    # each sample; no batch composition can change its input. Test inference
    # continues to encode one trial at a time, with exactly the same padding rule.
    missing_faces = [i for i in range(len(pack["labels"])) if not (cache / f"face_{i:05d}.npy").exists()]
    for start in tqdm(range(0, len(missing_faces), 32), desc="training face batches"):
        indices = missing_faces[start:start+32]
        embeddings = encoders.face_embeddings([str(pack["face_paths"][i]) for i in indices])
        for i, e in zip(indices, embeddings):
            save_npy(cache / f"face_{i:05d}.npy", e)
    groups = {}
    for i, path in enumerate(pack["voice_paths"]):
        if not (cache / f"voice_{i:05d}.npy").exists():
            info = sf.info(str(path))
            length = math.ceil(math.ceil(info.frames * 16000 / info.samplerate) / 16000)
            groups.setdefault(length, []).append(i)
    with tqdm(total=sum(len(g) for g in groups.values()), desc="training voices") as progress:
        for seconds, indices in sorted(groups.items()):
            batch = 8 if seconds <= 12 else 4 if seconds <= 24 else 2 if seconds <= 40 else 1
            for start in range(0, len(indices), batch):
                chosen = indices[start:start+batch]
                embeddings = encoders.voice_embeddings([str(pack["voice_paths"][i]) for i in chosen])
                for i, e in zip(chosen, embeddings):
                    save_npy(cache / f"voice_{i:05d}.npy", e)
                progress.update(len(chosen))
            if encoders.device.type == "mps":
                torch.mps.empty_cache()
    faces = [np.load(cache / f"face_{i:05d}.npy") for i in range(len(pack["labels"]))]
    voices = [np.load(cache / f"voice_{i:05d}.npy") for i in range(len(pack["labels"]))]
    # Retain complementary official features; each block normalized separately per sample.
    from .common import unit
    pack["faces"] = np.concatenate([unit(pack["faces"]), unit(np.asarray(faces))], axis=1)
    pack["voices"] = np.concatenate([unit(pack["voices"]), unit(np.asarray(voices))], axis=1)
    np.savez_compressed(out, **pack)
    save_json(out.with_suffix(".json"), {**source, **expected,
              "feature_set": "official_plus_facenet_wavlm", "pack_sha256": sha256(out)})
