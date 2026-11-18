"""Fully frozen public age representations; no additional encoder fitting."""
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from PIL import Image
from torch import nn
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoFeatureExtractor, ViTForImageClassification
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from .common import ROOT, device_for, load_pack, save_json, sha256, unit, save_npy
from .data import assert_training


class ModelHead(nn.Module):
    def __init__(self, config, labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, labels)

    def forward(self, x):
        return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    # Architecture matching the public audEERING model card. Loaded weights stay frozen.
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.post_init()

    def forward(self, input_values, attention_mask=None):
        states = self.wav2vec2(input_values, attention_mask=attention_mask).last_hidden_state
        if attention_mask is None:
            pooled = states.mean(dim=1)
        else:
            valid = self._get_feature_vector_attention_mask(states.shape[1], attention_mask).unsqueeze(-1)
            pooled = (states * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return pooled, self.age(pooled), self.gender(pooled).softmax(dim=-1)


class FrozenDemographics:
    def __init__(self, device="mps"):
        torch.set_num_threads(4)
        self.device = device_for(device)
        registry = json.loads((ROOT / "models/demographics_registry.json").read_text())
        for name, expected in registry["files"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Auxiliary model file changed: {name}")
        fp, vp = ROOT / "models/vit-age", ROOT / "models/wav2vec2-age-gender"
        self.face_processor = AutoImageProcessor.from_pretrained(fp, local_files_only=True)
        self.face = ViTForImageClassification.from_pretrained(fp, local_files_only=True,
                            attn_implementation="eager").eval().to(self.device)
        self.voice_processor = AutoFeatureExtractor.from_pretrained(vp, local_files_only=True)
        self.voice = AgeGenderModel.from_pretrained(vp, local_files_only=True,
                            attn_implementation="eager").eval().to(self.device)
        self.face.requires_grad_(False)
        self.voice.requires_grad_(False)

    @torch.inference_mode()
    def face_embeddings(self, paths):
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.face_processor(images=images, return_tensors="pt")
        states = self.face.vit(**{k: x.to(self.device) for k, x in inputs.items()}).last_hidden_state[:, 0]
        return states.cpu().numpy()

    @torch.inference_mode()
    def voice_embeddings(self, paths):
        waves, lengths = [], []
        for p in paths:
            audio, rate = sf.read(p, dtype="float32", always_2d=True)
            wav = torch.from_numpy(audio.mean(axis=1))
            if rate != 16000:
                wav = torchaudio.functional.resample(wav, rate, 16000)
            waves.append(wav.numpy())
            lengths.append(math.ceil(len(wav) / 16000) * 16000)
        if len(set(lengths)) != 1:
            raise ValueError("Auxiliary training batches require independently equal padded lengths")
        values = self.voice_processor(waves, sampling_rate=16000, return_tensors="pt",
                    padding="max_length", max_length=lengths[0], return_attention_mask=True)
        values = {k: v.to(self.device) for k, v in values.items()}
        return self.voice(**values)[0].cpu().numpy()

    @torch.inference_mode()
    def age_predictions(self, face_hidden, voice_hidden):
        # Native frozen heads, with fixed published age-bin midpoints. The final
        # open-ended 70+ bin uses 75 years. No age calibration is fitted.
        centers = torch.tensor([1,6,14.5,24.5,34.5,44.5,54.5,64.5,75], device=self.device) / 100
        f = torch.as_tensor(face_hidden, dtype=torch.float32, device=self.device)
        v = torch.as_tensor(voice_hidden, dtype=torch.float32, device=self.device)
        fa = self.face.classifier(f).softmax(dim=-1) @ centers
        va = self.voice.age(v).squeeze(-1)
        return fa.cpu().numpy(), va.cpu().numpy()


def add_age_predictions(pack_path, out, device="mps"):
    source = assert_training(pack_path)
    if source["feature_set"] != "official_plus_facenet_wavlm_age":
        raise ValueError("Requires the completed frozen demographic training pack")
    pack = load_pack(pack_path)
    cache = Path(pack_path).parent / (Path(pack_path).stem + "_cache")
    e = FrozenDemographics(device)
    face_age, voice_age = [], []
    for start in range(0, len(pack["labels"]), 256):
        ix = range(start, min(start+256, len(pack["labels"])))
        f = np.stack([np.load(cache / f"face_{i:05d}.npy") for i in ix])
        v = np.stack([np.load(cache / f"voice_{i:05d}.npy") for i in ix])
        fa, va = e.age_predictions(f, v)
        face_age.append(fa); voice_age.append(va)
    pack["face_age"], pack["voice_age"] = np.concatenate(face_age), np.concatenate(voice_age)
    np.savez_compressed(out, **pack)
    save_json(Path(out).with_suffix(".json"), {**source, "source_sha256": sha256(pack_path),
        "age_head_rule": "published_bin_midpoints_70plus75_no_calibration_v1", "pack_sha256": sha256(out)})


def extract_demographic_training(pack_path, out, device="mps"):
    source = assert_training(pack_path)
    if source["feature_set"] != "official_plus_facenet_wavlm":
        raise ValueError("Demographic enrichment requires the completed FaceNet/WavLM training pack")
    pack, out = load_pack(pack_path), Path(out)
    cache = out.parent / (out.stem + "_cache")
    cache.mkdir(parents=True, exist_ok=True)
    provenance = {"source_sha256": sha256(pack_path), "aux_registry_sha256": sha256(ROOT / "models/demographics_registry.json"),
                  "aux_preprocessing": "vit_standard_processor_cls_w2v2_rightpad1s_masked_mean_v1"}
    cp = cache / "provenance.json"
    if cp.exists() and json.loads(cp.read_text()) != provenance:
        raise ValueError("Auxiliary cache provenance changed")
    save_json(cp, provenance)
    e = FrozenDemographics(device)
    missing = [i for i in range(len(pack["labels"])) if not (cache / f"face_{i:05d}.npy").exists()]
    for start in tqdm(range(0, len(missing), 16), desc="training age face batches"):
        ix = missing[start:start+16]
        for i, embedding in zip(ix, e.face_embeddings([str(pack["face_paths"][i]) for i in ix])):
            save_npy(cache / f"face_{i:05d}.npy", embedding)
    groups = {}
    for i, p in enumerate(pack["voice_paths"]):
        if not (cache / f"voice_{i:05d}.npy").exists():
            info = sf.info(str(p))
            length = math.ceil(math.ceil(info.frames * 16000 / info.samplerate) / 16000)
            groups.setdefault(length, []).append(i)
    with tqdm(total=sum(len(g) for g in groups.values()), desc="training age voices") as progress:
        for seconds, indices in sorted(groups.items()):
            batch = 8 if seconds <= 12 else 4 if seconds <= 24 else 2 if seconds <= 40 else 1
            for start in range(0, len(indices), batch):
                ix = indices[start:start+batch]
                for i, embedding in zip(ix, e.voice_embeddings([str(pack["voice_paths"][i]) for i in ix])):
                    save_npy(cache / f"voice_{i:05d}.npy", embedding)
                progress.update(len(ix))
            if e.device.type == "mps":
                torch.mps.empty_cache()
    f = np.stack([np.load(cache / f"face_{i:05d}.npy") for i in range(len(pack["labels"]))])
    v = np.stack([np.load(cache / f"voice_{i:05d}.npy") for i in range(len(pack["labels"]))])
    pack["faces"] = np.concatenate([pack["faces"], unit(f)], axis=1)
    pack["voices"] = np.concatenate([pack["voices"], unit(v)], axis=1)
    np.savez_compressed(out, **pack)
    save_json(out.with_suffix(".json"), {**source, **provenance,
                "feature_set": "official_plus_facenet_wavlm_age", "pack_sha256": sha256(out)})
