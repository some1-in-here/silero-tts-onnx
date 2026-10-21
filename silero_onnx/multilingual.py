"""Torch-free low-level runtime for fixed-voice v3/v4/v5 exports.

Plain model text only: no automatic numbers, letter names, SSML, stress
insertion or native-script Indic romanization. See docs/multilingual.md.
Adapted by Jim Grimsby Jr. with AI assistance; original ports by Vladimir
Govorov and model architecture/weights by Silero Team. Repository LICENSE.
"""
import json
from pathlib import Path
import re
import numpy as np
import onnxruntime as ort

class MultilingualRuntime:
    def __init__(self, directory, lower_memory=False):
        self.directory=Path(directory)
        self.data=json.loads((self.directory/"frontend.json").read_text(encoding="utf-8"))
        if self.data.get("sample_rate") != 48000:
            raise ValueError("This runtime supports only the validated 48000-Hz exports")
        opts=ort.SessionOptions()
        opts.intra_op_num_threads=2
        opts.inter_op_num_threads=1
        opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        opts.log_severity_level=3
        opts.enable_cpu_mem_arena=not lower_memory
        opts.enable_mem_pattern=not lower_memory
        self.acoustic=ort.InferenceSession(str(self.directory/"acoustic.onnx"),opts,providers=["CPUExecutionProvider"])
        self.vocoder=ort.InferenceSession(str(self.directory/"vocoder.onnx"),opts,providers=["CPUExecutionProvider"])
        # ORT 1.23 cannot execute FP64 FusedMatMul emitted by extended
        # iSTFT optimization. BASIC preserves FP64 math without that fusion.
        opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.istft=ort.InferenceSession(str(self.directory/"istft.onnx"),opts,providers=["CPUExecutionProvider"]) if self.data["spectral"] else None

    @staticmethod
    def prepare_data(data,text,speaker,lang=None):
        if speaker not in data["speaker_to_id"] or speaker=="random":
            raise ValueError("Unknown fixed voice: "+speaker)
        if data["phons"]:
            raise NotImplementedError("Phonetic input not ported")
        if data["modern"]:
            text=text.strip()
            text=re.sub(r"\n",lambda m:" " if m.start()>0 and text[m.start()-1] in ",;:.!?" else ". ",text)
        text=text.lower().replace("—","–").replace("‑","-")
        if data["modern"]:
            ext=data["ext_alph"] or {}
            lang=lang or speaker.split("_")[0]
            letters={c for c in text if c.isalpha()}
            if letters and not letters<=set(data["alphabet"]) and lang in ext:
                text="".join(ext[lang].get(c,c) for c in text)
            text=re.sub(r"[^{}]".format(data["symbols"][3:]+r"\^"),"",text)
            text=text.replace("^"," ^")
            text=re.sub(r"\s+"," ",text).strip().replace("^ ","^")
            text=re.sub(r"\^+","^",text).strip()
            if text.endswith("^"):
                text=text[:-1].strip()
            clean=re.sub(r"[^а-я\- ]","",text)
        else:
            text=re.sub(r"[^{}]".format(data["symbols"][3:]),"",text)
            text=re.sub(r"\s+"," ",text).strip()
            text="".join(data["symb_ascii_dict"].get(c,c) for c in text)
            clean=re.sub(r"[^a-z1-9\- ]","",text)
        if not clean.replace(" ",""):
            raise ValueError("Original frontend has no supported text")
        seq=np.array([[data["symbol_to_id"][c] for c in data["sos"]+text+data["eos"]]],dtype=np.int64)
        if seq.shape[1] > 1002:
            raise ValueError("Input exceeds the exported 1000-symbol text limit; chunk it first")
        return {"sequence":seq,"speaker_ids":np.array([data["speaker_to_id"][speaker]],dtype=np.int64),
                "durs_rate":np.ones(seq.shape,dtype=np.float32),"pitch_coefs":np.ones(seq.shape,dtype=np.float32)}

    def synthesize(self,text,speaker,lang=None):
        feeds=self.prepare_data(self.data,text,speaker,lang)
        mel,durs=self.acoustic.run(None,feeds)
        values=self.vocoder.run(None,{"mel":mel})
        audio=self.istft.run(None,dict(zip(["mag","x","y"],values)))[0] if self.istft else values[0]
        audio=audio.reshape(-1)
        if not np.isfinite(audio).all():
            raise ValueError("Model returned non-finite audio")
        return audio,durs
