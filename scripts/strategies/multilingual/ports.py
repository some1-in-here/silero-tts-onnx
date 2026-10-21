"""Build-time FP32 acoustic and HiFi-GAN ports for multilingual checkpoints.

Adapted by Jim Grimsby Jr. with AI assistance from Vladimir Govorov's
c62b8f9 Russian port. Original model architecture/weights: Silero Team.
See MULTILINGUAL-NOTICES.md and the repository LICENSE.
"""
from __future__ import annotations
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT.parent / "v5"
sys.path.insert(0, str(UPSTREAM))
logging.getLogger("torch.onnx._internal.exporter._registration").setLevel(logging.ERROR)
import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
from encoder import PositionalEncoding
from fftblock import FFTBlock
from layernorm import ExportLayerNorm
from predictors import SeriesPredictor, LengthRegulator
from decoder import HourGlassDecoder



def child(root, *names):
    for name in names:
        root = dict(root.named_children())[name]
    return root


def copy_values(destination, source, names=("weight", "bias")):
    for name in names:
        target, original = getattr(destination, name), getattr(source, name)
        assert target.shape == original.shape, (name, target.shape, original.shape)
        target.copy_(original)


def copy_position(destination, source):
    copy_values(destination, source, ("scale", "pe"))


def copy_block(destination, source):
    assert not source.use_linear
    copy_values(destination.attn, source.self_attn, ("in_proj_weight", "in_proj_bias"))
    copy_values(destination.attn.out_proj, source.self_attn.out_proj)
    for name in ["norm1", "norm2", "conv1", "conv2"]:
        copy_values(getattr(destination, name), getattr(source, name))


class Transformer(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        first = child(source.layers, "0")
        width = int(first.self_attn.embed_dim)
        self.pos = PositionalEncoding(width, int(source.pos_encoder.pe.shape[0]))
        self.layers = torch.nn.ModuleList([
            FFTBlock(width, int(layer.self_attn.num_heads), int(layer.conv1.weight.shape[0]),
                     int(layer.conv1.weight.shape[2])) for layer in source.layers.children()])
        self.norm = ExportLayerNorm(width)
        copy_position(self.pos, source.pos_encoder)
        for target, original in zip(self.layers, source.layers.children()):
            copy_block(target, original)
        copy_values(self.norm, source.norm)

    def forward(self, value, mask):
        value = self.pos(value.transpose(0, 1)).transpose(0, 1)
        for layer in self.layers:
            value = layer(value, mask)
        return self.norm(value)


def predictor(source):
    e = source.embedding.weight
    s = source.speaker_embedding.weight
    first = child(source.transformer.layers, "0")
    result = SeriesPredictor(n_symbols=e.shape[0], n_speakers=s.shape[0], d_model=e.shape[1],
        n_heads=int(first.self_attn.num_heads), n_layers=len(list(source.transformer.layers.children())),
        ffn_dim=first.conv1.weight.shape[0], kernel=first.conv1.weight.shape[2],
        pe_max=source.transformer.pos_encoder.pe.shape[0])
    copy_values(result.embedding, source.embedding, ("weight",))
    copy_values(result.speaker_embedding, source.speaker_embedding, ("weight",))
    copy_position(result.pos, source.transformer.pos_encoder)
    for target, original in zip(result.layers, source.transformer.layers.children()):
        copy_block(target, original)
    copy_values(result.norm, source.transformer.norm)
    copy_values(result.lin, source.lin)
    return result


class EnglishAcoustic(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        taco = original.tacotron
        e, s = taco.embedding.weight, taco.speaker_embedding.weight
        self.embedding = torch.nn.Embedding(*e.shape)
        self.speaker_embedding = torch.nn.Embedding(*s.shape)
        copy_values(self.embedding, taco.embedding, ("weight",))
        copy_values(self.speaker_embedding, taco.speaker_embedding, ("weight",))
        self.encoder = Transformer(taco.encoder)
        self.dur = predictor(original.dur_predictor.dur_pred)
        self.pitch = predictor(original.pitch_predictor.pitch_pred)
        self.register_buffer("pitch_shift", torch.tensor(original.mean_std_coef, dtype=torch.float32))
        self.pitch_strength = float(taco.pitch_strength)
        self.pitch_proj = torch.nn.Conv1d(taco.pitch_proj.weight.shape[1], taco.pitch_proj.weight.shape[0],
                                        taco.pitch_proj.weight.shape[2], padding=1)
        copy_values(self.pitch_proj, taco.pitch_proj)
        self.len_reg = LengthRegulator()
        d = taco.decoder
        first = child(d.pre_vanilla_layers, "0")
        self.decoder = HourGlassDecoder(d_model=e.shape[1], n_heads=int(first.self_attn.num_heads),
            ffn_dim=first.conv1.weight.shape[0], kernel=first.conv1.weight.shape[2],
            pe_max=d.pos_encoder.pe.shape[0], shorten=int(d.shorten_factor))
        assert len(list(d.pre_vanilla_layers.children())) == 1
        assert len(list(d.post_vanilla_layers.children())) == 1
        assert len(list(d.shorten_layers.children())) == 2
        copy_position(self.decoder.pos, d.pos_encoder)
        copy_values(self.decoder.upsample.proj, d.upsample.proj)
        for name, original_layer in [
            ("pre", child(d.pre_vanilla_layers, "0")), ("short0", child(d.shorten_layers, "0")),
            ("short1", child(d.shorten_layers, "1")), ("post", child(d.post_vanilla_layers, "0"))]:
            copy_block(getattr(self.decoder, name), original_layer)
        copy_values(self.decoder.norm, d.norm)
        self.lin = torch.nn.Linear(taco.lin.weight.shape[1], taco.lin.weight.shape[0])
        copy_values(self.lin, taco.lin)

    def forward(self, sequence, speaker_ids, durs_rate, pitch_coefs):
        mask = sequence == 0
        raw = self.dur(sequence, speaker_ids, mask, 1.0).squeeze(-1)
        duration = torch.round(torch.clamp(torch.exp(raw)-1, min=0))/durs_rate
        pitch = self.pitch(sequence, speaker_ids, mask, 1.0).transpose(1, 2)
        pc = pitch_coefs.unsqueeze(1)
        shifted = torch.where(pc == 0, torch.ones_like(pc), pc)
        pitch = pitch*pc + (shifted-1)*self.pitch_shift[speaker_ids].reshape(-1, 1, 1)
        encoded = self.encoder(self.embedding(sequence), mask)
        encoded = encoded+self.speaker_embedding(speaker_ids).unsqueeze(1)
        encoded = encoded+self.pitch_proj(pitch).transpose(1, 2)*self.pitch_strength
        expanded = self.len_reg(encoded, duration).reshape(1, -1, self.embedding.embedding_dim)
        mel = self.lin(self.decoder(expanded, None)).transpose(1, 2)
        return mel, duration


def copy_conv(source, transpose=False):
    shape = source.weight.shape
    cls = torch.nn.ConvTranspose1d if transpose else torch.nn.Conv1d
    kw = dict(stride=tuple(source.stride), padding=tuple(source.padding),
              dilation=tuple(source.dilation), groups=int(source.groups), bias=source.bias is not None)
    if transpose:
        kw["output_padding"] = tuple(source.output_padding)
        channels = (shape[0], shape[1]*int(source.groups))
    else:
        channels = (shape[1]*int(source.groups), shape[0])
    result = cls(*channels, shape[2], **kw)
    copy_values(result, source, ("weight", "bias") if source.bias is not None else ("weight",))
    return result


class ResBlock(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.convs = torch.nn.ModuleList(copy_conv(c) for c in source.convs.children())
        self.slope = float(source.LRELU_SLOPE)

    def forward(self, value):
        for conv in self.convs:
            value = value + conv(torch.nn.functional.leaky_relu(value, self.slope))
        return value


class Vocoder(torch.nn.Module):
    """Faithful 48-kHz HiFi-GAN/PQMF port; avoids old TorchScript Optional typing."""
    def __init__(self, source):
        super().__init__()
        self.pre = copy_conv(child(source.conv_pre, "1"))
        self.ups = torch.nn.ModuleList(copy_conv(c, transpose=True) for c in source.ups_u.children())
        self.blocks = torch.nn.ModuleList(ResBlock(c) for c in source.resblocks_u.children())
        self.post = copy_conv(source.conv_post_u)
        self.slope = float(source.LRELU_SLOPE)
        self.kernels = int(source.num_kernels)
        assert len(self.blocks) == len(self.ups)*self.kernels
        assert source.multiband and source.resample == 1
        self.bands = int(source.pqmf_layer.N)
        self.pad = int(source.pqmf_layer.taps)//2
        self.register_buffer("updown_filter", source.pqmf_layer.updown_filter.clone())
        self.register_buffer("synthesis_filter", source.pqmf_layer.G.clone())

    def forward(self, mel):
        value = self.pre(mel)
        for i, up in enumerate(self.ups):
            value = up(torch.nn.functional.leaky_relu(value, self.slope))
            total = self.blocks[i*self.kernels](value)
            for j in range(1, self.kernels):
                total = total + self.blocks[i*self.kernels+j](value)
            value = total/self.kernels
        value = self.post(torch.nn.functional.leaky_relu(value, 0.01))
        value = torch.nn.functional.conv_transpose1d(value, self.updown_filter*self.bands, stride=self.bands)
        value = torch.nn.functional.conv1d(value, self.synthesis_filter, padding=self.pad)
        return torch.where(torch.amax(torch.abs(value)) <= 1, value, torch.tanh(value))


def make_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
