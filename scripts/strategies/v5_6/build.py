"""Single source of truth: build + transplant clean v5_6 modules.

All structural invariants below are MEASURED on v5_6_ru (not assumed);
any upstream change fails loudly here instead of silently degrading
audio. Conventions follow build_acoustic.py (copy_ per tensor, now
with per-tensor shape asserts rather than a bare total count).
"""
import torch

from forward import Acoustic56, Main56, Tacotron56
from series import DurPredictor56, PitchPredictor56
from vocoder import Vocoder56


def _g(root, *names):
    m = root
    for n in names:
        m = dict(m.named_children())[n]
    return m


def _copy(dst, src, log):
    assert tuple(dst.shape) == tuple(src.shape), (tuple(dst.shape),
                                                  tuple(src.shape))
    dst.copy_(src)
    log.append(1)


def _transplant_fft(dst, src, log, tag, d_model, ffn, kernel, heads):
    assert bool(src.use_linear) is False and bool(src.pitch_pred) is False, tag
    assert tuple(_g(src, "conv1").weight.shape) == (ffn, d_model, kernel), tag
    oa = _g(src, "self_attn")
    assert tuple(oa.in_proj_weight.shape) == (3 * d_model, d_model), tag
    assert int(oa.num_heads) == heads, tag
    dst.attn.in_proj_weight.copy_(oa.in_proj_weight)
    dst.attn.in_proj_bias.copy_(oa.in_proj_bias)
    dst.attn.out_proj.weight.copy_(oa.out_proj.weight)
    dst.attn.out_proj.bias.copy_(oa.out_proj.bias)
    for _ in range(4):
        log.append(1)
    for sub in ("norm1", "conv1", "conv2", "norm2"):
        for par in ("weight", "bias"):
            _copy(getattr(getattr(dst, sub), par),
                  getattr(_g(src, sub), par), log)


def transplant_predictors(jm, log=None):
    lg = log if log is not None else []
    # Speaker count from the checkpoint (dur predictor's table leads;
    # pitch and tacotron tables must agree — asserted on copy below).
    n_spk = _g(jm, "dur_predictor", "dur_pred",
               ).speaker_embedding.weight.shape[0]
    dur = DurPredictor56(n_types=12, n_speakers=n_spk).eval()
    pitch = PitchPredictor56(n_types=6, n_speakers=n_spk).eval()
    for clean, key, sub, n_types, want_emph in (
            (dur, "dur_predictor", "dur_pred", 12, False),
            (pitch, "pitch_predictor", "pitch_pred", 6, True)):
        src = _g(jm, key, sub)
        dst = clean.inner
        assert _g(src, "speaker_embedding").weight.shape[0] == n_spk, key
        assert bool(src.utt_type_emb) is True, key
        assert bool(src.focus_emb) is True, key
        assert bool(src.emphasis_emb) is want_emph, key
        assert str(src.speaker_emb) == "pre", key
        _copy(dst.embedding.weight, _g(src, "embedding").weight, lg)
        _copy(dst.speaker_embedding.weight,
              _g(src, "speaker_embedding").weight, lg)
        te = _g(src, "type_embedding")
        assert tuple(te.weight.shape) == (n_types, 64), key
        _copy(dst.type_embedding.weight, te.weight, lg)
        _copy(dst.focus_embedding.weight,
              _g(src, "focus_embedding").weight, lg)
        _copy(dst.emphasis_embedding.weight,
              _g(src, "emphasis_embedding").weight, lg)
        # emotion_embedding is dead weight (no flag, absent from the
        # forward math) and is deliberately not transplanted.
        _copy(dst.pos.scale, _g(src, "transformer", "pos_encoder").scale, lg)
        _copy(dst.pos.pe, _g(src, "transformer", "pos_encoder").pe, lg)
        for i in range(4):
            _transplant_fft(dst.layers[i],
                            _g(src, "transformer", "layers", str(i)), lg,
                            "%s/layer%d" % (key, i), 64, 64, 9, 2)
        _copy(dst.lin.weight, _g(src, "lin").weight, lg)
        _copy(dst.lin.bias, _g(src, "lin").bias, lg)
        _copy(dst.norm.weight, _g(src, "transformer", "norm").weight, lg)
        _copy(dst.norm.bias, _g(src, "transformer", "norm").bias, lg)
        dst.use_type = True
        dst.use_focus = True
        dst.use_emphasis = False
    return dur, pitch


def transplant_tacotron(jm, log=None):
    lg = log if log is not None else []
    src = _g(jm, "tacotron")
    n_spk = _g(src, "speaker_embedding").weight.shape[0]
    t = Tacotron56(n_speakers=n_spk).eval()
    assert str(src.pitch_emb_type) == "default", src.pitch_emb_type
    assert float(src.pitch_strength) == 1.0, src.pitch_strength
    assert float(src.energy_strength) == 0.0, src.energy_strength
    _copy(t.encoder.embedding.weight, _g(src, "embedding").weight, lg)
    oenc = _g(src, "encoder")
    _copy(t.encoder.pos.scale, _g(oenc, "pos_encoder").scale, lg)
    _copy(t.encoder.pos.pe, _g(oenc, "pos_encoder").pe, lg)
    for i in range(4):
        _transplant_fft(t.encoder.layers[i],
                        _g(oenc, "layers", str(i)), lg, "enc/layer%d" % i,
                        128, 512, 9, 2)
    _copy(t.encoder.norm.weight, _g(oenc, "norm").weight, lg)
    _copy(t.encoder.norm.bias, _g(oenc, "norm").bias, lg)
    _copy(t.spk_emb.weight, _g(src, "speaker_embedding").weight, lg)
    pp = _g(src, "pitch_proj")
    assert tuple(pp.weight.shape) == (128, 1, 3), pp.weight.shape
    assert tuple(pp.padding) == (1,), pp.padding
    _copy(t.pitch_proj.weight, pp.weight, lg)
    _copy(t.pitch_proj.bias, pp.bias, lg)
    dec, odec = t.decoder, _g(src, "decoder")
    assert int(odec.shorten_factor) == 3, odec.shorten_factor
    _copy(dec.pos.scale, _g(odec, "pos_encoder").scale, lg)
    _copy(dec.pos.pe, _g(odec, "pos_encoder").pe, lg)
    ap = _g(odec, "downsample", "avg_pool")
    assert tuple(ap.kernel_size) == (3,) and tuple(ap.stride) == (3,)
    up = _g(odec, "upsample")
    assert tuple(_g(up, "proj").weight.shape) == (384, 128), \
        _g(up, "proj").weight.shape
    assert int(up.dim) == 128, up.dim
    _copy(dec.upsample.proj.weight, _g(up, "proj").weight, lg)
    _copy(dec.upsample.proj.bias, _g(up, "proj").bias, lg)
    for tag, lay, dst in (("pre", _g(odec, "pre_vanilla_layers", "0"),
                           dec.pre),
                          ("short0", _g(odec, "shorten_layers", "0"),
                           dec.short0),
                          ("short1", _g(odec, "shorten_layers", "1"),
                           dec.short1),
                          ("post", _g(odec, "post_vanilla_layers", "0"),
                           dec.post)):
        _transplant_fft(dst, lay, lg, "dec/%s" % tag, 128, 512, 9, 2)
    _copy(dec.norm.weight, _g(odec, "norm").weight, lg)
    _copy(dec.norm.bias, _g(odec, "norm").bias, lg)
    _copy(t.lin.weight, _g(src, "lin").weight, lg)
    _copy(t.lin.bias, _g(src, "lin").bias, lg)
    return t


def transplant_vocoder(jm, log=None):
    lg = log if log is not None else []
    v = Vocoder56().eval()
    src = _g(jm, "vocoder")
    bb = _g(src, "backbone")
    emb = _g(bb, "embed")
    assert tuple(emb.weight.shape) == (512, 192, 7), emb.weight.shape
    assert tuple(emb.kernel_size) == (7,) and tuple(emb.stride) == (1,), \
        (emb.kernel_size, emb.stride)
    assert tuple(emb.padding) == (3,) and tuple(emb.dilation) == (1,), \
        (emb.padding, emb.dilation)
    _copy(v.backbone.embed.weight, emb.weight, lg)
    _copy(v.backbone.embed.bias, emb.bias, lg)
    _copy(v.backbone.norm.weight, _g(bb, "norm").weight, lg)
    _copy(v.backbone.norm.bias, _g(bb, "norm").bias, lg)
    blocks = list(_g(bb, "convnext").children())
    assert len(blocks) == 8, len(blocks)
    for i, b in enumerate(blocks):
        d = v.backbone.blocks[i]
        dw = _g(b, "dwconv")
        assert tuple(dw.weight.shape) == (512, 1, 7), dw.weight.shape
        assert int(dw.groups) == 512, dw.groups
        assert tuple(dw.padding) == (3,), dw.padding
        _copy(d.dwconv.weight, dw.weight, lg)
        _copy(d.dwconv.bias, dw.bias, lg)
        _copy(d.norm.weight, _g(b, "norm").weight, lg)
        _copy(d.norm.bias, _g(b, "norm").bias, lg)
        assert tuple(_g(b, "pwconv1").weight.shape) == (1536, 512)
        _copy(d.pwconv1.weight, _g(b, "pwconv1").weight, lg)
        _copy(d.pwconv1.bias, _g(b, "pwconv1").bias, lg)
        assert tuple(_g(b, "pwconv2").weight.shape) == (512, 1536)
        _copy(d.pwconv2.weight, _g(b, "pwconv2").weight, lg)
        _copy(d.pwconv2.bias, _g(b, "pwconv2").bias, lg)
        _copy(d.gamma, b.gamma, lg)
    _copy(v.backbone.final_norm.weight,
          _g(bb, "final_layer_norm").weight, lg)
    _copy(v.backbone.final_norm.bias, _g(bb, "final_layer_norm").bias, lg)
    out = _g(src, "head", "out")
    assert tuple(out.weight.shape) == (2402, 512), out.weight.shape
    _copy(v.out.weight, out.weight, lg)
    _copy(v.out.bias, out.bias, lg)
    ist = _g(src, "head", "istft")
    assert (int(ist.n_fft), int(ist.hop_length), int(ist.win_length),
            str(ist.padding)) == (2400, 600, 2400, "same"), \
        (ist.n_fft, ist.hop_length, ist.win_length, ist.padding)
    _copy(v.istft.window, ist.window, lg)
    return v


def build_main(jm, log=None):
    """Assemble the fused clean main (torch-parity proof only).

    NOT exported as one graph: the fold-based ISTFT tail does not
    dynamo-export inside the fused graph (data-dependent guard). The
    exportable units are Acoustic56 (-> mel, dur) and the standalone
    Vocoder56 (-> audio); both halves follow the proven v5 paths."""
    t = transplant_tacotron(jm, log)
    v = transplant_vocoder(jm, log)
    return Main56(Acoustic56(t), v).eval()


def build_acoustic56(jm, log=None):
    """Assemble the exportable acoustic half (text -> mel, dur)."""
    return Acoustic56(transplant_tacotron(jm, log)).eval()
