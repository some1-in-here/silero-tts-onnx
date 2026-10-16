"""Single source of truth: build + transplant clean AcousticClean.

Mirrors the frozen, passing transplant (verified by the torch-vs-pipeline
dur gate in export_acoustic.py). Any change here must re-pass that gate
before export.
"""
import torch

from acoustic import AcousticClean


def _g(root, *names):
    m = root
    for n in names:
        m = dict(m.named_children())[n]
    return m


def build_acoustic(jit_m, log=None):
    def lg(s):
        if log is not None:
            log.append(s)

    taco = _g(jit_m, "tacotron")
    # Speaker count and prosody table come from the checkpoint itself
    # (never hardcoded): every speaker table in the graph must agree on
    # N, and copy_ re-verifies each shape below.
    n_spk = _g(taco, "speaker_embedding").weight.shape[0]
    assert tuple(_g(taco, "speaker_embedding").weight.shape) == (n_spk, 128)
    mean_std = list(jit_m.mean_std_coef)
    assert len(mean_std) == n_spk, (len(mean_std), n_spk)
    new = AcousticClean(n_speakers=n_spk, mean_std=mean_std).eval()
    lg(f"speakers: N={n_spk} (tacotron/predictors/mean_std agree)")
    n = 0
    with torch.no_grad():
        new.encoder.embedding.weight.copy_(_g(taco, "embedding").weight)
        n += 1
        oenc = _g(taco, "encoder")
        new.encoder.pos.scale.copy_(_g(oenc, "pos_encoder").scale)
        new.encoder.pos.pe.copy_(_g(oenc, "pos_encoder").pe)
        n += 2
        for i in range(4):
            lay = _g(oenc, "layers", str(i))
            assert bool(lay.use_linear) is False and bool(lay.pitch_pred) is False
            assert tuple(_g(lay, "conv1").weight.shape) == (512, 128, 9)
            oa = _g(lay, "self_attn")
            assert tuple(oa.in_proj_weight.shape) == (384, 128) and int(oa.num_heads) == 2
            blk = new.encoder.layers[i]
            blk.attn.in_proj_weight.copy_(oa.in_proj_weight)
            blk.attn.in_proj_bias.copy_(oa.in_proj_bias)
            blk.attn.out_proj.weight.copy_(oa.out_proj.weight)
            blk.attn.out_proj.bias.copy_(oa.out_proj.bias)
            for sub in ("norm1", "conv1", "conv2", "norm2"):
                for par in ("weight", "bias"):
                    getattr(getattr(blk, sub), par).copy_(getattr(_g(lay, sub), par))
            n += 12
        for par in ("weight", "bias"):
            getattr(new.encoder.norm, par).copy_(getattr(_g(oenc, "norm"), par))
            n += 1
        for mod, omod in ((new.dur.inner, _g(jit_m, "dur_predictor", "dur_pred")),
                          (new.pitch.inner, _g(jit_m, "pitch_predictor", "pitch_pred"))):
            mod.embedding.weight.copy_(_g(omod, "embedding").weight)
            n += 1
            assert _g(omod, "speaker_embedding").weight.shape[0] == n_spk
            mod.speaker_embedding.weight.copy_(_g(omod, "speaker_embedding").weight)
            n += 1
            mod.pos.scale.copy_(_g(omod, "transformer", "pos_encoder").scale)
            mod.pos.pe.copy_(_g(omod, "transformer", "pos_encoder").pe)
            n += 2
            for i in range(4):
                lay = _g(omod, "transformer", "layers", str(i))
                assert tuple(_g(lay, "conv1").weight.shape) == (64, 64, 9)
                oa = _g(lay, "self_attn")
                assert tuple(oa.in_proj_weight.shape) == (192, 64) and int(oa.num_heads) == 2
                blk = mod.layers[i]
                blk.attn.in_proj_weight.copy_(oa.in_proj_weight)
                blk.attn.in_proj_bias.copy_(oa.in_proj_bias)
                blk.attn.out_proj.weight.copy_(oa.out_proj.weight)
                blk.attn.out_proj.bias.copy_(oa.out_proj.bias)
                for sub in ("norm1", "conv1", "conv2", "norm2"):
                    for par in ("weight", "bias"):
                        getattr(getattr(blk, sub), par).copy_(getattr(_g(lay, sub), par))
                n += 12
            mod.lin.weight.copy_(_g(omod, "lin").weight)
            mod.lin.bias.copy_(_g(omod, "lin").bias)
            n += 2
            mod.norm.weight.copy_(_g(omod, "transformer", "norm").weight)
            mod.norm.bias.copy_(_g(omod, "transformer", "norm").bias)
            n += 2
        assert tuple(_g(taco, "speaker_embedding").weight.shape) == (n_spk, 128)
        new.spk_emb.weight.copy_(_g(taco, "speaker_embedding").weight)
        n += 1
        pp = _g(taco, "pitch_proj")
        assert tuple(pp.weight.shape) == (128, 1, 3) and tuple(pp.padding) == (1,)
        new.pitch_proj.weight.copy_(pp.weight)
        new.pitch_proj.bias.copy_(pp.bias)
        n += 2
        assert tuple(_g(taco, "lin").weight.shape) == (192, 128)
        new.lin.weight.copy_(_g(taco, "lin").weight)
        new.lin.bias.copy_(_g(taco, "lin").bias)
        n += 2
        dec, odec = new.decoder, _g(taco, "decoder")
        assert int(odec.shorten_factor) == 3
        dec.pos.scale.copy_(_g(odec, "pos_encoder").scale)
        dec.pos.pe.copy_(_g(odec, "pos_encoder").pe)
        n += 2
        ap = _g(odec, "downsample", "avg_pool")
        assert tuple(ap.kernel_size) == (3,) and tuple(ap.stride) == (3,)
        up = _g(odec, "upsample")
        assert tuple(_g(up, "proj").weight.shape) == (384, 128) and int(up.dim) == 128
        dec.upsample.proj.weight.copy_(_g(up, "proj").weight)
        dec.upsample.proj.bias.copy_(_g(up, "proj").bias)
        n += 2
        for tag, orig_lay in (("pre", _g(odec, "pre_vanilla_layers", "0")),
                              ("short0", _g(odec, "shorten_layers", "0")),
                              ("short1", _g(odec, "shorten_layers", "1")),
                              ("post", _g(odec, "post_vanilla_layers", "0"))):
            assert bool(orig_lay.use_linear) is False and bool(orig_lay.pitch_pred) is False
            oa = _g(orig_lay, "self_attn")
            assert tuple(oa.in_proj_weight.shape) == (384, 128) and int(oa.num_heads) == 2
            blk = {"pre": dec.pre, "short0": dec.short0,
                   "short1": dec.short1, "post": dec.post}[tag]
            blk.attn.in_proj_weight.copy_(oa.in_proj_weight)
            blk.attn.in_proj_bias.copy_(oa.in_proj_bias)
            blk.attn.out_proj.weight.copy_(oa.out_proj.weight)
            blk.attn.out_proj.bias.copy_(oa.out_proj.bias)
            for sub in ("norm1", "conv1", "conv2", "norm2"):
                for par in ("weight", "bias"):
                    getattr(getattr(blk, sub), par).copy_(getattr(_g(orig_lay, sub), par))
            n += 12
        for par in ("weight", "bias"):
            getattr(dec.norm, par).copy_(getattr(_g(odec, "norm"), par))
            n += 1
    lg(f"transplanted {n} tensors")
    assert n == 224, n
    assert new.pitch_emb_type == "default" and new.pitch_strength == 1.0
    return new
