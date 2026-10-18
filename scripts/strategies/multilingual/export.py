"""Export trusted Silero v3/v4/v5 checkpoints with mandatory original-waveform gates."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback
from silero_onnx.multilingual import MultilingualRuntime as Runtime
from vocoder_ports import make_vocoder
from strip_onnx_diagnostics import strip_diagnostics

ROOT = Path(__file__).resolve().parents[3]
import ports as base

torch, np, ort, onnx, sf = base.torch, base.np, base.ort, base.onnx, base.sf
from istft_model import RealISTFT


def unpack(model):
    if hasattr(model, "unpack_q_model"):
        model.unpack_q_model()
        model.q_model_unpacked = True
    part = model.packages[0] if hasattr(model, "packages") else model
    if part is not model and hasattr(part, "unpack_q_model"):
        part.unpack_q_model()
        part.q_model_unpacked = True
    system = part.models[0] if hasattr(part, "models") else part.model
    original = system.tts_model if hasattr(system,"tts_model") else system
    return part, system, original


def frontend_data(model, part, system):
    table = dict(getattr(part, "symbol_to_id", getattr(system, "symbol_to_id", {})))
    if not table:
        table = {c:i for i,c in enumerate(part.symbols)}
    speakers = dict(part.speaker_to_ids[0]) if hasattr(part,"speaker_to_ids") else dict(part.speaker_to_id)
    return {"symbols":part.symbols,"symbol_to_id":table,"speaker_to_id":speakers,
        "symb_ascii_dict":getattr(part,"symb_ascii_dict",{}),"ext_alph":getattr(part,"ext_alph",None),
        "alphabet":list(getattr(part,"alphabet",[])),"sos":getattr(system,"sos_token","|"),
        "eos":getattr(system,"eos_token","~"),"phons":getattr(part,"phons",False),
        "modern":hasattr(model,"packages"),"sample_rate":48000}


def language_for(speaker, candidate):
    if candidate.startswith("v3_"):
        return candidate[-2:]
    if candidate == "v4_indic":
        return speaker.split("_")[0]
    if candidate == "v4_cyrillic":
        aliases={"b_bulb":"bg","b_bulc":"bg","b_ru":"ru","b_tat":"tat","marat_tt":"tat",
                 "b_kalmyk":"xal","kalmyk_erdni":"xal","kalmyk_delghir":"xal", "b_bashkir":"bak",
                 "b_sah":"sah","b_myv":"erz","b_kjh":"kjh","b_krc":"krc","b_cv":"chv",
                 "cv_ekaterina":"chv","b_udm":"udm","b_lez":"lez","b_oss":"oss","b_ava":"ava"}
        return aliases.get(speaker,"kaz" if speaker.startswith("kz_") else speaker[2:])
    return speaker.split("_")[0]


def feeds(part, system, text, speaker, lang):
    ids = part.get_speakers(speaker)
    if isinstance(ids, tuple):
        ids, mid = ids
        assert mid == 0
    kwargs = {"text":text,"ssml":False,"speaker_ids":ids}
    if "lang" in inspect.signature(part.prepare_tts_model_input).parameters:
        kwargs["lang"] = lang
    sents, cleaned, pauses, rates, pitches, spids = part.prepare_tts_model_input(**kwargs)
    merge = part if hasattr(part,"merge_batch_model") else system
    merged = merge.merge_batch_model(sents, pauses, rates, pitches)
    seq, pause, rate, pitch = merged[:4]
    assert not pause and not getattr(part,"accentor",None), "Needs additional original frontend components"
    return seq.long(), spids.long(), rate, pitch


class Acoustic(base.EnglishAcoustic):
    def __init__(self, source, modern):
        super().__init__(source)
        self.modern = modern
        assert not hasattr(source.tacotron,"pitch_emb_type") or source.tacotron.pitch_emb_type == "default"
        assert not getattr(source, "energy_predictor", None)

    def forward(self, sequence, speaker_ids, durs_rate, pitch_coefs):
        mask = sequence == 0
        raw = self.dur(sequence,speaker_ids,mask,1.0).squeeze(-1)
        duration = torch.round(torch.clamp(torch.exp(raw)-1,min=0))
        if self.modern:
            i = torch.arange(sequence.shape[1],device=sequence.device).unsqueeze(0)
            duration = torch.where(i == 0,torch.clamp(duration,max=5),duration)
            duration = torch.round(duration/durs_rate)
            duration = torch.where(i == 0,torch.clamp(duration,max=5),duration)
            duration = torch.where(i == sequence.shape[1]-1,torch.clamp(duration,max=7),duration)
            duration = torch.where(i == sequence.shape[1]-2,torch.full_like(duration,13),duration)
            duration = torch.where(i == sequence.shape[1]-3,torch.clamp(duration,max=13),duration)
        else:
            duration = duration/durs_rate
        pitch = self.pitch(sequence,speaker_ids,mask,1.0).transpose(1,2)
        if self.modern:
            pitch = torch.where(torch.abs(pitch)<0.001,torch.zeros_like(pitch),pitch)
        pc = pitch_coefs.unsqueeze(1)
        shifted = torch.where(pc == 0,torch.ones_like(pc),pc)
        pitch = pitch*pc+(shifted-1)*self.pitch_shift[speaker_ids].reshape(-1,1,1)
        encoded = self.encoder(self.embedding(sequence),mask)+self.speaker_embedding(speaker_ids).unsqueeze(1)
        encoded = encoded+self.pitch_proj(pitch).transpose(1,2)*self.pitch_strength
        expanded = self.len_reg(encoded,duration).reshape(1,-1,self.embedding.embedding_dim)
        return self.lin(self.decoder(expanded,None)).transpose(1,2),duration


def linear(source):
    m = torch.nn.Linear(source.weight.shape[1],source.weight.shape[0])
    base.copy_values(m,source)
    return m

def norm(source):
    m = torch.nn.LayerNorm(tuple(source.normalized_shape),eps=float(source.eps))
    base.copy_values(m,source)
    return m

class VocosBlock(torch.nn.Module):
    def __init__(self,source):
        super().__init__()
        self.dw = base.copy_conv(source.dwconv)
        self.norm = norm(source.norm)
        self.pw1,self.pw2 = linear(source.pwconv1),linear(source.pwconv2)
        self.register_buffer("gamma",source.gamma.clone())
    def forward(self,x):
        y=self.norm(self.dw(x).transpose(1,2))
        y=self.pw2(torch.nn.functional.gelu(self.pw1(y)))*self.gamma
        return x+y.transpose(1,2)

class Vocos(torch.nn.Module):
    def __init__(self,source):
        super().__init__()
        b=source.backbone
        self.embed=base.copy_conv(b.embed)
        self.norm,self.final=norm(b.norm),norm(b.final_layer_norm)
        self.blocks=torch.nn.ModuleList(VocosBlock(v) for v in b.convnext.children())
        self.out=linear(source.head.out)
    def forward(self,mel):
        x=self.norm(self.embed(mel).transpose(1,2)).transpose(1,2)
        for block in self.blocks:
            x=block(x)
        x=self.out(self.final(x.transpose(1,2))).transpose(1,2)
        mag,phase=torch.chunk(x,2,1)
        return torch.clamp(torch.exp(mag),max=100),torch.cos(phase),torch.sin(phase)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("id", choices=[
        "v3_en", "v3_de", "v3_es", "v3_fr", "v4_indic", "v4_cyrillic",
        "v5_turkic", "v5_caucasian", "v5_cis_base_nostress", "v5_cis_ext"])
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trusted official torch.package file (loading it executes Python)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="New/empty model directory; existing data is never overwritten")
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    out=args.out_dir
    if out.exists() and any(out.iterdir()):
        parser.error("--out-dir must be empty to avoid mixing stale graphs or evidence")
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=args.checkpoint
    with checkpoint.open("rb") as stream:
        source_sha256=hashlib.file_digest(stream,"sha256").hexdigest()
    inventory=json.loads((ROOT/"scripts/multilingual-checkpoints.json").read_text(encoding="utf-8"))
    expected=inventory[args.id]
    if source_sha256 != expected["sha256"]:
        raise ValueError("Checkpoint hash differs from the reviewed inventory; review before export")
    model=torch.package.PackageImporter(str(checkpoint)).load_pickle("tts_models","model")
    part,system,original=unpack(model)
    modern=hasattr(model,"packages")
    data=frontend_data(model,part,system)
    assert not data["phons"], "Phoneme-specific frontend needs a separate port"
    data["id"]=args.id
    data["source_sha256"]=source_sha256
    data["source_url"]=expected["url"]
    data["model_license"]=expected["license"]
    data["format"]="silero-multilingual-1"
    samples=json.loads((ROOT / "scripts/multilingual-samples.json").read_text(encoding="utf-8"))
    voices=[s for s in model.speakers if s != "random"]
    cases=[]
    seen=set()
    for speaker in voices:
        lang=language_for(speaker,args.id)
        if lang in samples and lang not in seen:
            seen.add(lang)
            cases.append({"speaker":speaker,"lang":lang,**samples[lang]})
    assert cases, "No language-matched sample available"
    chosen=cases[:3]
    first=cases[0]
    chosen=chosen+[{**first,"text":first["text"].split()[0]+".","scope":"short-input parity"}]
    if len(cases)==1 and len(voices)>1:
        chosen.append({**first,"speaker":voices[1],"scope":"second-voice parity"})
    acoustic=Acoustic(original,modern).eval()
    results=[]
    tensors=[]
    for index,row in enumerate(chosen):
        inp=feeds(part,system,row["text"],row["speaker"],row["lang"])
        pure=Runtime.prepare_data(data,row["text"],row["speaker"],row["lang"])
        for name,value in zip(["sequence","speaker_ids","durs_rate","pitch_coefs"],inp):
            np.testing.assert_array_equal(pure[name],value.numpy())
        mel,dur=acoustic(*inp)
        kwargs={"text":row["text"],"speaker":row["speaker"],"sample_rate":48000}
        if "lang" in inspect.signature(model.apply_tts).parameters:
            kwargs["lang"]=row["lang"]
        reference=model.apply_tts(**kwargs).reshape(-1)
        # Gate against the original complete graph, not only the reconstructed modules.
        native,native_dur=original(sequence=inp[0],speaker_ids=inp[1],sr=48000,
            symb_durs={},durs_rate=inp[2].clone(),pitch_coefs=inp[3].clone(),device="cpu")
        native=native.reshape(-1)
        assert torch.equal(reference,native), "Frontend/direct original path mismatch"
        wave=original.vocoder(mel,48000).reshape(-1)
        native_pitch=original.pitch_predictor(inp[0],inp[1],inp[0]==0)
        native_pitch=original.update_pitch_coef(native_pitch,inp[3].clone(),inp[1])
        native_mel=original.tacotron(inp[0],inp[1],inp[0]==0,dur.clone(),native_pitch)
        print("MEL_PARITY",float((native_mel-mel).abs().max()),flush=True)
        if native_dur.shape == dur.shape:
            dmax=float((dur-native_dur).abs().max())
        else:
            # Older v3 returns total waveform lengths, not per-symbol durations.
            expected_dur=torch.round(torch.clamp(torch.exp(original.dur_predictor(inp[0],inp[1],inp[0]==0,1.0))-1,min=0))/inp[2]
            dmax=float((dur-expected_dur).abs().max())
            assert int(dur.sum())*600 == int(native_dur.reshape(-1)[0])
        wmax=float((wave-reference).abs().max()) if wave.shape==reference.shape else None
        print("EAGER_PARITY",json.dumps({"id":args.id,"voice":row["speaker"],"dur_diff":dmax,"wave_diff":wmax}),flush=True)
        assert dmax==0 and wmax is not None and wmax<0.005
        tensors.append((row,inp,mel,reference,dur))
        sf.write(out/f"reference-{index}.wav",reference.numpy(),48000,subtype="PCM_16")
    data["cases"]=cases
    data["all_speakers"]=voices
    (out/"frontend.json").write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    ap=out/"acoustic.onnx"
    torch.onnx.export(acoustic,tensors[0][1],str(ap),dynamo=True,opset_version=18,external_data=False,
        input_names=["sequence","speaker_ids","durs_rate","pitch_coefs"],output_names=["mel","duration"],
        dynamic_shapes=({1:torch.export.Dim("text",min=3,max=1002)},{},{1:torch.export.Dim("text",min=3,max=1002)},{1:torch.export.Dim("text",min=3,max=1002)}))
    vp=out/"vocoder.onnx"
    spectral=hasattr(original.vocoder,"backbone")
    if spectral:
        voc=Vocos(original.vocoder).eval()
        fft=original.vocoder.head.istft
        istft=RealISTFT(n_fft=int(fft.n_fft),hop_length=int(fft.hop_length),win_length=int(fft.win_length)).eval()
        sv=voc(tensors[0][2])
        wave=istft(*sv).reshape(-1)
        expected=original.vocoder(tensors[0][2],48000).reshape(-1)
        vmax=float((wave-expected).abs().max())
        print("VOCODER_PARITY",vmax,flush=True)
        assert vmax<0.001
        torch.onnx.export(voc,(tensors[0][2],),str(vp),dynamo=False,opset_version=17,
            input_names=["mel"],output_names=["mag","x","y"],
            dynamic_axes={k:{2:"frames"} for k in ["mel","mag","x","y"]})
        ip=out/"istft.onnx"
        frame_dim=torch.export.Dim("frames",min=1,max=4096)
        torch.onnx.export(istft,sv,str(ip),dynamo=True,opset_version=18,external_data=False,
            input_names=["mag","x","y"],output_names=["audio"],
            dynamic_shapes=({2:frame_dim},{2:frame_dim},{2:frame_dim}))
    else:
        voc=make_vocoder(original.vocoder,base).eval()
        for row,inp,mel,reference,dur in tensors:
            expected=original.vocoder(mel,48000).reshape(-1)
            diff=float((voc(mel).reshape(-1)-expected).abs().max())
            print("VOCODER_PARITY",row["speaker"],diff,flush=True)
            assert diff<0.0001
        torch.onnx.export(voc,(tensors[0][2],),str(vp),dynamo=False,opset_version=17,
            input_names=["mel"],output_names=["audio"],dynamic_axes={"mel":{2:"frames"},"audio":{2:"samples"}})
    data["spectral"]=spectral
    for p in out.glob("*.onnx"):
        strip_diagnostics(p)
        onnx.checker.check_model(str(p))
    sessions=[base.make_session(p) for p in ([ap,vp,ip] if spectral else [ap,vp])]
    for index,(row,inp,mel,reference,dur) in enumerate(tensors):
        feeds_np=dict(zip(["sequence","speaker_ids","durs_rate","pitch_coefs"],[x.numpy() for x in inp]))
        actual_mel,actual_dur=sessions[0].run(None,feeds_np)
        waves=sessions[1].run(None,{"mel":actual_mel})
        audio=sessions[2].run(None,dict(zip(["mag","x","y"],waves)))[0].reshape(-1) if spectral else waves[0].reshape(-1)
        ref=reference.numpy()
        diff=float(np.max(np.abs(audio-ref))) if audio.shape==ref.shape else None
        dmax=float(np.max(np.abs(actual_dur-dur.numpy())))
        result={**row,"samples_reference":len(ref),"samples_onnx":len(audio),"wave_max_diff":diff,"duration_max_diff":dmax,
                "finite":bool(np.isfinite(audio).all())}
        print("ONNX_PARITY",json.dumps(result,ensure_ascii=True),flush=True)
        assert diff is not None and diff<0.005 and dmax==0 and result["finite"]
        sf.write(out/f"converted-{index}.wav",audio,48000,subtype="PCM_16")
        results.append(result)
    data["graphs"]={p.name:{"bytes":p.stat().st_size,"sha256":hashlib.file_digest(p.open("rb"),"sha256").hexdigest()} for p in out.glob("*.onnx")}
    (out/"frontend.json").write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"export-parity.json").write_text(json.dumps({"passed":True,"cases":results,"graphs":data["graphs"]},indent=2,ensure_ascii=False),encoding="utf-8")
    print("EXPORT_PASSED",args.id,flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
