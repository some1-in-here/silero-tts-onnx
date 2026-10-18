# Multilingual evaluation: scope and limitations

The retained conversion evaluation was performed on Windows with an Intel
Core i5-8400, two inference threads and one inter-op thread. Export used
PyTorch 2.14.0 CPU; the model runtime is torch-free. The lab used ONNX Runtime
1.29/1.30; the separate Sonata portable integration used ORT 1.23.2.

Ten checkpoints were converted: English V3, German V3, Spanish V3, French
V3, Indic V4, Cyrillic V4, Turkic V5, Caucasian V5, CIS base no-stress V5,
and CIS extended V5. This is not Turkish support: Turkic is a family and no
Turkish checkpoint was supplied in this set.

## Conversion evidence (2026-10-16)

The nine-model non-English sweep recorded 360 frontend comparisons, 72
complete original-checkpoint/ONNX language waveform comparisons (maximum
absolute difference 0.00052913), 221 fixed-voice smoke tests, and 464
identical-output comparisons between normal and lower-memory modes.
The English lab recorded five additional original/ONNX waveform comparisons.
These numerical checks establish fidelity for those samples, not universal
quality or language coverage. Source identities, measurements, ASR results
and indexed recordings are in the separate `silero-evaluation-20261016.zip`
handoff asset, not committed generated records.

ASR checked 33 non-English cases and skipped 39 unsupported cases. Only
three had an exact normalized transcript; script mismatches, limited
recognizer coverage and some errors were also present in original-model
audio. ASR is a sanity check, not proof of intelligibility for languages the
reviewer does not speak. No blanket native-language approval is claimed.

Representative paragraph generation rates were roughly 20-22 times real
time for spectral V5, 49-50 for V4 and 10 for Western V3 with normal memory
settings. Lower-memory settings traded throughput for reduced retained
memory. Consult per-checkpoint measurements rather than inferring one
universal rate. These are dated complete-utterance timings, not first PCM.

## Separate NVDA integration (2026-10-16)

The exact internal Sonata 2026.2.37 package was exercised in portable NVDA
2026.1.1, with actual model/voice identity checked before each test. Ten
checkpoint types and English plus fifteen foreign profiles emitted audio.
English words-only Say All, character/word/line navigation, cancellation and
recovery, voice switching and speech-control routing were exercised. These
are integration observations for that exact package, not certification of
the generic Python API or the current Sonata development version.

Limitations observed: unexpanded numeric-only chunks could stop Say All;
some ASR results were inconclusive; voice-manager UI acceptance was not
completed; installation/update/removal, current NVDA beta compatibility,
and all-language native listening were not established by that run. Sonata
text processing subsequently advanced, but this report does not relabel the
older run as a test of newer code. No raw personal NVDA logs are included.

## Contribution revalidation

Current-branch model-free tests, exporter smoke/rebuild checks and retained
model runtime comparisons are recorded separately in the handoff validation
record. They do not silently replace these original measurements.
