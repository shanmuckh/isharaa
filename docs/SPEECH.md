# Ishaara speech: method and implementation

Speech connects hearing users to both sign directions:

```
Speech -> Sign:  microphone -> 16 kHz PCM -> POST /api/transcribe (Whisper) -> English
                 -> gloss translator -> 3D avatar
Sign -> Speech:  camera -> DETR -> transcript -> browser speech synthesis
```

## Speech recognition

* **Model:** Whisper `base.en` through faster-whisper (CTranslate2, int8 on CPU),
  with its voice-activity filter. It runs on this computer. The model is
  downloaded once and audio is never sent to an external service. Browser speech
  APIs were avoided because Chrome's sends audio to a remote service.
* **Capture** (`static/speech/app.js`): an AudioWorklet records raw samples, which
  are resampled to 16 kHz 16-bit PCM in the browser. The server needs no codec.
* **End of speech:** level-based detection. Listening stops after 1.2 s of quiet
  following speech, after 8 s with no speech at all, or at 30 s.
* **Live preview:** the audio so far is transcribed every 1.5 s, one request at a
  time, so words appear while the user speaks. A final pass produces the text
  that is translated.

## Speech output

The browser's speech synthesis reads the transcript on request, or each sign as
it is recognised ("Read signs aloud"). The microphone cancels any speech in
progress so the app never transcribes itself.

## Evaluation

```bash
uv run python src/evaluate_speech.py [voice]
```

Speaks the 50 sentences of the gloss evaluation set with a macOS voice, transcribes
them, and compares the resulting gloss with the gloss of the typed sentence.

| Voice | WER | Same gloss as typed | Speed (M2 CPU) |
|---|---|---|---|
| Samantha (synthetic) | 0.0 % | 100 % | 0.55 s per sentence, 2.3x real time |

**This is an upper bound.** A clean synthetic voice is far easier than real speech,
so these numbers do not measure accuracy for real users. For the paper, record
real speakers, including Indian-accented English, in normal rooms and report WER
on those recordings.

## Limitations

* **Names are the weak point.** In a browser test with audio processing on,
  "Priya" was heard as "Kriya". Recognition errors flow straight into the gloss;
  the text box stays editable, and "Sign it" re-signs after a correction.
* **English only.** `base.en` matches the English -> ASL translator. Hindi or Telugu
  input would need a multilingual model plus translation into English first.
* **Level-based end-of-speech** can stop early during long pauses or run on in
  loud rooms. Pressing the button ends listening by hand.
* The voice used for output depends on the operating system and browser.
