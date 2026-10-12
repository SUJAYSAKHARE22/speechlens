"""FastAPI backend for the SpeechLens dashboard."""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from speechlens import __version__
from speechlens.analysis.pipeline import Analyzer, Reference
from speechlens.analysis.rubric import PRESETS
from speechlens.audio import load_audio
from speechlens.config import Config
from speechlens.text import tokenize
from speechlens.tts import espeak_available

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_SECONDS = 300.0
MIN_SECONDS = 2.0


class SampleRequest(BaseModel):
    sample_id: str
    preset: str = "balanced"


def decode_audio(data: bytes, suffix: str, sr: int) -> np.ndarray:
    """Decode an upload to mono float audio at ``sr`` (soundfile, then librosa/ffmpeg)."""
    with tempfile.NamedTemporaryFile(suffix=suffix or ".wav", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            return load_audio(tmp.name, sr)
        except Exception:  # noqa: BLE001 - fall back to a broader decoder for mp3/m4a/etc.
            try:
                import librosa

                y, _ = librosa.load(tmp.name, sr=sr, mono=True)
                return y.astype("float64")
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(400, "Could not decode the audio file.") from exc


def create_app(data_dir: str | Path = "data/demo", config: Config | None = None) -> FastAPI:
    cfg = config or Config()
    sr = cfg.audio.sample_rate
    data = Path(data_dir)
    analyzer = Analyzer(cfg)
    lock = threading.Lock()  # one analysis at a time keeps memory bounded on small hosts
    refs: dict[str, Reference] = {}

    def reference_dirs() -> dict[str, Path]:
        root = data / "references"
        return {p.name: p for p in sorted(root.iterdir()) if p.is_dir()} if root.exists() else {}

    def sample_dirs() -> dict[str, Path]:
        root = data / "samples"
        return {p.name: p for p in sorted(root.iterdir()) if p.is_dir()} if root.exists() else {}

    def get_reference(rid: str) -> Reference:
        if rid not in refs:
            d = reference_dirs().get(rid)
            if d is None:
                raise HTTPException(404, f"unknown reference {rid!r}")
            refs[rid] = Reference.load(d, sr)
        return refs[rid]

    def run(audio: np.ndarray, transcript: str, reference: Reference | None, preset: str):
        if preset not in PRESETS:
            raise HTTPException(400, f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        dur = len(audio) / sr
        if not MIN_SECONDS <= dur <= MAX_SECONDS:
            raise HTTPException(
                400, f"audio must be {MIN_SECONDS:.0f}-{MAX_SECONDS:.0f} s (got {dur:.1f} s)"
            )
        if reference is None and not espeak_available():
            raise HTTPException(
                503,
                "Reference-free analysis needs espeak-ng; choose a reference instead.",
            )
        try:
            with lock:
                return analyzer.analyze(
                    audio, transcript, sr=sr, reference=reference, preset=preset
                )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    app = FastAPI(title="SpeechLens", version=__version__)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "espeak": espeak_available()}

    @app.get("/api/presets")
    def presets() -> dict:
        return PRESETS

    @app.get("/api/references")
    def references() -> list[dict]:
        out = []
        for rid, d in reference_dirs().items():
            text = (d / "transcript.txt").read_text(encoding="utf-8").strip()
            out.append({"id": rid, "n_words": len(tokenize(text)), "transcript": text})
        return out

    @app.get("/api/samples")
    def samples() -> list[dict]:
        return [
            {"id": sid, **json.loads((d / "meta.json").read_text(encoding="utf-8"))}
            for sid, d in sample_dirs().items()
        ]

    @app.get("/api/samples/{sample_id}/audio")
    def sample_audio(sample_id: str) -> FileResponse:
        d = sample_dirs().get(sample_id)
        if d is None:
            raise HTTPException(404, "unknown sample")
        return FileResponse(d / "audio.flac", media_type="audio/flac")

    @app.post("/api/analyze")
    async def analyze(
        audio: UploadFile = File(...),
        transcript: str = Form(""),
        reference_id: str = Form(""),
        preset: str = Form("balanced"),
    ) -> JSONResponse:
        data_bytes = await audio.read()
        if len(data_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file too large (max 30 MB)")
        ref = get_reference(reference_id) if reference_id else None
        text = transcript.strip() or (ref.transcript if ref else "")
        if not text:
            raise HTTPException(400, "provide a transcript, or choose a reference to use its text")
        x = decode_audio(data_bytes, Path(audio.filename or "").suffix, sr)
        return JSONResponse(json.loads(run(x, text, ref, preset).model_dump_json()))

    @app.post("/api/analyze_sample")
    def analyze_sample(req: SampleRequest) -> JSONResponse:
        d = sample_dirs().get(req.sample_id)
        if d is None:
            raise HTTPException(404, "unknown sample")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        text = (d / "transcript.txt").read_text(encoding="utf-8").strip()
        x = load_audio(d / "audio.flac", sr)
        res = run(x, text, get_reference(meta["reference_id"]), req.preset)
        return JSONResponse(json.loads(res.model_dump_json()))

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC / "index.html")

    return app
