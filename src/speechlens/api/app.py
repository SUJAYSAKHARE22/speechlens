
"""FastAPI backend for the SpeechLens dashboard."""

from __future__ import annotations

import json
import logging
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


# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------

logger = logging.getLogger("speechlens")

STATIC = Path(__file__).parent / "static"

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_SECONDS = 300.0
MIN_SECONDS = 2.0


class SampleRequest(BaseModel):
    sample_id: str
    preset: str = "balanced"


# ---------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------

def decode_audio(data: bytes, suffix: str, sr: int) -> np.ndarray:
    """Decode uploaded audio safely on Windows and other platforms."""

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded audio file is empty."
        )

    suffix = suffix.lower() if suffix else ".wav"
    tmp_path = None

    try:
        # Create temporary file.
        # delete=False is important on Windows because the file
        # must be closed before SoundFile or Librosa can reopen it.

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        # The temporary file is now closed.

        # First attempt: SpeechLens primary audio decoder
        try:
            audio = load_audio(tmp_path, sr)

            if audio is None or len(audio) == 0:
                raise ValueError("Decoded audio is empty.")

            logger.info(
                "Audio decoded successfully using primary decoder."
            )

            return np.asarray(audio, dtype=np.float64)

        except Exception:
            logger.warning(
                "Primary audio decoder failed. "
                "Trying librosa fallback.",
                exc_info=True
            )

        # Second attempt: Librosa
        try:
            import librosa

            audio, _ = librosa.load(
                tmp_path,
                sr=sr,
                mono=True
            )

            if audio is None or len(audio) == 0:
                raise ValueError("Librosa returned empty audio.")

            logger.info(
                "Audio decoded successfully using librosa."
            )

            return audio.astype(np.float64)

        except Exception as exc:
            logger.exception(
                "Both audio decoding attempts failed."
            )

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Audio decoding failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            ) from exc

    finally:
        # Always clean up the temporary file.
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
                logger.debug(
                    "Temporary audio file removed: %s",
                    tmp_path
                )
            except OSError:
                logger.warning(
                    "Could not remove temporary audio file: %s",
                    tmp_path,
                    exc_info=True
                )


# ---------------------------------------------------------
# Application factory
# ---------------------------------------------------------

def create_app(
    data_dir: str | Path = "data/demo",
    config: Config | None = None
) -> FastAPI:

    cfg = config or Config()
    sr = cfg.audio.sample_rate

    data = Path(data_dir)

    analyzer = Analyzer(cfg)

    # One analysis at a time keeps memory usage bounded.
    lock = threading.Lock()

    refs: dict[str, Reference] = {}

    # -----------------------------------------------------
    # Reference directories
    # -----------------------------------------------------

    def reference_dirs() -> dict[str, Path]:

        root = data / "references"

        return (
            {
                p.name: p
                for p in sorted(root.iterdir())
                if p.is_dir()
            }
            if root.exists()
            else {}
        )

    # -----------------------------------------------------
    # Sample directories
    # -----------------------------------------------------

    def sample_dirs() -> dict[str, Path]:

        root = data / "samples"

        return (
            {
                p.name: p
                for p in sorted(root.iterdir())
                if p.is_dir()
            }
            if root.exists()
            else {}
        )

    # -----------------------------------------------------
    # Reference loading
    # -----------------------------------------------------

    def get_reference(rid: str) -> Reference:

        if rid not in refs:

            d = reference_dirs().get(rid)

            if d is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown reference {rid!r}"
                )

            refs[rid] = Reference.load(d, sr)

        return refs[rid]

    # -----------------------------------------------------
    # Main analysis function
    # -----------------------------------------------------

    def run(
        audio: np.ndarray,
        transcript: str,
        reference: Reference | None,
        preset: str
    ):

        if preset not in PRESETS:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown preset {preset!r}; "
                    f"choose from {sorted(PRESETS)}"
                )
            )

        dur = len(audio) / sr

        logger.info(
            "Audio duration: %.2f seconds | Sample rate: %s",
            dur,
            sr
        )

        if not MIN_SECONDS <= dur <= MAX_SECONDS:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Audio must be between "
                    f"{MIN_SECONDS:.0f} and {MAX_SECONDS:.0f} seconds "
                    f"(got {dur:.1f} seconds)."
                )
            )

        if reference is None and not espeak_available():

            raise HTTPException(
                status_code=503,
                detail=(
                    "Reference-free analysis needs espeak-ng; "
                    "choose a reference instead."
                )
            )

        try:

            with lock:

                return analyzer.analyze(
                    audio,
                    transcript,
                    sr=sr,
                    reference=reference,
                    preset=preset
                )

        except ValueError as exc:

            logger.exception(
                "Analysis validation failed."
            )

            raise HTTPException(
                status_code=422,
                detail=str(exc)
            ) from exc

    # -----------------------------------------------------
    # FastAPI application
    # -----------------------------------------------------

    app = FastAPI(
        title="SpeechLens",
        version=__version__
    )

    # -----------------------------------------------------
    # Health check
    # -----------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:

        return {
            "status": "ok",
            "version": __version__,
            "espeak": espeak_available()
        }

    # -----------------------------------------------------
    # Available presets
    # -----------------------------------------------------

    @app.get("/api/presets")
    def presets() -> dict:

        return PRESETS

    # -----------------------------------------------------
    # Available references
    # -----------------------------------------------------

    @app.get("/api/references")
    def references() -> list[dict]:

        out = []

        for rid, d in reference_dirs().items():

            text = (
                d / "transcript.txt"
            ).read_text(encoding="utf-8").strip()

            out.append({
                "id": rid,
                "n_words": len(tokenize(text)),
                "transcript": text
            })

        return out

    # -----------------------------------------------------
    # Available sample recordings
    # -----------------------------------------------------

    @app.get("/api/samples")
    def samples() -> list[dict]:

        return [
            {
                "id": sid,
                **json.loads(
                    (d / "meta.json").read_text(encoding="utf-8")
                )
            }
            for sid, d in sample_dirs().items()
        ]

    # -----------------------------------------------------
    # Sample audio endpoint
    # -----------------------------------------------------

    @app.get("/api/samples/{sample_id}/audio")
    def sample_audio(sample_id: str) -> FileResponse:

        d = sample_dirs().get(sample_id)

        if d is None:

            raise HTTPException(
                status_code=404,
                detail="Unknown sample"
            )

        return FileResponse(
            d / "audio.flac",
            media_type="audio/flac"
        )

    # -----------------------------------------------------
    # Analyze uploaded audio
    # -----------------------------------------------------

    @app.post("/api/analyze")
    async def analyze(
        audio: UploadFile = File(...),
        transcript: str = Form(""),
        reference_id: str = Form(""),
        preset: str = Form("balanced"),
    ) -> JSONResponse:

        try:

            data_bytes = await audio.read()

            logger.info(
                "New analysis request | filename=%s | size=%d bytes "
                "| preset=%s | reference=%s",
                audio.filename,
                len(data_bytes),
                preset,
                reference_id
            )

            # Validate upload
            if not data_bytes:

                raise HTTPException(
                    status_code=400,
                    detail="Uploaded audio file is empty."
                )

            if len(data_bytes) > MAX_UPLOAD_BYTES:

                raise HTTPException(
                    status_code=413,
                    detail="File too large (max 30 MB)."
                )

            # Load reference if selected
            ref = (
                get_reference(reference_id)
                if reference_id
                else None
            )

            # Get transcript
            text = transcript.strip() or (
                ref.transcript if ref else ""
            )

            if not text:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Provide a transcript, or choose a reference "
                        "to use its text."
                    )
                )

            # Decode uploaded audio
            suffix = Path(
                audio.filename or ""
            ).suffix

            x = decode_audio(
                data_bytes,
                suffix,
                sr
            )

            logger.info(
                "Audio decoded | samples=%d | duration=%.2f seconds",
                len(x),
                len(x) / sr
            )

            # Run SpeechLens analyzer
            result = run(
                x,
                text,
                ref,
                preset
            )

            logger.info(
                "Audio analysis completed successfully."
            )

            return JSONResponse(
                json.loads(result.model_dump_json())
            )

        except HTTPException:
            raise

        except Exception as exc:

            logger.exception(
                "Unexpected error in /api/analyze"
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    f"Analysis failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            ) from exc

        finally:

            await audio.close()

    # -----------------------------------------------------
    # Analyze predefined sample
    # -----------------------------------------------------

    @app.post("/api/analyze_sample")
    def analyze_sample(req: SampleRequest) -> JSONResponse:

        d = sample_dirs().get(req.sample_id)

        if d is None:

            raise HTTPException(
                status_code=404,
                detail="Unknown sample"
            )

        meta = json.loads(
            (d / "meta.json").read_text(encoding="utf-8")
        )

        text = (
            d / "transcript.txt"
        ).read_text(encoding="utf-8").strip()

        x = load_audio(
            d / "audio.flac",
            sr
        )

        res = run(
            x,
            text,
            get_reference(meta["reference_id"]),
            req.preset
        )

        return JSONResponse(
            json.loads(res.model_dump_json())
        )

    # -----------------------------------------------------
    # Static dashboard
    # -----------------------------------------------------

    if STATIC.exists():

        app.mount(
            "/static",
            StaticFiles(directory=STATIC),
            name="static"
        )

        @app.get("/")
        def index() -> FileResponse:

            return FileResponse(
                STATIC / "index.html"
            )

    return app
