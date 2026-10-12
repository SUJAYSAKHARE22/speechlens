import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from speechlens.analysis.models import AnalysisResult
from speechlens.api.app import create_app
from tests.conftest import SR


@pytest.fixture(scope="module")
def client(demo_dir):
    return TestClient(create_app(demo_dir))


def _flac(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="FLAC")
    return buf.getvalue()


def test_health_and_static_assets(client):
    assert client.get("/api/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200 and "SpeechLens" in page.text
    plotly = client.get("/static/plotly.min.js")
    assert plotly.status_code == 200 and len(plotly.content) > 1_000_000


def test_listings(client):
    refs = client.get("/api/references").json()
    assert refs[0]["id"] == "ref_a" and refs[0]["n_words"] > 100
    sample = client.get("/api/samples").json()[0]
    assert sample["id"] == "pause" and sample["ground_truth"][0]["type"] == "overlong_pause"
    assert "balanced" in client.get("/api/presets").json()
    audio = client.get("/api/samples/pause/audio")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/flac"


def test_analyze_sample_returns_a_valid_grounded_result(client):
    r = client.post("/api/analyze_sample", json={"sample_id": "pause", "preset": "persuasive"})
    assert r.status_code == 200
    result = AnalysisResult.model_validate(r.json())
    assert result.mode == "reference" and result.scores.preset == "persuasive"
    assert any(g.type == "overlong_pause" for g in result.regions)


def test_analyze_upload_uses_the_reference_text_when_transcript_is_blank(client, b_dataset):
    files = {"audio": ("take.flac", _flac(b_dataset["L0_ideal"].audio), "audio/flac")}
    r = client.post("/api/analyze", files=files, data={"reference_id": "ref_a"})
    assert r.status_code == 200
    assert r.json()["regions"] == [] and r.json()["scores"]["overall"] > 85


def test_input_validation_errors_are_clear(client, b_dataset):
    audio = _flac(b_dataset["L0_ideal"].audio)
    ok = {"audio": ("t.flac", audio, "audio/flac")}
    assert client.post("/api/analyze", files=ok).status_code == 400  # no transcript, no reference
    assert client.post("/api/analyze", files=ok, data={"reference_id": "nope"}).status_code == 404
    bad = client.post("/api/analyze", files=ok, data={"reference_id": "ref_a", "preset": "x"})
    assert bad.status_code == 400 and "unknown preset" in bad.json()["detail"]
    short = {"audio": ("s.flac", _flac(np.zeros(SR // 2)), "audio/flac")}
    assert (
        client.post("/api/analyze", files=short, data={"reference_id": "ref_a"}).status_code == 400
    )
    junk = {"audio": ("x.wav", b"this is not audio", "audio/wav")}
    assert (
        client.post("/api/analyze", files=junk, data={"reference_id": "ref_a"}).status_code == 400
    )
    mismatch = client.post(
        "/api/analyze", files=ok, data={"reference_id": "ref_a", "transcript": "far too few words"}
    )
    assert mismatch.status_code == 422 and "word count" in mismatch.json()["detail"]
    assert client.post("/api/analyze_sample", json={"sample_id": "ghost"}).status_code == 404
    assert client.get("/api/samples/ghost/audio").status_code == 404


def test_empty_data_dir_still_serves(tmp_path):
    c = TestClient(create_app(tmp_path))
    assert c.get("/api/references").json() == [] and c.get("/api/samples").json() == []
