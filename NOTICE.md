# Third-party components and licensing notes

* **Plotly.js** (`src/speechlens/api/static/plotly.min.js`, vendored so the dashboard works offline):
  MIT licence, (c) Plotly, Inc.
* **espeak-ng**: GPL-3.0. It is used only as an *external executable* (never linked or bundled) to
  synthesise simulated speakers and for reference-free alignment. Audio it produces is not
  encumbered by the engine's licence in the usual interpretation, but check your own policy.
* **Praat / Parselmouth** (GPL-3.0 / GPLv3-compatible), librosa (ISC), SciPy/NumPy (BSD),
  FastAPI (MIT), Pydantic (MIT), soundfile (BSD).
* **Gettysburg Address** (Abraham Lincoln, 1863): public domain.
* **FFmpeg** : Used for audio processing and conversion in SpeechLens. FFmpeg is distributed under the GNU LGPL v2.1 or later, subject to the applicable build configuration and included components. Some configurations include GPL-licensed components.
Official licensing information: https://ffmpeg.org/legal.html.

The project's own licence is yours to choose; because Praat/Parselmouth is GPL, choose a
GPL-compatible licence if you redistribute combined work.
