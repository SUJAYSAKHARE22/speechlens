"""Causal explanations: translate raw deviation into human-readable, number-backed text.

Deterministic templates (no LLM) keep results reproducible. Every explanation carries
(1) the acoustic evidence, (2) the mathematical rationale (formula, tolerance, z-score), and
(3) a concrete suggestion. Invariant: ``evidence.z == evidence.delta / evidence.sigma``, and the
rationale line quotes exactly those numbers.
"""

from __future__ import annotations

import numpy as np

from speechlens.analysis.compare import Deviations
from speechlens.analysis.detect import (
    FAMILY,
    RawRegion,
    confidence_from_z,
    severity_from_z,
    smoothed_pace,
)
from speechlens.analysis.models import Evidence, FlawRegion
from speechlens.config import DetectionConfig, Tolerance
from speechlens.features import WordFeatures

_PUNCT_NAME = {"clause": "a comma/clause break", "sentence": "a full stop"}


def _mean(a: np.ndarray) -> float:
    v = a[np.isfinite(a)]
    return float(v.mean()) if len(v) else 0.0


def explain_region(
    raw: RawRegion,
    dev: Deviations,
    z: dict[str, np.ndarray],
    words: list[WordFeatures],
    tol: Tolerance,
    cfg: DetectionConfig,
    region_id: int,
) -> FlawRegion:
    i0, i1, thr = raw.i0, raw.i1, cfg.z_threshold
    span = slice(i0, i1 + 1)
    text = " ".join(w.text for w in words[span])
    sev, conf = severity_from_z(raw.peak_z, thr), confidence_from_z(raw.peak_z, thr)
    ref = "your overall tempo" if dev.mode == "reference" else "your average pace"
    expected_src = "the reference" if dev.mode == "reference" else "typical delivery"
    focus_s = focus_e = None
    d = dev.detail
    syll = sum(w.n_syll for w in words[span])

    def evidence(metric, observed, expected, unit, delta, delta_unit, sigma) -> Evidence:
        return Evidence(
            metric=metric, observed=observed, expected=expected, unit=unit, delta=delta,
            delta_unit=delta_unit, sigma=sigma, z=delta / sigma,
        )  # fmt: skip

    if raw.type in ("rush", "drag"):
        zz = float(np.median(smoothed_pace(z)[span]))
        delta = zz * tol.pace_sigma
        obs = words[i1].end - words[i0].start
        exp = float(d["exp_duration"][span].sum() + dev.pause_exp[i0:i1].sum())
        o_rate, e_rate = syll / obs, syll / exp
        if raw.type == "rush":
            head = f"Rushing: {exp / obs:.1f}x faster than expected"
            expl = (
                f'Over "{text}" you spoke at {o_rate:.1f} syllables/s where {expected_src} implies '
                f"{e_rate:.1f} (matched to {ref}). Pauses inside the phrase were squeezed too, "
                "which removes the beat that gives the words weight."
            )
            tip = "Slow down here and let the phrase breathe; mark one short pause inside it."
        else:
            head = f"Dragging: {obs / exp:.1f}x slower than expected"
            expl = (
                f'Over "{text}" you spoke at {o_rate:.1f} syllables/s where {expected_src} implies '
                f"{e_rate:.1f} (matched to {ref}). The slow stretch breaks the momentum of the line."
            )
            tip = "Pick up the pace to match the surrounding delivery; keep the energy constant."
        ev = evidence("pace deviation", delta, 0.0, "octaves", delta, "octaves", tol.pace_sigma)
        rat = (
            "per-word pace = log2(duration_you / duration_expected), tempo-matched; median over this "
            f"stretch (3-word smoothing) = {delta:+.2f} octaves; sigma = {tol.pace_sigma:.2f} -> "
            f"z = {ev.z:+.1f} (flag at |z| >= {thr}). Whole phrase: {obs:.2f} s vs {exp:.2f} s expected."
        )
    elif raw.type in ("misplaced_pause", "overlong_pause", "missing_pause"):
        i = raw.focus_after if raw.focus_after is not None else i0
        obs, exp = float(dev.pause_obs[i]), float(dev.pause_exp[i])
        w_a, w_b = words[i], words[min(i + 1, len(words) - 1)]
        focus_s, focus_e = w_a.end, w_b.start
        sigma = tol.pause_sigma_abs + tol.pause_sigma_rel * exp
        where = _PUNCT_NAME.get(w_a.boundary)
        if raw.type == "misplaced_pause":
            head = f"Misplaced pause: {obs:.1f} s silence mid-phrase"
            expl = (
                f'A {obs:.2f} s silence falls between "{w_a.text}" and "{w_b.text}", where the text has '
                f"no punctuation (expected about {exp:.2f} s). Breaking a phrase here cuts the sentence in two."
            )
            tip = f'Say "{w_a.text} {w_b.text}" in one breath; keep pauses for punctuation.'
        elif raw.type == "overlong_pause":
            head = f'Overlong pause: {obs:.1f} s after "{w_a.text}"'
            expl = (
                f'After "{w_a.text}" ({where}) you paused {obs:.2f} s; about {exp:.2f} s is expected '
                "here. Held this long, the silence reads as hesitation rather than emphasis."
            )
            tip = "Shorten this pause to roughly the expected length and move on with confidence."
        else:
            head = f"Missing pause at {where or 'punctuation'}"
            expl = (
                f'After "{w_a.text}" the text calls for {where or "a pause"} (about {exp:.2f} s) but you '
                f"left only {obs:.2f} s, running two thoughts together."
            )
            tip = "Stop for a beat at the punctuation; let the listener absorb the point."
        ev = evidence("pause length", obs, exp, "s", obs - exp, "s", sigma)
        rule = (
            f"Flagged by the missing-pause rule: observed <= {cfg.missing_pause_ratio:.0%} of expected."
            if raw.type == "missing_pause"
            else f"Flagged at z >= {thr} with excess >= {cfg.pause_min_excess_s:.2f} s."
        )
        rat = (
            f"delta = observed - expected = {obs:.2f} - {exp:.2f} = {ev.delta:+.2f} s; sigma = "
            f"{tol.pause_sigma_abs:.2f} + {tol.pause_sigma_rel:.2f} x expected = {sigma:.2f} s -> "
            f"z = {ev.z:+.1f}. {rule}"
        )
    elif raw.type == "monotone":
        zz = float(np.median(z["pitch"][span]))
        delta = zz * tol.pitch_sigma
        o, e = _mean(d["pitch_obs_std"][span]), _mean(d["pitch_exp_std"][span])
        head = f"Monotone: pitch range {2**delta:.0%} of expected"
        expl = (
            f'Across "{text}" your pitch varied by only {o:.1f} semitones (standard deviation) against '
            f"{e:.1f} in {expected_src}. A flat contour hides which words matter."
        )
        tip = "Lift the stressed words and let the pitch fall at phrase ends; exaggerate a little."
        ev = evidence("F0 std", o, e, "semitones", delta, "octaves", tol.pitch_sigma)
        rat = (
            "pitch = log2(std_you / std_expected) over a 5-word window (F0 in semitones re your "
            f"own median); median here = {delta:+.2f} octaves; sigma = {tol.pitch_sigma:.2f} -> z = {ev.z:+.1f}."
        )
    elif raw.type == "uptalk":
        o, e = float(d["tail_obs"][i1]), float(d["tail_exp"][i1])
        head = f'Rising end on "{words[i1].text}": +{o - e:.1f} st'
        expl = (
            f'The statement ending "{words[i1].text}" finishes with pitch {o:+.1f} st re your median, '
            f"{o - e:.1f} st higher than expected ({e:+.1f}). A rising ending sounds like a question "
            "and weakens the claim."
        )
        tip = "Let the pitch settle downward on the last word of a statement."
        ev = evidence("terminal F0", o, e, "semitones", o - e, "semitones", tol.terminal_sigma)
        rat = (
            f"terminal delta = F0_tail(you) - F0_tail(expected) = {o:+.2f} - ({e:+.2f}) = {ev.delta:+.2f} st; "
            f"sigma = {tol.terminal_sigma:.1f} st -> z = {ev.z:+.1f}."
        )
    elif raw.type == "trail_off":
        ed = d["energy_obs"][span] - d["energy_exp"][span]
        first, last = float(ed[0]), float(ed[-1])
        head = f'Trailing off: {abs(last - first):.0f} dB fade by "{words[i1].text}"'
        expl = (
            f'Energy on "{text}" falls steadily, ending {abs(last):.0f} dB below expectation on '
            f'"{words[i1].text}". The sentence loses its landing and the audience loses the point.'
        )
        tip = "Keep the volume up through the final word; finish as strongly as you started."
        ev = evidence("energy at final word", last, 0.0, "dB", last, "dB", tol.energy_sigma)
        rat = (
            "energy delta per word = E_you - E_expected (dB re each recording's 90th percentile): "
            f"{first:+.1f} -> {last:+.1f} dB; z uses the final word: sigma = {tol.energy_sigma:.1f} dB -> z = {ev.z:+.1f}."
        )
    elif raw.type == "volume_collapse":
        mean = _mean(d["energy_obs"][span] - d["energy_exp"][span])
        head = f"Volume drop: {abs(mean):.0f} dB quieter than expected"
        expl = (
            f'Across "{text}" you were on average {abs(mean):.0f} dB quieter than {expected_src}, '
            "so the phrase becomes hard to hear."
        )
        tip = "Project this phrase; keep breath support and distance to the microphone steady."
        ev = evidence("mean energy delta", mean, 0.0, "dB", mean, "dB", tol.energy_sigma)
        rat = (
            f"mean energy delta = {mean:+.1f} dB (levels normalised to each recording's own 90th "
            f"percentile); sigma = {tol.energy_sigma:.1f} dB -> z = {ev.z:+.1f}."
        )
    else:  # mumble
        mean = _mean(d["hf_obs"][span] - d["hf_exp"][span])
        head = f"Reduced clarity: {abs(mean):.0f} dB less high-frequency detail"
        expl = (
            f'On "{text}" the consonant/high-frequency content (above 2 kHz) is {abs(mean):.0f} dB weaker '
            "than expected, a sign of mumbling or turning away from the microphone."
        )
        tip = "Articulate consonants crisply, especially word endings; face the microphone."
        ev = evidence("HF share delta", mean, 0.0, "dB", mean, "dB", tol.clarity_sigma)
        rat = (
            "clarity = (share of energy above 2 kHz, centred on the recording median) you - expected "
            f"= {mean:+.1f} dB; sigma = {tol.clarity_sigma:.1f} dB -> z = {ev.z:+.1f}."
        )

    return FlawRegion(
        id=region_id, type=raw.type, family=FAMILY[raw.type],
        start=words[i0].start, end=words[i1].end, word_start=i0, word_end=i1, words=text,
        severity=sev, confidence=conf, evidence=ev, headline=head, explanation=expl,
        rationale=rat, suggestion=tip, focus_start=focus_s, focus_end=focus_e,
    )  # fmt: skip
