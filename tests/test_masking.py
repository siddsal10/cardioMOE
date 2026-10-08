import pandas as pd

from cardiomoe.data.text_prep import MASK, build_mask_terms, compile_mask, mask_text, normalize_text

SCP = pd.DataFrame(
    {"description": ["inferior myocardial infarction", "left ventricular hypertrophy", "sinus rhythm",
                     "non-specific st changes"],
     "diagnostic": [1, 1, None, None], "form": [None, None, None, 1], "rhythm": [None, None, 1, None]},
    index=["IMI", "LVH", "SR", "NST_"])


def masked(text, scope="nonrhythm"):
    return mask_text(normalize_text(text), compile_mask(build_mask_terms(SCP, scope)))


def test_diagnostic_terms_masked():
    out = masked("Sinus rhythm. Inferior myocardial infarction, LVH.")
    assert "infarction" not in out and "lvh" not in out
    assert out.count(MASK) >= 1


def test_rhythm_rate_axis_kept():
    out = masked("sinus rhythm, heart rate 72, left axis deviation, pr interval normal")
    assert "sinus rhythm" in out and "72" in out and "left axis" in out and "pr interval" in out


def test_stems_and_phrases():
    out = masked("anteroseptal infarct, ischemic st depression, t-wave inversion, AV block, mi")
    for w in ["infarct", "ischemic", "st depression", "t-wave", "block", " mi"]:
        assert w not in out


def test_word_boundaries():
    out = masked("minimal changes, admission")  # 'mi' must not hit inside words
    assert out == "minimal changes, admission"


def test_runs_collapse():
    out = masked("infarct infarct, ischemia")
    assert out.count(MASK) == 1


def test_scope_all_masks_rhythm():
    assert "sinus rhythm" not in masked("sinus rhythm", scope="all")


def test_spelling_variants_and_plurals():
    out = masked("normal ekg. ischaemic changes. t waves flat")
    assert "normal ekg" not in out and "ischaemic" not in out and "t waves" not in out
