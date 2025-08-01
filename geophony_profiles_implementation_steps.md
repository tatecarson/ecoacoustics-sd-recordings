# Geophony Profiles Implementation — Numbered Steps

This document provides **numbered, atomic steps** you can give to a local coding agent to implement **geophony tuning profiles** (wind, rain, surf_river, thunder, geophony_general) in your ecoacoustic beamforming pipeline. Every step includes **checkpoints** and **acceptance criteria**.

---

## 1) Prerequisites and Setup

**Tasks for Agent**
- Ensure the project installs and runs locally.
- Confirm a short 30–60 s test ambisonic file is available.
- Confirm Python version and dependencies (e.g., `scikit-maad`, `numpy`, `scipy`, `soundfile`, `matplotlib`).

**Suggested Commands**
```bash
python -V
pip install -r requirements.txt  # if present
```

**Checkpoint**
- The main script runs to completion on a short file **without** profiles:
```bash
python beamforming-export-scikit-maad.py path/to/test.wav --duration 20
```
- Outputs exist: `maad_indices_all_directions.csv`, `selection_report.txt`, exported WAVs.

**Acceptance Criteria**
- No import errors.
- Output CSV contains expected columns (ACI, ADI, Hf, Ht, spatial uniqueness, total_score).
- `selection_report.txt` created.

-- DONE --

---

## 2) Create a Feature Branch

**Tasks for Agent**
- Create and switch to a new branch.

**Commands**
```bash
git checkout -b feature/geophony-profiles
```

**Checkpoint**
- `git status` shows branch `feature/geophony-profiles`.

**Acceptance Criteria**
- Branch created and active.

-- DONE --

---

## 3) Add a Profile Registry to `beamforming_utils/config.py`

**Tasks for Agent**
- Open `beamforming_utils/config.py`.
- Add a **profile registry** with weights and parameters and a `get_profile(profile_name)` helper.
- Keep existing defaults; add the following near the bottom:

**Code to Insert**
```python
# --- Profile registry for geophony tuning ---

PROFILE_WEIGHTS = {
    "wind":              dict(Hf=0.15, ADI=0.10, TEMP=0.30, ACI=0.15, SPATIAL=0.30),
    "rain":              dict(Hf=0.10, ADI=0.15, TEMP=0.30, ACI=0.10, SPATIAL=0.35),
    "surf_river":        dict(Hf=0.10, ADI=0.30, TEMP=0.10, ACI=0.20, SPATIAL=0.30),
    "thunder":           dict(Hf=0.05, ADI=0.05, TEMP=0.40, ACI=0.15, SPATIAL=0.35),
    "geophony_general":  dict(Hf=0.10, ADI=0.20, TEMP=0.30, ACI=0.10, SPATIAL=0.30),
    "none":              dict(Hf=W_ACTIVITY, ADI=W_FREQDIV, TEMP=W_TEMP, ACI=W_ACI, SPATIAL=W_SPATIAL),
}

PROFILE_PARAMS = {
    "wind":             dict(ADI_dB=-30, corr=0.60, min_angle=45.0, hpf_hz=120),
    "rain":             dict(ADI_dB=-25, corr=0.60, min_angle=45.0, hpf_hz=None, envelope_median_ms=15),
    "surf_river":       dict(ADI_dB=-30, corr=0.65, min_angle=40.0, hpf_hz=60),
    "thunder":          dict(ADI_dB=-35, corr=0.55, min_angle=50.0, hpf_hz=None),
    "geophony_general": dict(ADI_dB=-30, corr=0.60, min_angle=45.0, hpf_hz=90),
    "none":             dict(ADI_dB=ADI_AEI_DB_THRESHOLD, corr=CORRELATION_THRESHOLD,
                             min_angle=MIN_ANGULAR_SEPARATION_DEG, hpf_hz=None),
}

def get_profile(profile_name: str):
    p = (profile_name or "none").lower()
    weights = PROFILE_WEIGHTS.get(p, PROFILE_WEIGHTS["none"])
    params  = PROFILE_PARAMS.get(p,  PROFILE_PARAMS["none"])
    return weights, params
```

**Checkpoint**
```bash
python -c "from beamforming_utils.config import get_profile; print(get_profile('wind'))"
```

**Acceptance Criteria**
- No syntax errors; returns dicts with keys: `Hf, ADI, TEMP, ACI, SPATIAL` and `ADI_dB, corr, min_angle, hpf_hz`.

-- DONE -- 

---

## 4) Wire Profile Weights Into Uniqueness Score

**Tasks for Agent**
- Open `beamforming_utils/beamforming_analysis.py`.
- Locate where **normalized indices** are combined (Hf, ADI, 1–Ht, ACI, spatial uniqueness).
- Update function to accept `profile_weights` and use them instead of hard-coded globals.

**Code Edit**
```python
def calculate_uniqueness_metrics(..., profile_weights=None, ADI_dB_threshold=None, preproc=None):
    # ... after computing raw metrics ...
    activity_variation  = _norm(Hf_vals)
    frequency_diversity = _norm(ADI_vals)
    temporal_complexity = 1.0 - _norm(Ht_vals)
    spatial_uniqueness  = _norm(spatial_unique)
    acoustic_complexity = _norm(ACI_vals)

    w = profile_weights or {"Hf": W_ACTIVITY, "ADI": W_FREQDIV, "TEMP": W_TEMP, "ACI": W_ACI, "SPATIAL": W_SPATIAL}

    total_score = (
        w["Hf"]      * activity_variation +
        w["ADI"]     * frequency_diversity +
        w["TEMP"]    * temporal_complexity +
        w["ACI"]     * acoustic_complexity +
        w["SPATIAL"] * spatial_uniqueness
    )
    # ... pack and return as before ...
```

**Checkpoint**
```bash
python -c "import beamforming_utils.beamforming_analysis as ba; print('OK')"
```

**Acceptance Criteria**
- Module imports successfully; `profile_weights` parameter appears and is used.

-- DONE -- 

---

## 5) Pass Profile ADI Threshold Into Index Calculation

**Tasks for Agent**
- In `beamforming_analysis.py`, find the call to `maad.features.all_spectral_alpha_indices` (or ADI computation).
- Pass the `ADI_dB_threshold` argument using the profile value (fallback to global).

**Code Edit**
```python
df_alpha, _ = maad.features.all_spectral_alpha_indices(
    Sxx_power, tn, fn,
    flim_bioPh=NDSI_BIO,
    flim_antroPh=NDSI_ANTH,
    R_compatible="soundecology",
    ADI_dB_threshold=ADI_dB_threshold if ADI_dB_threshold is not None else ADI_AEI_DB_THRESHOLD,
    verbose=False, display=False
)
```

**Checkpoint**
- Search/grep shows the parameter is now wired through.

**Acceptance Criteria**
- ADI dB threshold now configurable via the profile parameter.

-- DONE --

---

## 6) Add and Control the Pre-Processing Hook (Revised)

**Purpose:** Stabilize index calculations in geophony (wind, rain, surf/river) **without** altering the WAVs you export by default. The hook runs **only when needed**: driven by the active profile, a CLI override, or an optional auto detector.

---

### 6.1 Add the Hook (analysis-only)

**Tasks for Agent**

* In `beamforming_utils/beamforming_analysis.py` (or a small `preproc.py`), add the hook function.

**Code to Add**

```python
# beamforming_utils/beamforming_analysis.py
import numpy as np
import scipy.signal as sps

def maybe_preprocess(x, sr, preproc):
    """
    Light, analysis-only preprocessing to reduce geophony bias in indices.
    Keys used in `preproc`:
      - hpf_hz: int/float or None (e.g., 60, 90, 120)
      - envelope_median_ms: int/float or None (e.g., 15 for rain tick smoothing)
    Do not use this on exported WAVs unless explicitly requested.
    """
    if not preproc:
        return x

    y = x
    # High-pass (wind/surf/river low-frequency rumble)
    if preproc.get("hpf_hz"):
        wc = preproc["hpf_hz"] / (0.5 * sr)
        wc = min(max(wc, 1e-4), 0.9999)  # clamp numeric safety
        sos = sps.butter(2, wc, btype="highpass", output="sos")
        y = sps.sosfiltfilt(sos, y)

    # Envelope median smoothing (rain micro-impulses)
    if preproc.get("envelope_median_ms"):
        env = np.abs(sps.hilbert(y))
        k = max(3, int(preproc["envelope_median_ms"] * 1e-3 * sr / 256))
        k = 2 * (k // 2) + 1  # odd
        env_s = sps.medfilt(env, kernel_size=k)
        y = y * (env_s / (env + 1e-12))

    return y
```

**Checkpoint**

```bash
python - <<'PY'
import numpy as np
from beamforming_utils.beamforming_analysis import maybe_preprocess
x = np.random.randn(22050*2); sr=22050
y = maybe_preprocess(x, sr, {"hpf_hz":120})
assert len(y) == len(x)
print("OK")
PY
```

**Acceptance Criteria**

* Function imports and returns same-length arrays.
* No runtime errors.

-- DONE -- 

---

### 6.2 Control When It Runs (profile, CLI, auto)

**Tasks for Agent**

* In the **main script** (`beamforming-export-scikit-maad.py`), add CLI switches and merge them with profile params.

**Add CLI arguments**

```python
parser.add_argument(
    "--preproc",
    choices=["off", "force", "auto"],
    default=None,  # None => profile-driven default
    help="Preprocess mode: off (never), force (always if params exist), auto (detect). Default: profile-driven."
)
# Optional explicit overrides (useful when --profile none)
parser.add_argument("--hpf_hz", type=float, default=None)
parser.add_argument("--envelope_median_ms", type=float, default=None)
```

**Merge CLI overrides with profile params**

```python
# after: weights, params = get_profile(args.profile)
if args.hpf_hz is not None:
    params["hpf_hz"] = args.hpf_hz
if args.envelope_median_ms is not None:
    params["envelope_median_ms"] = args.envelope_median_ms
```

**Optional: auto detector (fast heuristics)**

```python
import numpy as np
import scipy.signal as sps
from scipy import stats

def should_preprocess_auto(x, sr, profile_name):
    # analyze at most ~10 s for speed
    x = x[:min(len(x), sr*10)]

    # LF ratio (<120 Hz)
    freqs = np.fft.rfftfreq(len(x), 1/sr)
    X = np.abs(np.fft.rfft(x)) + 1e-12
    LF_ratio = X[freqs < 120].sum() / X.sum()

    # Spectral flatness (proxy for evenness / surf)
    S_flat = np.exp(np.mean(np.log(X))) / (np.mean(X) + 1e-12)

    # Envelope kurtosis (rain-ish)
    env = np.abs(sps.hilbert(x))
    kurt = stats.kurtosis(env)

    if profile_name == "wind" and LF_ratio > 0.35:
        return True, dict(LF_ratio=LF_ratio, S_flat=S_flat, kurt=kurt)
    if profile_name == "surf_river" and S_flat > 0.60:
        return True, dict(LF_ratio=LF_ratio, S_flat=S_flat, kurt=kurt)
    if profile_name == "rain" and kurt > 4.0:
        return True, dict(LF_ratio=LF_ratio, S_flat=S_flat, kurt=kurt)
    if profile_name == "geophony_general" and (LF_ratio > 0.30 or S_flat > 0.55):
        return True, dict(LF_ratio=LF_ratio, S_flat=S_flat, kurt=kurt)
    return False, dict(LF_ratio=LF_ratio, S_flat=S_flat, kurt=kurt)
```

**Apply hook per beam (analysis path only)**

```python
from beamforming_utils.beamforming_analysis import maybe_preprocess

# inside the loop over directions/beams:
beam_for_indices = beam  # keep original for export unless user requests otherwise

# Decide whether to apply preprocessing
if args.preproc == "force":
    apply_preproc = True
elif args.preproc == "off":
    apply_preproc = False
elif args.preproc == "auto":
    apply_preproc, auto_metrics = should_preprocess_auto(beam, sample_rate, args.profile)
    # stash for report provenance
    auto_log_line = (f"AUTO-PREPROC={apply_preproc} "
                     f"(LF_ratio={auto_metrics['LF_ratio']:.2f}, "
                     f"S_flat={auto_metrics['S_flat']:.2f}, kurt={auto_metrics['kurt']:.2f})")
    report_lines.append(auto_log_line)
else:
    # Default: profile-driven only if profile provides params
    apply_preproc = bool(params.get("hpf_hz") or params.get("envelope_median_ms"))

if apply_preproc:
    beam_for_indices = maybe_preprocess(beam, sample_rate, params)

# IMPORTANT: compute spectrogram/indices on beam_for_indices,
# but export the ORIGINAL beam by default.
```

**Checkpoint**

* `--preproc off` with `--profile wind` → no preprocessing applied.
* `--preproc force` with `--profile none --hpf_hz 100` → preprocessing applied (HPF\@100).
* `--preproc auto` with `--profile surf_river` on a river clip → auto logs `AUTO-PREPROC=True`.
* `--preproc auto` on a quiet clip → auto logs `False`.

**Acceptance Criteria**

* Decisions match the mode selected (off/force/auto/profile).
* No changes to default export unless an explicit “export preprocessed” flag is added.

-- DONE -- 

---

### 6.3 Provenance in Reports

**Tasks for Agent**

* In `beamforming_utils/beamforming_report.py`, include preprocessing metadata at the top of `selection_report.txt`.

**Report lines to add**

```python
f.write(f"PREPROC_MODE: {args.preproc or 'profile'}\n")
f.write(f"PREPROC_PARAMS: hpf_hz={params.get('hpf_hz')}, "
        f"envelope_median_ms={params.get('envelope_median_ms')}\n")
# If auto used, include the decision/metrics once (e.g., averaged or sample line)
if 'AUTO-PREPROC' in ''.join(report_lines):
    f.write(next(line for line in report_lines if 'AUTO-PREPROC' in line) + "\n")
```

**Checkpoint**

* Run a profile that sets `hpf_hz` and confirm the report shows `PREPROC_MODE` and `PREPROC_PARAMS`.
* In `auto` mode, confirm the decision and metrics are logged.

**Acceptance Criteria**

* Reproducible runs: the report clearly states whether and how preprocessing was applied.

-- DONE -- 

---

### 6.4 (Optional) Exporting Preprocessed Audio

**Tasks for Agent**

* Add `--export_preprocessed` to let users opt into exporting the processed beam.

**Sketch**

```python
parser.add_argument("--export_preprocessed", action="store_true",
    help="Export preprocessed beam WAVs instead of originals.")

# when writing WAVs
out_audio = beam_for_indices if args.export_preprocessed else beam
```

**Checkpoint**

* Two runs on the same input (default vs `--export_preprocessed`) produce different WAVs only in the second case.

**Acceptance Criteria**

* Default export remains unchanged; opting in works as intended.

-- DONE -- 

---

## 7) Add `--profile` CLI Argument and Route Parameters

**Tasks for Agent**
- Edit `beamforming-export-scikit-maad.py`:
  - Add `--profile` with choices.
  - `from beamforming_utils.config import get_profile`
  - Pass weights/params to `calculate_uniqueness_metrics` and `smart_direction_selection`.

**Code to Insert**
```python
parser.add_argument("--profile", type=str, default="none",
    choices=["wind","rain","surf_river","thunder","geophony_general","none"],
    help="Geophony profile to tune indices/selection.")

from beamforming_utils.config import get_profile
weights, params = get_profile(args.profile)

uniqueness_scores, df_all = calculate_uniqueness_metrics(
    ...,
    profile_weights=weights,
    ADI_dB_threshold=params.get("ADI_dB"),
    preproc=params
)

selected = smart_direction_selection(
    ...,
    correlation_threshold=params["corr"],
    min_angular_separation_deg=params["min_angle"],
    ...
)
```

**Checkpoint**
```bash
python beamforming-export-scikit-maad.py --help
```
- Help text shows `--profile` choices.

**Acceptance Criteria**
- CLI accepts and exposes the new option.

--- DONE ---

---

## 8) Annotate the Report With Active Profile and Weights

**Tasks for Agent**
- Update `beamforming_utils/beamforming_report.py` (or report generator) to include profile metadata in `selection_report.txt`:
  - profile name
  - weights
  - ADI threshold
  - correlation threshold
  - min-angle
  - pre-processing params

**Sketch**
```python
f.write(f"PROFILE: {profile_name}\n")
f.write(f"Weights: {weights}\n")
f.write(f"Params: ADI_dB={params['ADI_dB']}, corr={params['corr']}, min_angle={params['min_angle']}, hpf={params['hpf_hz']}\n")
```

**Checkpoint**
- Run once with any profile; open the report and verify header shows profile and parameters.

**Acceptance Criteria**
- Report includes profile metadata for reproducibility.

-- DONE -- 

---

## 9) Quick Sanity Tests Without Audio

**Tasks for Agent**
- Add a tiny test that validates profile dicts exist and weights sum to 1.0.

**Code Sketch**
```python
# tests/test_profiles.py
from beamforming_utils.config import get_profile

def test_profile_weights_exist():
    for p in ["wind","rain","surf_river","thunder","geophony_general","none"]:
        w, params = get_profile(p)
        assert set(w.keys()) == {"Hf","ADI","TEMP","ACI","SPATIAL"}
        assert round(sum(w.values()), 6) == 1.0
        assert {"ADI_dB","corr","min_angle"}.issubset(params.keys())
```

**Checkpoint**
```bash
pytest -k test_profiles -q
```

**Acceptance Criteria**
- Tests pass and weights sum correctly.

-- DONE --

---

## 10) Smoke Test With Real Audio (Per Profile)

**Tasks for Agent**
- Run the pipeline against the same short test file with each profile and compare outputs.

**Commands**
```bash
for P in wind rain surf_river thunder geophony_general none; do
  python beamforming-export-scikit-maad.py test.wav --duration 20 --profile $P --output out_$P
done
```

**Checkpoints**
- Each run creates outputs in separate directories.
- In `selection_report.txt`: confirm profile header, thresholds, and differing rankings vs `none`.

**Acceptance Criteria**
- Per-profile runs succeed; `maad_indices.csv` reflects expected shifts (e.g., higher ADI contribution under `surf_river`).

-- DONE --

---

## 11) Targeted Functional Checks

**Tasks for Agent**
1. **Weighting changes totals**: For a fixed direction `D`, compare `total_score(D)` under `none` vs `wind`. Verify numeric change matches reweighted sum.
2. **ADI threshold routing**: Report shows ADI dB matches profile.
3. **Pre-processing effect**: Log LF energy <120 Hz pre/post HPF for `wind` on one beam and verify reduction.

**Sketch (debug log)**
```python
if preproc and preproc.get("hpf_hz"):
    # compute and log LF energy reduction
    pass
```

**Acceptance Criteria**
- Evidence of all three checks in logs/report snippets.

-- DONE --

---

## 12) Document the Feature (README)

**Tasks for Agent**
- Add a “Geophony Profiles” section with:
  - Available profiles and when to use them.
  - CLI examples.
  - Note about report provenance.

**Acceptance Criteria**
- README updated with a concise, clear section.

-- DONE --

---

## 13) Optional: Auto-Suggest Profile

**Tasks for Agent**
- Implement a lightweight analyzer (LF ratio, crest factor, modulation cues) that **suggests** a profile; never enforce it.
- Log the suggestion near the start of processing.

**Acceptance Criteria**
- Suggestion appears in logs when enabled; manual override always respected.

---

## 14) Final Review and PR

**Tasks for Agent**
- Run full pipeline on several clips (windy field, rain, river).
- Open a PR with code diffs, sample reports, and brief rationale per profile.

**Acceptance Criteria**
- PR includes: code changes, test outputs, selection plots/screenshots, and report headers demonstrating profile metadata.

---

## 15) Fast Rollback

**Commands**
```bash
git checkout main
git reset --hard origin/main
# or
git restore -SW .
git clean -fd
```

**Acceptance Criteria**
- Working tree restored to mainline state.

---

## Reference Weights (sum = 1.0)

| Profile            | Hf  | ADI | 1–Ht | ACI | Spatial |
|--------------------|-----|-----|------|-----|---------|
| wind               | .15 | .10 | .30  | .15 | .30     |
| rain               | .10 | .15 | .30  | .10 | .35     |
| surf_river         | .10 | .30 | .10  | .20 | .30     |
| thunder            | .05 | .05 | .40  | .15 | .35     |
| geophony_general   | .10 | .20 | .30  | .10 | .30     |
| none (baseline)    | use existing project defaults |

**Key params per profile**

| Profile          | ADI_dB | Corr | MinAngle° | HPF Hz | EnvMedian ms |
|------------------|--------|------|-----------|--------|--------------|
| wind             | -30    | .60  | 45        | 120    | –            |
| rain             | -25    | .60  | 45        | –      | 15           |
| surf_river       | -30    | .65  | 40        | 60     | –            |
| thunder          | -35    | .55  | 50        | –      | –            |
| geophony_general | -30    | .60  | 45        | 90     | –            |
