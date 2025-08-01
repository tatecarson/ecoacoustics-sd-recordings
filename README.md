
# README — Creative Ecoacoustic Beamforming (Option 2, single‑pass indices)

## Purpose

This repository provides a **pragmatic, artist‑oriented pipeline** for steering a 2nd‑order ambisonic recording into many directions (“beams”), computing ecoacoustic indices, ranking beams by a composite uniqueness score, and exporting a diverse, non‑redundant subset for creative soundscape composition and analysis.

It is not a biodiversity survey tool. Instead, it uses ecological signal descriptors (e.g., ACI, ADI, spectral/temporal entropy) to guide creative beam selection—helping you find meaningful spatial perspectives in a complex soundfield for composition and installation work.

---

## Quick Start

1. Place your 9‑channel 2nd‑order ambisonic WAV in the project folder.
2. Ensure `ambisonic_beamforming.py` is present and importable.
3. In the main script, set:
   ```python
   input_file = "your_ambi_2nd_order.wav"
   GRID_MODE = "latlong"  # or "fibonacci"
   ```
4. Run:
   ```bash
   python main.py
   ```
5. Inspect outputs in:
   ```
   ecoacoustic_analysis/<input_stem>/
     correlation_matrix.csv
     maad_indices_all_directions.csv
     maad_indices.csv
     selection_report.txt
     selection_analysis.png
     exported_spectrograms.png
     ecoacoustic_<direction>_<dur>s.wav
   ```

---

## Geophony Profiles

The pipeline includes **geophony profiles** that optimize ecoacoustic analysis for different natural soundscape contexts. These profiles adjust index weights, thresholds, and preprocessing to better handle environmental sounds like wind, rain, rivers, and thunder.

### Available Profiles

| Profile | Best For | Key Features |
|---------|----------|--------------|
| `wind` | Windy environments, rustling vegetation | HPF@120Hz, emphasizes temporal complexity |
| `rain` | Rainfall, light precipitation | Envelope smoothing, balances ADI/spatial |
| `surf_river` | Water sounds, rivers, ocean surf | HPF@60Hz, high ADI weight (0.3) |
| `thunder` | Storms, low-frequency events | No HPF, high temporal weight (0.4) |
| `geophony_general` | Mixed natural environments | Moderate HPF@90Hz, balanced weights |
| `none` | Default behavior | Uses original project weights/thresholds |

### Profile Weights Comparison

Profiles reweight the uniqueness score components to emphasize different acoustic characteristics:

| Profile | Spectral Activity (Hf) | Frequency Diversity (ADI) | Temporal Complexity (1-Ht) | Acoustic Complexity (ACI) | Spatial Uniqueness |
|---------|------------------------|---------------------------|----------------------------|---------------------------|-------------------|
| `wind` | 0.15 | 0.10 | **0.30** | 0.15 | **0.30** |
| `rain` | 0.10 | 0.15 | **0.30** | 0.10 | **0.35** |
| `surf_river` | 0.10 | **0.30** | 0.10 | 0.20 | **0.30** |
| `thunder` | 0.05 | 0.05 | **0.40** | 0.15 | **0.35** |
| `geophony_general` | 0.10 | 0.20 | **0.30** | 0.10 | **0.30** |

### CLI Usage

```bash
# Use wind profile for windy recordings
python beamforming-export-scikit-maad.py --input_file windy_forest.wav --profile wind

# Compare profiles on the same recording
python beamforming-export-scikit-maad.py --input_file stream.wav --profile surf_river --output_dir out_river
python beamforming-export-scikit-maad.py --input_file stream.wav --profile none --output_dir out_baseline

# Override profile parameters
python beamforming-export-scikit-maad.py --input_file test.wav --profile wind --hpf_hz 150

# Control preprocessing explicitly
python beamforming-export-scikit-maad.py --input_file test.wav --profile wind --preproc off    # disable
python beamforming-export-scikit-maad.py --input_file test.wav --profile none --preproc force --hpf_hz 100  # force
```

### Preprocessing Features

Profiles can apply light preprocessing **for index calculation only** (exported WAVs remain unprocessed by default):

- **High-pass filtering:** Reduces low-frequency rumble from wind/handling (wind: 120Hz, surf_river: 60Hz, geophony_general: 90Hz)
- **Envelope smoothing:** Reduces rain tick artifacts (rain: 15ms median filter)
- **Auto-detection:** `--preproc auto` can detect when preprocessing is beneficial based on spectral characteristics

### Report Provenance

All profile settings and preprocessing decisions are logged in `selection_report.txt`:

```plaintext
PROFILE METADATA
--------------------------------------------------
PROFILE: wind
Weights: {'Hf': 0.15, 'ADI': 0.1, 'TEMP': 0.3, 'ACI': 0.15, 'SPATIAL': 0.3}
Params: ADI_dB=-30, corr=0.6, min_angle=45.0, hpf=120, envelope_median_ms=None

PREPROCESSING PROVENANCE
--------------------------------------------------
PREPROC_MODE: None
PREPROC_PARAMS: hpf_hz=120, envelope_median_ms=None
DEBUG: LF_energy<120Hz pre=0.6854 post=0.2093 (profile=wind, preproc=applied)
DEBUG: total_score(direction_0)=0.2953 (profile=wind)
DEBUG: ADI_dB threshold used: -30
```

This ensures **full reproducibility** and helps verify that profiles are working as expected.

### When to Use Each Profile

- **`wind`**: Outdoor recordings with significant low-frequency wind noise, rustling leaves, or handling artifacts
- **`rain`**: Light to moderate rainfall, drizzle, or recordings with repetitive water droplet sounds
- **`surf_river`**: Flowing water, ocean waves, streams, or waterfalls where frequency diversity is key
- **`thunder`**: Storm recordings, distant thunder, or any environment with important low-frequency events
- **`geophony_general`**: Mixed natural environments or when unsure which specific profile to use
- **`none`**: Controlled environments, indoor recordings, or when you prefer the original algorithm

### Profile Validation

You can test profile integrity with:

```bash
pytest tests/test_profiles.py -v
```

This validates that all profiles exist, weights sum to 1.0, and required parameters are present.

---

## Requirements

* Python 3.9+ (recommended)
* Install:
  `pip install scikit-maad pandas soundfile matplotlib scipy`
* `ambisonic_beamforming.py` must define `AmbisonicBeamformer` compatible with **2nd‑order (9‑channel)** ambisonic input.

---

## What the Script Does

### End‑to‑End Flow

1. **Load and segment** a 9‑channel, 2nd‑order ambisonic WAV.
2. **Generate a direction grid** (lat/long by default; Fibonacci supported).
3. **Beamform** the segment into one mono signal per direction.
4. **Compute indices once** per beam with scikit‑maad (ACI, ADI, Hf, Ht), plus **spatial uniqueness** from cross‑beam correlations.
5. **Normalize and weight** components into a **total uniqueness score** per beam.
6. **Select** up to *N* beams greedily by score while enforcing **redundancy checks**: maximum time‑series correlation and minimum angular separation.
7. **Export** selected beams as WAVs.
8. **Save CSVs** without recomputation:

   * all beams → `maad_indices_all_directions.csv`
   * selected beams → `maad_indices.csv`
9. **Write a report and plots** summarizing scores, correlations, components, and spectrograms.

### Why “Option 2” Matters

* **Indices are computed once** and reused everywhere.
* Selection and reporting use the **same numbers**, improving speed, reproducibility, and auditability.
* No extra passes over files to recompute indices.

---

The pipeline includes **geophony profiles** that optimize ecoacoustic analysis for different natural soundscape contexts. These profiles adjust index weights, thresholds, and preprocessing to better handle environmental sounds like wind, rain, rivers, and thunder.

### Available Profiles

| Profile | Best For | Key Features |
|---------|----------|--------------|
| `wind` | Windy environments, rustling vegetation | HPF@120Hz, emphasizes temporal complexity |
| `rain` | Rainfall, light precipitation | Envelope smoothing, balances ADI/spatial |
| `surf_river` | Water sounds, rivers, ocean surf | HPF@60Hz, high ADI weight (0.3) |
| `thunder` | Storms, low-frequency events | No HPF, high temporal weight (0.4) |
| `geophony_general` | Mixed natural environments | Moderate HPF@90Hz, balanced weights |
| `none` | Default behavior | Uses original project weights/thresholds |

### Profile Weights Comparison

Profiles reweight the uniqueness score components to emphasize different acoustic characteristics:

| Profile | Spectral Activity (Hf) | Frequency Diversity (ADI) | Temporal Complexity (1-Ht) | Acoustic Complexity (ACI) | Spatial Uniqueness |
|---------|------------------------|---------------------------|----------------------------|---------------------------|-------------------|
| `wind` | 0.15 | 0.10 | **0.30** | 0.15 | **0.30** |
| `rain` | 0.10 | 0.15 | **0.30** | 0.10 | **0.35** |
| `surf_river` | 0.10 | **0.30** | 0.10 | 0.20 | **0.30** |
| `thunder` | 0.05 | 0.05 | **0.40** | 0.15 | **0.35** |
| `geophony_general` | 0.10 | 0.20 | **0.30** | 0.10 | **0.30** |

### CLI Usage

```bash
# Use wind profile for windy recordings
python beamforming-export-scikit-maad.py --input_file windy_forest.wav --profile wind

# Compare profiles on the same recording
python beamforming-export-scikit-maad.py --input_file stream.wav --profile surf_river --output_dir out_river
python beamforming-export-scikit-maad.py --input_file stream.wav --profile none --output_dir out_baseline

# Override profile parameters
python beamforming-export-scikit-maad.py --input_file test.wav --profile wind --hpf_hz 150

# Control preprocessing explicitly
python beamforming-export-scikit-maad.py --input_file test.wav --profile wind --preproc off    # disable
python beamforming-export-scikit-maad.py --input_file test.wav --profile none --preproc force --hpf_hz 100  # force
```

### Preprocessing Features

Profiles can apply light preprocessing **for index calculation only** (exported WAVs remain unprocessed by default):

- **High-pass filtering:** Reduces low-frequency rumble from wind/handling (wind: 120Hz, surf_river: 60Hz, geophony_general: 90Hz)
- **Envelope smoothing:** Reduces rain tick artifacts (rain: 15ms median filter)
- **Auto-detection:** `--preproc auto` can detect when preprocessing is beneficial based on spectral characteristics

### Report Provenance

All profile settings and preprocessing decisions are logged in `selection_report.txt`:

```plaintext
PROFILE METADATA
--------------------------------------------------
PROFILE: wind
Weights: {'Hf': 0.15, 'ADI': 0.1, 'TEMP': 0.3, 'ACI': 0.15, 'SPATIAL': 0.3}
Params: ADI_dB=-30, corr=0.6, min_angle=45.0, hpf=120, envelope_median_ms=None

PREPROCESSING PROVENANCE
--------------------------------------------------
PREPROC_MODE: None
PREPROC_PARAMS: hpf_hz=120, envelope_median_ms=None
DEBUG: LF_energy<120Hz pre=0.6854 post=0.2093 (profile=wind, preproc=applied)
DEBUG: total_score(direction_0)=0.2953 (profile=wind)
DEBUG: ADI_dB threshold used: -30
```

This ensures **full reproducibility** and helps verify that profiles are working as expected.

### When to Use Each Profile

- **`wind`**: Outdoor recordings with significant low-frequency wind noise, rustling leaves, or handling artifacts
- **`rain`**: Light to moderate rainfall, drizzle, or recordings with repetitive water droplet sounds
- **`surf_river`**: Flowing water, ocean waves, streams, or waterfalls where frequency diversity is key
- **`thunder`**: Storm recordings, distant thunder, or any environment with important low-frequency events
- **`geophony_general`**: Mixed natural environments or when unsure which specific profile to use
- **`none`**: Controlled environments, indoor recordings, or when you prefer the original algorithm

### Profile Validation

You can test profile integrity with:

```bash
pytest tests/test_profiles.py -v
```

This validates that all profiles exist, weights sum to 1.0, and required parameters are present.

---

## Configuration (Short Reference)

* **Direction grid:** `GRID_MODE` = `latlong` (default) or `fibonacci`.
* **Lat/long density:** `AZ_STEP_DEG=30`, `EL_PLANES_DEG=(-45,0,45)`, `INCLUDE_POLES=True`.
* **Fibonacci density:** `FIBONACCI_COUNT=128` (increase for smoother coverage).
* **Selection:** `MAX_EXPORTS`, `MIN_UNIQUENESS_THRESHOLD`, `USE_CORRELATION_FILTER` + `CORRELATION_THRESHOLD`, `USE_MIN_ANGLE_FILTER` + `MIN_ANGULAR_SEPARATION_DEG`.
* **Indices parity:** `ADI_AEI_DB_THRESHOLD` (e.g., −50 dB; try −30 dB for some habitats).
* **Bands:** `NDSI_BIO=(1000,10000)`, `NDSI_ANTH=(0,1000)`.

---

## Acoustic Indices Used

* **ACI (Acoustic Complexity Index):** frame‑to‑frame amplitude change across frequencies; highlights modulated bio‑signals.
* **ADI (Acoustic Diversity Index):** Shannon diversity across frequency bands (1 kHz bins by default) with a dB threshold to suppress low‑level noise.
* **Hf (Spectral Entropy):** evenness of energy across frequencies; higher = broader spectral spread.
* **Ht (Temporal Entropy):** evenness of energy over time; higher = more uniform/steady.
  *Temporal complexity* in this framework is `1 − Ht_norm` (lower entropy → more rhythmic/eventful → higher complexity).
* **Spatial Uniqueness:** `1 − mean(|corr|)` to other beams (time‑series redundancy proxy).

**Robust Ht Handling:** The script falls back to waveform‑based `temporal_entropy` and, if necessary, a manual envelope‑entropy to avoid NaNs in very quiet or short beams.

---

## Ecoacoustic Indices as **Creative Filters**: Rationale and Evidence

### Concept

This framework uses ecoacoustic indices not only for long‑term monitoring but as **curatorial features**—filters that help you **find** and **frame** compelling spatial perspectives in a single (or small number of) recording(s). You compute indices on beamformed excerpts and use them to rank, de‑duplicate, and select beams for composition.

### Where This Fits in Ecoacoustics

* **Rapid acoustic appraisal:** Indices were introduced for quick comparisons and **short‑window** characterization of soundscapes, not exclusively for long time‑series.
* **Navigation and segmentation:** Multi‑index summaries are widely used to **navigate** long audio, **surface events**, and **jump** to salient segments—functionally a filtering role.
* **Event‑centric analysis:** Indices can be treated as features for **event detection/segmentation** within recordings, emphasizing **structure** rather than only trends.
* **Art–science practice:** Ecoacoustic descriptors are increasingly adopted to **inform composition**, exhibition, and sound art, encouraging translation between ecological structure and musical design.

### Practical Implications for This Project

* Treat indices as **wayfinding signals** for creative decisions (orchestration, texture, pacing, spatial staging).
* Combine **score‑based selection** (indices) with **redundancy checks** (correlation + angular spacing) to ensure distinct beams.
* Keep **listening‑in‑the‑loop**: indices point to candidate material; ears make the final call.

### Limitations and Cautions

* Indices are **proxies** of signal ecology; they are sensitive to confounds (wind, rain, insects, machinery).
* Absolute values depend on analysis parameters (STFT settings, thresholds, bandwidth); maintain **consistent settings** within a project.
* For compositional use, these sensitivities are acceptable and often **musically interesting**, but avoid reifying indices as biodiversity measures.

---

## How the Uniqueness Score is Computed

Each beam (direction) is evaluated using a **uniqueness score** that combines several normalized ecoacoustic indices and a spatial redundancy measure. The uniqueness score is a weighted sum of the following components, each normalized across all beams:

- **Spectral Activity (Hf):** Normalized spectral entropy, representing the evenness of energy across frequencies.
- **Frequency Diversity (ADI):** Normalized Acoustic Diversity Index, reflecting the distribution of energy across frequency bands.
- **Temporal Complexity (1 – Ht):** Inverted and normalized temporal entropy, so that lower entropy (more temporal structure) increases the score.
- **Acoustic Complexity (ACI):** Normalized Acoustic Complexity Index, highlighting modulated, articulated signals.
- **Spatial Uniqueness:** 1 minus the mean absolute correlation to all other beams, normalized. This penalizes beams that are highly redundant with others.

The formula is:

```
uniqueness_score = w₁·Hf_norm + w₂·ADI_norm + w₃·(1–Ht_norm) + w₄·ACI_norm + w₅·spatial_uniqueness_norm
```

where the weights (w₁–w₅) are set in the script and sum to 1.0 by default.

**Interpretation:**  
A higher uniqueness score means the beam is more distinct in its spectral, temporal, and spatial characteristics compared to other beams. This helps prioritize beams that are both ecologically and creatively interesting, while avoiding redundant or similar perspectives.

---

## How Correlation is Computed and What It Means

**Correlation** in this context measures the similarity between the time-series waveforms of different beams (directions). For every pair of beams, the Pearson correlation coefficient is calculated between their audio signals. This results in a correlation matrix, where each value ranges from −1 (perfectly inverted) to +1 (identical), and 0 means no linear relationship.

- **High correlation (close to 1 or −1):** The two beams capture very similar (or inverted) audio content, indicating redundancy.
- **Low correlation (close to 0):** The beams are acoustically distinct, capturing different spatial perspectives.

**Usage in Selection:**  
During beam selection, a maximum allowed correlation threshold is enforced. If a candidate beam is too highly correlated with any already-selected beam (i.e., their absolute correlation exceeds the threshold), it is rejected to ensure the exported set covers diverse, non-redundant perspectives.

The **spatial uniqueness** component of the uniqueness score is also derived from correlation:  
For each beam, spatial uniqueness is calculated as `1 − mean(|corr|)` to all other beams, then normalized. This rewards beams that are less similar to the rest of the field.

---

## Interpreting Results (For Creative Practice)

* **High ACI:** articulated gestures; use for rhythmic motifs or triggers.
* **High ADI:** rich band distribution; use for harmonic density or chord size.
* **High Hf:** bright/broad spectra; drive timbral brightness and wide filters.
* **Low Ht (high temporal complexity):** patterned/event‑dense; choose staccato envelopes or percussive granulation.
* **High spatial uniqueness:** distinct perspective; promote in the mix or place at spatial extremes.

---

## Tuning and Troubleshooting

* Too many similar beams → lower `CORRELATION_THRESHOLD` or raise `MIN_ANGULAR_SEPARATION_DEG`.
* Too few selections → lower `MIN_UNIQUENESS_THRESHOLD`, increase `MAX_EXPORTS`, relax filters.
* Ht returns NaN → the script’s fallback should fix it; if not, increase `duration_seconds` or gate very low‑RMS beams.
* Grid feels coarse → use `GRID_MODE="fibonacci"` with `FIBONACCI_COUNT ≥ 128`.

---

## Assumptions and Limitations

* Assumes 2nd‑order, 9‑channel ambisonic input decoded/steered correctly by your beamformer.
* Indices describe **signal ecology**, not species ID/abundance.
* Greedy selection is **interpretable** and fast but not globally optimal; submodular or ILP optimization can be added later.

---

## Glossary (Selected Terms)

* **Ambisonics:** spherical soundfield representation; decode to any direction.
* **Beamforming:** spatial filtering toward a given azimuth/elevation.
* **Beam:** mono signal obtained by steering to a direction.
* **Azimuth / Elevation:** horizontal/vertical angles (deg).
* **Fibonacci sphere:** near‑uniform sampling of directions.
* **ACI / ADI / Hf / Ht / NDSI:** common ecoacoustic indices; see references for definitions.

---

## Extending the Project

* **Local refinement:** re‑sample ±10–15° around top beams, then re‑select.
* **Quota by region:** enforce per‑octant or per‑ring quotas for spatial coverage.
* **Add indices:** incorporate AEI, BI, NDSI into the composite score with small weights.
* **Reproducibility:** write a YAML config and version stamp per run.

---

## Selected References and Further Reading

* Pieretti, N., Farina, A., & Morri, D. (2011). Acoustic Complexity Index (ACI).
* Pijanowski, B. C., et al. (2011). Soundscape ecology: principles, patterns, methods.
* Sueur, J., et al. (2008). Rapid acoustic survey for biodiversity appraisal.
* Towsey, M., et al. (various). Long‑duration eco‑acoustic analyses; index maps for navigation and event discovery.
* Reviews/Guides: “Guidelines for the use of acoustic indices” (Methods in Ecology & Evolution); ecoacoustic index user guides/tutorials (various).
* Monacchi, D., & Krause, B. (2017). Ecoacoustics and its Expression through the Voice of the Arts.


