# Project Overview — Creative Ecoacoustic Beamforming with Geophony Profiles

Purpose
- A pragmatic, artist‑oriented pipeline that steers 2nd‑order ambisonic recordings (9‑channel B‑format) toward many directions (“beams”), computes ecoacoustic indices once per beam, ranks beams by a composite “uniqueness” score, and exports a diverse, non‑redundant subset for creative composition, installation, and analysis.
- Not a biodiversity survey tool; indices are used as curatorial features to surface compelling spatial perspectives.

Core Capabilities
- Ambisonic beamforming to a 3D grid (lat/long or Fibonacci).
- Single‑pass computation of ecoacoustic indices with scikit‑maad (ACI, ADI, Hf, Ht) plus spatial uniqueness from cross‑beam correlation.
- Greedy selection with redundancy controls: correlation threshold and minimum angular separation.
- Geophony profiles that retune index weights/thresholds and optional analysis‑only preprocessing for context‑specific robustness (wind, rain, surf/river, thunder, general, none).
- Text and interactive HTML reports, spectrograms, CSV exports, and batch processing.

Repository Structure (selected)
- beamforming-export-scikit-maad.py — Main entry script (batch mode, CLI, reports, profiles, auto-suggestion).
- ambisonic_beamforming.py — Beamformer for 2nd‑order ambisonics (expected to provide AmbisonicBeamformer).
- beamforming_utils/
  - beamforming_analysis.py — Indices computation (single pass), uniqueness scoring, selection, preprocessing (maybe_preprocess), profile suggestion.
  - beamforming_export.py — Writing selected/non‑selected beam WAVs.
  - beamforming_grid.py — Direction grids: lat/long and Fibonacci; angular utils.
  - beamforming_report.py — Text report, visualization PNG, interactive HTML report, spectrograms.
  - config.py — Defaults for grids, thresholds, weights, and geophony profile registry via get_profile.
- run_profiles.sh — Batch runner across all profiles; produces per‑profile output dirs.
- tests/test_profiles.py — Sanity tests for profile registry/weights (referenced in docs).
- out_*/ ecoacoustic_analysis/* — Example output directories and generated artifacts.
- README.md — Detailed usage, rationale, and profile documentation.
- geophony_profiles_implementation_steps.md — Implementation plan with numbered steps and acceptance criteria.
- profile_batch_test_plan.md — Batch test checklist and script.

Data Flow (Option 2: compute indices once)
1) Load 9‑ch ambisonic WAV; segment by duration/start.
2) Beamform to a direction set (lat/long grid, Fibonacci, or beamformer default).
3) Optionally apply light analysis‑only preprocessing (per profile/CLI/auto) for index stability; keep original audio for export unless --export_preprocessed is used.
4) Compute scikit‑maad alpha indices per beam once; derive normalized components:
   - Hf → spectral activity
   - ADI → frequency diversity
   - 1 − Ht → temporal complexity
   - ACI → acoustic complexity
   - Spatial uniqueness → 1 − mean(|corr|) to others
5) Weighted sum (profile‑aware) → total uniqueness score.
6) Greedy selection with thresholds and redundancy filters (correlation and optional angular separation).
7) Export selected beams; save CSVs and reports; generate plots and optional HTML report.

Uniqueness Score
total = w_Hf·Hf_norm + w_ADI·ADI_norm + w_TEMP·(1 − Ht_norm) + w_ACI·ACI_norm + w_SPATIAL·spatial_uniqueness_norm
- Weights are profile‑dependent; each profile sums to 1.0.

Geophony Profiles
Profiles tune both the composite scoring and selection to different natural contexts. Each profile defines:
- Weights: contributions of Hf, ADI, 1−Ht, ACI, spatial uniqueness.
- Params: ADI_dB threshold, correlation threshold, min angular separation, and optional preprocessing such as HPF or envelope median filtering.

Available profiles
- wind: Emphasizes temporal complexity and spatial uniqueness; HPF@120 Hz typical.
- rain: Emphasizes temporal/spatial; envelope median smoothing for rain tick artifacts.
- surf_river: Emphasizes ADI; HPF@60 Hz.
- thunder: Emphasizes temporal complexity; no HPF default.
- geophony_general: Balanced profile; HPF@90 Hz.
- none: Baseline weights/thresholds from config.

Preprocessing (analysis‑only)
- maybe_preprocess(x, sr, params) applies optional high‑pass filter and/or envelope median smoothing to stabilize indices in challenging geophony; used only for metric computation unless --export_preprocessed is set.
- Modes:
  - --preproc off | force | auto, or default “profile‑driven” when parameters exist.
  - Auto considers LF ratio, spectral flatness, envelope kurtosis to decide.
- Provenance is recorded in selection_report.txt and the HTML report.

Profile Suggestion
- A quick forward‑beam analysis suggests a profile with a confidence score using heuristics (LF energy ratio, crest factor, spectral flatness, envelope stats).
- CLI:
  - Suggest only: --suggest-profile-only --input_file ... (prints suggestion and exits)
  - During runs: --suggest-profile off|always|if_none

Key CLI Workflows
- Single file:
  python beamforming-export-scikit-maad.py --input_file your_ambi_2nd_order.wav --profile wind --duration_seconds 30 --html_report
- Directory:
  python beamforming-export-scikit-maad.py --input_dir profile-audio-test --profile geophony_general --html_report
- Batch across all profiles (see run_profiles.sh):
  PROFILES="wind rain surf_river thunder geophony_general none"
  INPUT_DIR="profile-audio-test"
  for P in $PROFILES; do
    python beamforming-export-scikit-maad.py --input_dir "$INPUT_DIR" --profile "$P" --output_dir "out_${P}" --html_report
  done
- Suggest profile only:
  python beamforming-export-scikit-maad.py --input_file yourfile.wav --suggest-profile-only
- Override preprocessing:
  python beamforming-export-scikit-maad.py --input_file test.wav --profile none --preproc force --hpf_hz 100
  python beamforming-export-scikit-maad.py --input_file test.wav --profile rain --envelope_median_ms 20
- Export preprocessed audio instead of originals:
  add --export_preprocessed

Inputs and Assumptions
- 9‑channel, 2nd‑order ambisonic WAVs.
- AmbisonicBeamformer implementation available in ambisonic_beamforming.py, with beamform_to_directions or beamform_3d_directions.
- Python 3.9+; install: pip install scikit-maad pandas soundfile matplotlib scipy

Outputs and Artifacts
Per input file (nested under output_dir/input_stem/):
- maad_indices_all_directions.csv — indices for all beams (single‑pass).
- maad_indices.csv — indices for exported beams only.
- correlation_matrix.csv — cross‑beam correlations.
- selection_report.txt — detailed text report with profile and preprocessing provenance.
- selection_analysis.png — overview visualization (scores, correlation heatmap, components).
- exported_spectrograms.png — spectrograms of exported WAVs.
- spectrograms/ — per‑beam spectrogram images for HTML.
- ecoacoustic_<direction>_<dur>s.wav — exported beams (and optional rejected_ files if --export_all).
- analysis_report.html — interactive report when --html_report is set.

Profile Comparison Dashboard (profile_comparison_general-geo.html)
- Purpose: A static, self‑contained dashboard to quickly compare exported audio across geophony profiles for several sound types (general‑geo, none, water‑flow, wind). It helps audition selected directions, view spectrograms, and see compact ecoacoustic indices per beam—supporting rapid qualitative/quantitative comparison of profile behavior.
- What it shows:
  - A profile weights legend at the top with explicit numbers for Hf, ADI, 1−Ht, ACI, and Spatial (including the baseline “none”).
  - For each sound type, a grid of audio players per profile/direction with spectrogram thumbnails.
  - Inline indices line per audio (ACI, ADI, Hf, Ht, and Total if present) loaded from the run’s CSVs.
- How indices are loaded:
  - For each (profile, soundType), it auto‑detects the analysis subfolder by probing for analysis_report.html under:
    out_<profile>/<soundType>/<candidate>/
    Candidates include common stems (general‑geo, none, water‑flow, wind), then the soundType label, then the profile name. This resolves layouts like out_rain/wind/wind/.
  - Prefers maad_indices.csv (exported/selected), falls back to maad_indices_all_directions.csv, and matches the CSV “direction” column to the grid’s direction label.
- Expected directory structure:
  - Audio: out_<profile>/<soundType>/ecoacoustic_<direction>_<dur>s.wav
  - Spectrograms: out_<profile>/<soundType>/spectrograms/…_spectrogram.png
  - Reports/CSVs (same folder as analysis_report.html): out_<profile>/<soundType>/<input_stem>/maad_indices*.csv
- Usage:
  - Generate outputs (e.g., run_profiles.sh). Then open profile_comparison_general-geo.html in a browser. The weights render immediately; indices populate after a brief fetch. If CSVs are missing for a case, the cell shows “Indices: n/a”.
- Notes:
  - If you change config profile weights, update the weightsMap constants in the HTML so the legend matches config.

Testing and Validation
- Lightweight profile registry test:
  pytest tests/test_profiles.py -v
- Batch test plan:
  See profile_batch_test_plan.md for a run script across profiles and the checklist for HTML report content, provenance, suggestion correctness, and ranking changes.
- Suggested functional checks in reports:
  - DEBUG lines record ADI_dB threshold, pre/post LF energy when HPF applied, and example total_score for direction_0.

Design Notes
- Single‑pass indices enable reproducibility: tables, selection, and plots share the same computed values.
- Selection is interpretable and tunable; future optimization (e.g., submodular, ILP) can be explored.
- Profiles are additive and non‑destructive: they route weights/thresholds and optional preprocessing without altering default exports unless opted in.

Typical Next Steps
- For coarse grids, try grid_mode=fibonacci with higher FIBONACCI_COUNT for smoother coverage.
- If selections are too similar, lower correlation threshold or increase angular separation.
- If too few selections, reduce minimum uniqueness or raise max exports.
- For windy/rainy/surf contexts, prefer the closest profile and consider --preproc auto.

References
- README.md for rationale, detailed explanations of indices, and creative usage guidance.
- geophony_profiles_implementation_steps.md for stepwise implementation with acceptance criteria.
- profile_batch_test_plan.md for end‑to‑end testing across profiles.

GitHub
- Remote: origin https://github.com/tatecarson/ecoacoustics-sd-recordings.git
