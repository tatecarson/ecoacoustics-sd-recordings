# Geophony Profile Batch Test Plan

This document provides a comprehensive checklist and shell script for testing the geophony profile feature across multiple files and profiles.

---

## 1. Batch Processing Script

Run the following shell script to process all WAV files in `profile-audio-test/` with every geophony profile and generate HTML reports:

```bash
PROFILES="wind rain surf_river thunder geophony_general none"
INPUT_DIR="profile-audio-test"

for P in $PROFILES; do
    python beamforming-export-scikit-maad.py \
        --input_dir "$INPUT_DIR" \
        --profile "$P" \
        --output_dir "out_${P}" \
        --html_report
done
```

- Each profile's outputs will be in a separate directory (e.g., `out_wind`, `out_rain`, etc.).
- The `--html_report` flag ensures that an HTML report is generated for each batch.

-- ran this and checking results manually --

---

## 2. Auto-Suggestion Check

For each file, run:

```bash
python beamforming-export-scikit-maad.py --input_file profile-audio-test/yourfile.wav --suggest-profile-only
```

- **Verify:** The suggested profile matches your expectation for the file's geophony.

---

## 3. Manual Override and CLI Robustness

Test CLI overrides:

```bash
python beamforming-export-scikit-maad.py --input_file profile-audio-test/yourfile.wav --profile none --hpf_hz 100
python beamforming-export-scikit-maad.py --input_file profile-audio-test/yourfile.wav --profile rain --envelope_median_ms 20
```

- **Verify:** The overrides are reflected in the report.

---

## 4. Report and Output Checks

For each output directory and file:
- Open `selection_report.txt`, `maad_indices.csv`, and the HTML report (typically `selection_report.html`).
- **Check:**
  - The report header (in both text and HTML) lists the correct profile, weights, and parameters.
  - The HTML report displays a summary section at the top with geophony profile metadata and the profile suggestion/provenance.
  - Index weights and thresholds match the profile.
  - Preprocessing provenance is logged in both reports.
  - The HTML report matches or exceeds the text report in terms of profile and preprocessing provenance.
  - The ranking of selected directions changes sensibly between profiles.

---

## 5. Preprocessing and Auto Mode

- Run with `--preproc auto` and check for `AUTO-PREPROC` lines in the report.
- **Verify:** Preprocessing is only applied when appropriate.

---

## 6. Error Handling

- Try running with a non-ambisonic or corrupted file.
- **Verify:** The script fails gracefully and logs a clear error.

---

## 7. Summary Table (Optional)

Create a table to track results:

| File | Expected Geophony | Suggested Profile | Profiles Tested | Notes |
|------|-------------------|-------------------|----------------|-------|
|      |                   |                   |                |       |

---

## 8. Regression/Consistency

- Compare `maad_indices.csv` and `selection_report.txt` across profiles and files.
- **Verify:** Selected beams and index values shift in ways that make ecological sense.

---

## 9. Documentation

- Ensure all findings, issues, and observations are recorded in this markdown for future reference.
