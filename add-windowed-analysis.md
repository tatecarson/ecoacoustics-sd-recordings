## Goal

Add **windowed analysis** (e.g., 30 s windows with hop) and **pooled scoring** to your Option‑2 pipeline so you can rank beams over the whole file while still exporting the **best window** per selected beam.

Each step below tells you:

* **What to implement**
* **What you should observe** when it’s working
* A **GitHub Copilot prompt** you can paste into Copilot Chat or a code comment to generate the change.

---

## Step 1 — Add CLI options for windowing and pooling

**Implement**

* New flags: `--window`, `--hop`, `--pool {mean,p80,meanp80}`, `--pool_alpha`, `--highlights`, `--min_rms_db`.
* Profile defaults: `eco → meanp80, alpha=0.5`; `water → meanp80, alpha=0.3`.

**You should see**

* `python main.py -h` lists the new options.
* Running `python main.py --window 30 --hop 15 --pool meanp80 --pool_alpha 0.5` doesn’t error (even if not yet used downstream).

**Copilot prompt**

> Add argparse options to main.py: `--window` (float, default 30.0), `--hop` (float, default 15.0), `--pool` with choices `mean,p80,meanp80` (default meanp80), `--pool_alpha` (float 0..1, default 0.5), `--highlights` (int, default 0), and `--min_rms_db` (float, default -60.0). For profile `water`, set default pool\_alpha to 0.3 unless explicitly provided.

---

## Step 2 — Create a window iterator

**Implement**

* Utility `iter_windows(total_duration_s, start_time_s, window_s, hop_s)` returning a list of `(start_sample, end_sample, start_time_s)` that never exceeds file end; if the file is shorter than `window_s`, return one window.

**You should see**

* Printing the returned list for a 120 s file with `window=30, hop=15` yields starts at `0, 15, 30, 45, 60, 75, 90`.

**Copilot prompt**

> Write a function `iter_windows(total_duration_s, start_time_s, window_s, hop_s, fs)` that returns a list of `(i0, i1, t0_s)` sample ranges. Do not pad the last window beyond the file; include a single window if the file is shorter than `window_s`.

---

## Step 3 — Factor per‑window analysis into `analyze_window(...)`

**Implement**

* Function that slices the multichannel ambisonic data for a window, beamforms to the configured grid, calls your **existing** `calculate_uniqueness_metrics(...)`, and returns:

  * `directions`, `direction_vectors`
  * `window_scores: dict[direction→total_score]`
  * `window_components_df` with columns: `direction, total_score, ACI, ADI, Hf, Ht, RMS, modulation, spatial_uniqueness, spatial_gradient, profile, diffuse_field`

**You should see**

* Calling `analyze_window` once produces a DataFrame with one row per direction and non‑NaN totals for most beams.

**Copilot prompt**

> Create `analyze_window(audio_mc, fs, i0, i1, profile, grid_mode, fib_count, cached_dirvecs=None)` that: (1) slices audio, (2) beamforms to directions (lat/long or Fibonacci), (3) calls `calculate_uniqueness_metrics` once, (4) returns a dict of total scores per direction and a DataFrame of components. Reuse cached direction vectors if provided.

---

## Step 4 — Loop over windows and collect a score table

**Implement**

* In the main flow, iterate windows from Step 2.
* For each window, call `analyze_window`. Fill a **scores table** `scores_df` with rows indexed by `window_start_s`, columns = `direction`, values = `total_score`.
* Also collect `all_rows` (per‑window component frames) into `maad_indices_all_windows.csv`.

**You should see**

* `maad_indices_all_windows.csv` contains multiple windows × directions.
* Console logs show “Step 2: Analyzing uniqueness metrics...” repeating per window.

**Copilot prompt**

> In `analyze_and_export_best_directions`, iterate windows using `iter_windows`. For each window, call `analyze_window` and append: (a) a Series of total scores to a list for building a DataFrame (`scores_df`), and (b) the components DataFrame to `all_rows`. After the loop, concatenate `all_rows` into `maad_indices_all_windows.csv` and build `scores_df` with index=window\_start\_s, columns=directions.

---

## Step 5 — Add RMS gating per beam per window

**Implement**

* Convert RMS to dBFS‑like value (relative to max of 1.0). Where `< --min_rms_db`, set that window’s score to `NaN` (and optionally mark in the all‑windows table).

**You should see**

* Very quiet beams/windows disappear from pooled stats.
* Log lines summarize how many cells were gated.

**Copilot prompt**

> Before storing scores for a window, compute dBFS for each beam as `20*log10(max(RMS/1.0, 1e-12))`. If `< min_rms_db`, set the score to `np.nan` and mark a `rms_gated=True` column in the components DataFrame.

---

## Step 6 — Implement pooling (`mean`, `p80`, `meanp80`)

**Implement**

* Function `pool_scores(scores_df, method, alpha)` returning:

  * `pooled_scores: pd.Series` (per direction)
  * `best_window_start: dict[direction→t0_s]` via `idxmax` on the column.

**You should see**

* For a synthetic constant score matrix, `mean ≈ p80 ≈ meanp80`.
* For a matrix with one peaked window, `p80 > mean` and `best_window_start` equals that window.

**Copilot prompt**

> Implement `pool_scores(scores_df, method='meanp80', alpha=0.5)` that computes: `mean = df.mean(axis=0)`, `p80 = df.quantile(0.8, axis=0)`, and returns `pooled = mean` / `p80` / `alpha*mean + (1-alpha)*p80` depending on `method`. Also return a dict of best window per direction using `df.idxmax(axis=0)`.

---

## Step 7 — Build a pooled “uniqueness” list for selection

**Implement**

* Transform `pooled_scores` into the same structure your `smart_direction_selection` expects: a list of dicts with keys `direction`, `total_score`, and (optionally) `rms_power=0.0`.

**You should see**

* Selection runs without recomputing indices; rankings reflect pooled scores.

**Copilot prompt**

> Create `uniqueness_pooled = [{"direction": d, "total_score": float(pooled_scores[d]), "rms_power": 0.0} for d in pooled_scores.index]` and pass it to `smart_direction_selection` instead of per-window scores.

---

## Step 8 — Export the **best window** per selected beam

**Implement**

* Modify `export_selected_directions(...)` to accept `window_starts` and `window_duration_s`.
* Slice from `best_window_start_s` and export that window’s audio for each selected direction, not the original global segment.

**You should see**

* Exported WAV filenames correspond to selected `direction` and have exactly `--window` seconds of audio from the best window region.

**Copilot prompt**

> Extend `export_selected_directions` signature with `window_starts: dict[str,float]` and `window_duration_s: float`. Use these to compute sample slices per direction and export the normalized audio segment from that time range.

---

## Step 9 — Correlation used in the report should reflect exported segments

**Implement**

* Recompute (or extract) a correlation submatrix for **just the exported segments** (concatenate per-beam audio from the best window, or compute on those slices only) and print that in the report.

**You should see**

* The correlation matrix shown in `selection_report.txt` matches what you’d get if you computed Pearson correlation directly on the exported WAVs.

**Copilot prompt**

> In `create_selection_report`, build the correlation submatrix using the audio corresponding to each exported direction’s best window slice (the same range used for export), not the original global correlation matrix.

---

## Step 10 — Write the pooled summary CSV

**Implement**

* `maad_indices_pooled.csv` with columns: `direction, pooled_score, pool_method, pool_alpha, best_window_start_s, mean_score, p80_score, stdev_score`.

**You should see**

* A single row per direction with pooled stats and the recommended export start time.

**Copilot prompt**

> After pooling, create `pooled_df` with columns `direction`, `pooled_score`, `pool_method`, `pool_alpha`, `best_window_start_s`, plus `mean_score`, `p80_score`, and `stdev_score`. Save as `maad_indices_pooled.csv`.

---

## Step 11 — Update logs and README text

**Implement**

* Console logs list window count, pool method, alpha, and a short summary (mean, p80 ranges).
* README: add a section “Windowed Analysis & Pooling”.

**You should see**

* Clear run header like: “Windows: 7 (30 s, hop 15 s), Pool: meanp80 α=0.5”.

**Copilot prompt**

> Add informative logging after pooling: number of windows, window/hop lengths, pool method and alpha, min/median/max of pooled scores. Update README with a short section describing windowing and pooling options.

---

## Step 12 — Quick synthetic tests

**Implement**

* Two short in‑memory tests (or tiny files):

  1. **Constant tone** → all pooling methods equal; best window should be the first.
  2. **Single event** in one window → `p80 > mean`; best window equals event location.

**You should see**

* Printed assertions pass; pooled behavior matches expectations.

**Copilot prompt**

> Add a `__main__` unit test block guarded by `--test` that synthesizes: (a) a constant beam with identical windows and checks `mean≈p80≈meanp80`; (b) a beam with a burst in one window and checks that `best_window_start` matches the burst window and that `p80 > mean`. Use small arrays and skip beamforming for the test.

---

## Step 13 — Water‑mode defaults for pooling

**Implement**

* If `--profile water` and user didn’t override pooling, set `pool=meanp80, pool_alpha=0.3`.

**You should see**

* Running with `--profile water` and no pool flags uses α=0.3 automatically (printed in logs).

**Copilot prompt**

> In CLI parsing, if `args.profile == 'water'` and the user didn’t pass `--pool_alpha`, set `args.pool = 'meanp80'` (if unset) and `args.pool_alpha = 0.3`. Reflect this in the startup log line.

---

## Step 14 — Performance sanity check

**Implement**

* Ensure **indices are computed once per window**; subsequent stages reuse scores/CSVs. Guard against accidental recompute in selection/reporting.

**You should see**

* No extra “Analyzing uniqueness metrics…” lines outside the window loop.
* Profiling indicates the window loop dominates runtime, not selection/report.

**Copilot prompt**

> Audit the code paths to ensure `calculate_uniqueness_metrics` is only called inside `analyze_window` within the window loop. Selection and report should consume cached scores/CSVs. Add comments and asserts to prevent accidental recomputation.

---

## Step 15 — End‑to‑end run and checklist

**Implement**

* Run a full file:

  ```bash
  python main.py --input ambi.wav --profile eco --window 30 --hop 15 \
                 --pool meanp80 --pool_alpha 0.5 --max_exports 5
  ```

**You should see**

* Files created under `ecoacoustic_analysis/<stem>/`:

  * `maad_indices_all_windows.csv`
  * `maad_indices_pooled.csv`
  * `maad_indices.csv` (exported subset)
  * `selection_report.txt`
  * `selection_analysis.png`
  * `exported_spectrograms.png`
  * `ecoacoustic_<direction>_<window>s.wav` for each selected beam
* Report lists the **best window start** per beam; correlation reflects exported segments.

**Copilot prompt**

> Generate a final checklist of expected output files and a verification function that asserts their existence and non-empty size after a run. Print a short summary with the top 5 pooled directions and their best-window start times.

---

## What to do if something looks wrong

* **All pooled scores similar:** increase window length or change pooling to `p80`; verify RMS gating is not too aggressive.
* **Exported segments not aligned:** confirm `best_window_start_s` is used in `export_selected_directions`.
* **Correlation in report seems off:** recompute correlation on the **exact exported slices**.
* **Performance slow:** reduce grid density or increase `hop`; cache direction vectors.
