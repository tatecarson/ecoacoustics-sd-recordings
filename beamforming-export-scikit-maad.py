import os
from pathlib import Path
import warnings
import math

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal

from maad import sound as maad_sound
from maad import features as maad_features

# Import the beamformer class (assumes ambisonic_beamforming.py is in same directory)
from ambisonic_beamforming import AmbisonicBeamformer


# =========================
# Configuration
# =========================
GRID_MODE = "fibonacci"   # "latlong", "fibonacci", or "beamformer_default"
AZ_STEP_DEG = 30        # used if GRID_MODE == "latlong"
EL_PLANES_DEG = (-45, 0, 45)
INCLUDE_POLES = True

FIBONACCI_COUNT = 128   # used if GRID_MODE == "fibonacci"

# Selection constraints
USE_CORRELATION_FILTER = True
CORRELATION_THRESHOLD = 0.70
USE_MIN_ANGLE_FILTER = True        # requires az/el data
MIN_ANGULAR_SEPARATION_DEG = 30.0  # applied only if we know az/el of beams

# Uniqueness score weights (should sum to 1.0)
W_ACTIVITY = 0.25      # Hf (spectral entropy, normalized)
W_FREQDIV = 0.25       # ADI (normalized)
W_TEMP = 0.15          # 1 - Ht (normalized)
W_ACI = 0.20           # ACI (normalized)
W_SPATIAL = 0.15       # 1 - mean|corr| (normalized)

# Indices thresholds/bands for scikit-maad
ADI_AEI_DB_THRESHOLD = -50
NDSI_BIO = (1000, 10000)
NDSI_ANTH = (0, 1000)

# Export/analysis defaults
DEFAULT_DURATION_SECONDS = 30
DEFAULT_START_TIME = 5
MAX_EXPORTS = 5
MIN_UNIQUENESS_THRESHOLD = 0.30


# =========================
# Direction grid helpers
# =========================
def latlong_grid(az_step_deg=30, el_planes_deg=(-45, 0, 45), include_poles=True):
    """
    Build a lat/long direction grid as (az_deg, el_deg) pairs.
    azimuth: 0..360-az_step
    elevation planes: list in degrees
    """
    dirs = []
    for el in el_planes_deg:
        for az in range(0, 360, int(az_step_deg)):
            dirs.append((float(az), float(el)))
    if include_poles:
        dirs.append((0.0, 90.0))
        dirs.append((0.0, -90.0))
    return dirs


def fibonacci_sphere_grid(n_points=128):
    """
    Generate ~uniform points on a sphere using Fibonacci lattice.
    Returns list of (az_deg, el_deg) where az is [0,360), el is [-90,90].
    """
    points = []
    phi = (1 + 5 ** 0.5) / 2
    for i in range(n_points):
        z = 1 - (2 * i + 1) / n_points
        r = math.sqrt(max(0.0, 1 - z * z))
        theta = 2 * math.pi * (i / phi % 1.0)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        # Convert to az/el
        az = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
        el = math.degrees(math.asin(z))
        points.append((az, el))
    return points


def unit_vector_from_azel(az_deg, el_deg):
    """
    Convert azimuth/elevation in degrees to a 3D unit vector.
    Azimuth 0° = +X by this math; adjust if your convention differs.
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = math.cos(el) * math.cos(az)
    y = math.cos(el) * math.sin(az)
    z = math.sin(el)
    v = np.array([x, y, z], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def angular_separation_deg(v1, v2):
    """Angle between two 3D unit vectors in degrees."""
    c = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.degrees(math.acos(c))


# =========================
# Core analysis pipeline
# (Option 2: compute indices once, reuse for CSVs and selection)
# =========================
def analyze_and_export_best_directions(
    input_file,
    output_dir="ecoacoustic_analysis",
    duration_seconds=DEFAULT_DURATION_SECONDS,
    start_time=DEFAULT_START_TIME,
    max_exports=MAX_EXPORTS,
    min_uniqueness_threshold=MIN_UNIQUENESS_THRESHOLD,
    correlation_threshold=CORRELATION_THRESHOLD,
    use_correlation_filter=USE_CORRELATION_FILTER,
    use_min_angle_filter=USE_MIN_ANGLE_FILTER,
    min_angular_separation_deg=MIN_ANGULAR_SEPARATION_DEG,
    grid_mode=GRID_MODE,
):
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    input_stem = Path(input_file).stem
    nested_output_path = output_path / input_stem
    nested_output_path.mkdir(exist_ok=True)

    print("SMART ECOACOUSTIC BEAMFORMING ANALYSIS")
    print("=" * 50)
    print(f"Loading ambisonic file: {input_file}")

    # Load ambisonic audio
    try:
        audio, sample_rate = sf.read(input_file)
        n_ch = audio.shape[1] if audio.ndim > 1 else 1
        print(f"Loaded: {audio.shape[0]/sample_rate:.1f}s, {n_ch} channels, {sample_rate} Hz")
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

    # Validate 9-channel 2nd-order ambisonics
    if audio.ndim != 2 or audio.shape[1] != 9:
        print(f"Error: Expected 9 channels for 2nd-order ambisonics, got {n_ch}")
        return []

    # Create beamformer
    beamformer = AmbisonicBeamformer(sample_rate)

    # Segment bounds
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + duration_seconds) * sample_rate)
    if end_sample > audio.shape[0]:
        end_sample = audio.shape[0]
        actual_duration = (end_sample - start_sample) / sample_rate
        print(f"Note: Using {actual_duration:.1f}s (less than requested {duration_seconds}s)")

    # Extract segment
    audio_segment = audio[start_sample:end_sample]
    print(f"Analyzing {audio_segment.shape[0]/sample_rate:.1f} seconds starting at {start_time}s\n")

    # Prepare direction grid (if supported)
    azel_list = None
    if grid_mode == "latlong":
        azel_list = latlong_grid(AZ_STEP_DEG, EL_PLANES_DEG, INCLUDE_POLES)
    elif grid_mode == "fibonacci":
        azel_list = fibonacci_sphere_grid(FIBONACCI_COUNT)
    elif grid_mode == "beamformer_default":
        azel_list = None
    else:
        print(f"Unknown GRID_MODE={grid_mode}; using beamformer's default.")
        azel_list = None

    # Beamforming
    print("Beamforming to 3D directions...")
    direction_vectors = None  # 3D unit vectors aligned with directions
    if azel_list is not None and hasattr(beamformer, "beamform_to_directions"):
        beamformed_audio, directions = beamformer.beamform_to_directions(audio_segment, azel_list)
        direction_vectors = [unit_vector_from_azel(az, el) for az, el in azel_list]
        if len(direction_vectors) != len(directions):
            print("Warning: direction_vectors length mismatch; disabling angular filter.")
            direction_vectors = None
    else:
        if azel_list is not None:
            print("Warning: beamformer.beamform_to_directions(...) not found; "
                  "falling back to beamformer's default direction set.")
        beamformed_audio, directions = beamformer.beamform_3d_directions(audio_segment)
        direction_vectors = None

    print(f"Beamformed to {len(directions)} directions.")

    # Step 1: Correlation matrix across all directions
    print("Step 1: Calculating correlation matrix...")
    correlation_matrix = calculate_correlation_analysis(beamformed_audio, directions, nested_output_path)

    # Step 2: Uniqueness metrics (single pass) -> reuse everywhere
    print("Step 2: Analyzing uniqueness metrics...")
    uniqueness_scores, indices_df_all = calculate_uniqueness_metrics(
        beamformed_audio, directions, sample_rate
    )

    # Save ALL-directions CSV directly (no recomputation)
    all_csv = nested_output_path / "maad_indices_all_directions.csv"
    indices_df_all.to_csv(all_csv, index=False)
    print(f"Saved indices for ALL directions to: {all_csv}")

    # Step 3: Selection (greedy, with correlation and optional angular spacing)
    print("Step 3: Selecting best directions for ecoacoustic analysis...")
    selected_directions = smart_direction_selection(
        beamformed_audio=beamformed_audio,
        directions=directions,
        correlation_matrix=correlation_matrix,
        uniqueness_scores=uniqueness_scores,
        max_exports=max_exports,
        min_uniqueness_threshold=min_uniqueness_threshold,
        correlation_threshold=correlation_threshold,
        use_correlation_filter=use_correlation_filter,
        use_min_angle_filter=use_min_angle_filter and (direction_vectors is not None),
        min_angular_separation_deg=min_angular_separation_deg,
        direction_vectors=direction_vectors,
    )

    # Step 4: Export selected directions
    print(f"Step 4: Exporting {len(selected_directions)} selected directions...")
    exported_files = export_selected_directions(
        beamformed_audio, directions, selected_directions, nested_output_path, duration_seconds, sample_rate
    )

    # Step 4a: Exported-beams CSV by subsetting cached indices (no recomputation)
    if selected_directions:
        exported_df = indices_df_all[indices_df_all["direction"].isin(selected_directions)].copy()
        out_csv = nested_output_path / "maad_indices.csv"
        exported_df.to_csv(out_csv, index=False)
        print(f"Saved scikit-maad indices table (exported beams) to: {out_csv}")
    else:
        print("No exported beams; skipping exported-beams CSV.")

    # Step 5: Plots of exported beams
    plot_exported_spectrograms(exported_files, nested_output_path, sample_rate)

    # Step 6: Selection report
    create_selection_report(
        selected_directions, uniqueness_scores, correlation_matrix, directions, nested_output_path, exported_files
    )

    # Append compact indices summary to report
    if selected_directions:
        report_file = nested_output_path / "selection_report.txt"
        with open(report_file, "a") as f:
            f.write("\n\nSCIKIT-MAAD ALPHA INDICES (per exported beam)\n")
            f.write("-" * 50 + "\n")
            cols = [c for c in exported_df.columns if c in ("direction", "ACI", "ADI", "AEI", "BI", "NDSI", "Hf", "Ht", "LEQf")]
            if cols:
                f.write(exported_df[cols].to_string(index=False))
                f.write("\n")

    # Step 7: Overview visualization
    create_selection_visualization(
        beamformed_audio,
        directions,
        selected_directions,
        uniqueness_scores,
        correlation_matrix,
        nested_output_path,
        sample_rate,
    )

    print("\nANALYSIS COMPLETE")
    print(f"Results saved to: {nested_output_path.absolute()}")
    print(f"{len(exported_files)} optimized files ready for ecoacoustic analysis")
    return exported_files


# =========================
# Analysis helpers
# =========================
def calculate_correlation_analysis(beamformed_audio, directions, output_path):
    """Calculate correlation matrix and save to CSV."""
    correlation_matrix = np.corrcoef(beamformed_audio, rowvar=False)
    correlation_file = output_path / "correlation_matrix.csv"
    np.savetxt(
        correlation_file,
        correlation_matrix,
        delimiter=",",
        header=",".join(directions),
        comments="",
    )
    return correlation_matrix


def calculate_uniqueness_metrics(beamformed_audio, directions, sample_rate):
    """
    Compute scikit-maad indices ONCE for each direction and reuse:
      - activity_variation  -> Hf  (spectral entropy; normalized)
      - frequency_diversity -> ADI (normalized)
      - temporal_complexity -> 1 - Ht (lower entropy => more temporal structure; normalized)
      - acoustic_complexity -> ACI (normalized)
      - spatial_uniqueness  -> 1 - mean |corr| to other beams (normalized)
    Returns:
      uniqueness_scores (list of dicts, sorted by total_score),
      indices_df_all (DataFrame with raw and normalized metrics per direction)
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    def _norm(arr):
        arr = np.asarray(arr, dtype=float)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros_like(arr, dtype=float)
        mn, mx = np.nanmin(arr[finite]), np.nanmax(arr[finite])
        if mx - mn <= 1e-12:
            out = np.zeros_like(arr, dtype=float)
        else:
            out = (arr - mn) / (mx - mn)
        out[~finite] = 0.0
        return out

    n_dirs = beamformed_audio.shape[1]
    if n_dirs != len(directions):
        raise ValueError("directions length must match beamformed_audio columns")

    print(f"Calculating scikit-maad indices for {n_dirs} directions...")

    ACI_vals = np.full(n_dirs, np.nan, dtype=float)
    ADI_vals = np.full(n_dirs, np.nan, dtype=float)
    Hf_vals  = np.full(n_dirs, np.nan, dtype=float)
    Ht_vals  = np.full(n_dirs, np.nan, dtype=float)
    RMS_vals = np.full(n_dirs, np.nan, dtype=float)

    # Precompute correlation matrix once for spatial uniqueness
    corr_mat = np.corrcoef(beamformed_audio, rowvar=False)

    for i, direction in enumerate(directions):
        try:
            s = np.asarray(beamformed_audio[:, i], dtype=np.float64).ravel()
            if not np.any(np.isfinite(s)) or np.max(np.abs(s)) == 0 or s.size < 128:
                raise ValueError("degenerate or too-short signal")

            # Choose STFT params that fit segment length
            nperseg = min(2048, max(256, int(2**np.floor(np.log2(max(256, s.size//2))))))
            noverlap = nperseg // 2

            # Power spectrogram for all_spectral_alpha_indices
            Sxx_power, tn, fn, _ = maad_sound.spectrogram(
                s,
                sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                mode="psd",
                detrend=False,
            )

            # Compute indices (soundecology-compatible flags)
            df, _ = maad_features.all_spectral_alpha_indices(
                Sxx_power,
                tn,
                fn,
                flim_low=(0, 1000),
                flim_mid=(1000, 10000),
                flim_hi=(10000, 20000),
                flim_bioPh=NDSI_BIO,
                flim_antroPh=NDSI_ANTH,
                R_compatible="soundecology",
                ADI_dB_threshold=ADI_AEI_DB_THRESHOLD,
                AEI_dB_threshold=ADI_AEI_DB_THRESHOLD,
                verbose=False,
                display=False,
            )

            # --- Safe extractors ---
            def _safe_get(name, default=np.nan):
                if name in df.columns:
                    val = df.at[0, name]
                    if isinstance(val, (np.ndarray, list, tuple)):
                        val = np.asarray(val).ravel()
                        return float(val[0]) if val.size else float(default)
                    try:
                        return float(val)
                    except Exception:
                        return float(default)
                return float(default)

            ACI_vals[i] = _safe_get("ACI", np.nan)
            ADI_vals[i] = _safe_get("ADI", np.nan)
            Hf_vals[i]  = _safe_get("Hf",  np.nan)

            # --- Robust Ht: try df, then maad temporal_entropy, then manual envelope entropy ---
            Ht_raw = _safe_get("Ht", np.nan)

            if not np.isfinite(Ht_raw):
                # 1) Try scikit-maad temporal_entropy on the waveform
                try:
                    Ht_raw = maad_features.temporal_entropy(s)
                except Exception:
                    Ht_raw = np.nan

            if not np.isfinite(Ht_raw):
                # 2) Manual fallback on smoothed amplitude envelope
                try:
                    env = np.abs(signal.hilbert(s))
                    # Savitzky-Golay smoothing with valid odd window
                    def _odd(n):  # ensure odd >= 5, <= len(env)
                        n = max(5, min(int(n), len(env) - (1 - len(env) % 2)))
                        return n + 1 if n % 2 == 0 else n
                    win = _odd(min(101, len(env)//4))
                    if win >= 5 and win < len(env):
                        env = signal.savgol_filter(env, window_length=win, polyorder=3, mode="interp")
                    # Safe probabilities
                    eps = 1e-12
                    p = env + eps
                    p = p / np.sum(p)
                    # Normalized Shannon entropy in [0,1]
                    Ht_raw = float(-np.sum(p * np.log(p)) / np.log(len(p))) if len(p) > 1 else 1.0
                except Exception:
                    Ht_raw = 1.0  # maximally uniform as a last resort

            Ht_vals[i] = Ht_raw

            # RMS (for diagnostics)
            RMS_vals[i] = float(np.sqrt(np.mean(s ** 2)))

            print(
                f"  {direction}: ACI={ACI_vals[i]:.4f}, ADI={ADI_vals[i]:.4f}, "
                f"Hf={Hf_vals[i]:.4f}, Ht={Ht_vals[i]:.4f}, RMS={RMS_vals[i]:.4f}"
            )

        except Exception as e:
            print(f"  Error computing indices for {direction}: {e}")


    # Spatial uniqueness from correlations: 1 - mean |corr| to others
    spatial_unique = np.zeros(n_dirs, dtype=float)
    for i in range(n_dirs):
        others = np.delete(np.abs(corr_mat[i, :]), i)
        spatial_unique[i] = 1.0 - np.nanmean(others) if others.size else 0.0

    # Normalize metrics across directions
    activity_variation  = _norm(Hf_vals)          # Hf
    frequency_diversity = _norm(ADI_vals)         # ADI
    temporal_complexity = 1.0 - _norm(Ht_vals)    # invert Ht
    acoustic_complexity = _norm(ACI_vals)         # ACI
    spatial_uniqueness  = _norm(spatial_unique)   # spatial uniqueness

    # Weighted total score
    total_score = (
        W_ACTIVITY * activity_variation
        + W_FREQDIV * frequency_diversity
        + W_TEMP    * temporal_complexity
        + W_ACI     * acoustic_complexity
        + W_SPATIAL * spatial_uniqueness
    )

    # Pack outputs: list of dicts for selection/plots
    uniqueness_scores = []
    for i, direction in enumerate(directions):
        uniqueness_scores.append(
            {
                "direction": direction,
                "total_score": float(total_score[i]) if np.isfinite(total_score[i]) else 0.0,
                "activity_variation": float(activity_variation[i]) if np.isfinite(activity_variation[i]) else 0.0,
                "frequency_diversity": float(frequency_diversity[i]) if np.isfinite(frequency_diversity[i]) else 0.0,
                "temporal_complexity": float(temporal_complexity[i]) if np.isfinite(temporal_complexity[i]) else 0.0,
                "acoustic_complexity": float(acoustic_complexity[i]) if np.isfinite(acoustic_complexity[i]) else 0.0,
                "spatial_uniqueness": float(spatial_uniqueness[i]) if np.isfinite(spatial_uniqueness[i]) else 0.0,
                "rms_power": float(RMS_vals[i]) if np.isfinite(RMS_vals[i]) else 0.0,
            }
        )
    uniqueness_scores.sort(key=lambda x: x["total_score"], reverse=True)

    # DataFrame for all directions (raw + normalized components + total)
    rows = []
    for i, direction in enumerate(directions):
        rows.append({
            "direction": direction,
            "ACI": float(ACI_vals[i]) if np.isfinite(ACI_vals[i]) else np.nan,
            "ADI": float(ADI_vals[i]) if np.isfinite(ADI_vals[i]) else np.nan,
            "Hf":  float(Hf_vals[i])  if np.isfinite(Hf_vals[i])  else np.nan,
            "Ht":  float(Ht_vals[i])  if np.isfinite(Ht_vals[i])  else np.nan,
            "spatial_unique": float(spatial_unique[i]) if np.isfinite(spatial_unique[i]) else np.nan,
            "rms_power": float(RMS_vals[i]) if np.isfinite(RMS_vals[i]) else np.nan,
            "activity_variation": float(activity_variation[i]),
            "frequency_diversity": float(frequency_diversity[i]),
            "temporal_complexity": float(temporal_complexity[i]),
            "acoustic_complexity": float(acoustic_complexity[i]),
            "spatial_uniqueness": float(spatial_uniqueness[i]),
            "total_score": float(total_score[i])
        })
    indices_df_all = pd.DataFrame(rows)

    print("Uniqueness metrics (maad) analysis complete.")
    return uniqueness_scores, indices_df_all


def smart_direction_selection(
    beamformed_audio,
    directions,
    correlation_matrix,
    uniqueness_scores,
    max_exports,
    min_uniqueness_threshold,
    correlation_threshold,
    use_correlation_filter=True,
    use_min_angle_filter=False,
    min_angular_separation_deg=30.0,
    direction_vectors=None,  # list of 3D unit vectors aligned with 'directions'
):
    """
    Select the most valuable directions for ecoacoustic analysis, with:
      1) correlation filter (time-series redundancy)
      2) minimum angular separation filter (geometric diversity)
    """
    selected_directions = []
    selected_indices = []

    print("Selection criteria:")
    print(f"  • Max exports: {max_exports}")
    print(f"  • Min uniqueness: {min_uniqueness_threshold:.2f}")
    if use_correlation_filter:
        print(f"  • Max correlation between selections: {correlation_threshold:.2f}")
    if use_min_angle_filter and direction_vectors is not None:
        print(f"  • Min angular separation: {min_angular_separation_deg:.1f}°")
    print()

    dir_to_idx = {direction: i for i, direction in enumerate(directions)}

    # Greedy by score
    for score_data in uniqueness_scores:
        direction = score_data["direction"]
        uniqueness = score_data["total_score"]

        if uniqueness < min_uniqueness_threshold:
            print(f"{direction}: Below uniqueness threshold ({uniqueness:.3f} < {min_uniqueness_threshold})")
            continue
        if len(selected_directions) >= max_exports:
            print(f"Reached maximum exports ({max_exports})")
            break

        current_idx = dir_to_idx[direction]
        ok = True

        # Correlation filter
        if use_correlation_filter:
            for selected_dir in selected_directions:
                selected_idx = dir_to_idx[selected_dir]
                correlation = abs(correlation_matrix[current_idx, selected_idx])
                if correlation > correlation_threshold:
                    print(f"{direction}: Too correlated with {selected_dir} (r={correlation:.3f})")
                    ok = False
                    break

        # Angular separation filter
        if ok and use_min_angle_filter and direction_vectors is not None:
            v_cur = direction_vectors[current_idx]
            for selected_dir in selected_directions:
                v_sel = direction_vectors[dir_to_idx[selected_dir]]
                ang = angular_separation_deg(v_cur, v_sel)
                if ang < min_angular_separation_deg:
                    print(f"{direction}: Too close in angle to {selected_dir} ({ang:.1f}° < {min_angular_separation_deg}°)")
                    ok = False
                    break

        if ok:
            selected_directions.append(direction)
            selected_indices.append(current_idx)
            print(f"{direction}: Selected (uniqueness={uniqueness:.3f}, power={score_data['rms_power']:.4f})")

    print(f"\nSelected {len(selected_directions)} directions for export: {selected_directions}")
    return selected_directions


def export_selected_directions(
    beamformed_audio, directions, selected_directions, output_path, duration_seconds, sample_rate
):
    """Export only the selected directions as mono WAV files."""
    exported_files = []
    dir_to_idx = {direction: i for i, direction in enumerate(directions)}
    for direction in selected_directions:
        idx = dir_to_idx[direction]
        filename = f"ecoacoustic_{direction}_{duration_seconds}s.wav"
        filepath = output_path / filename
        direction_audio = beamformed_audio[:, idx]
        max_val = np.max(np.abs(direction_audio))
        if max_val > 0:
            direction_audio = direction_audio / max_val * 0.95
        sf.write(filepath, direction_audio, sample_rate)
        exported_files.append(filepath)
        rms_level = np.sqrt(np.mean(direction_audio**2))
        print(f"  {direction}: {filename} (RMS: {rms_level:.4f})")
    return exported_files


def create_selection_report(
    selected_directions, uniqueness_scores, correlation_matrix, all_directions, output_path, exported_files
):
    """Create a detailed report explaining the selection decisions."""
    report_file = output_path / "selection_report.txt"
    dir_to_idx = {direction: i for i, direction in enumerate(all_directions)}
    score_lookup = {s["direction"]: s for s in uniqueness_scores}

    with open(report_file, "w") as f:
        f.write("ECOACOUSTIC BEAMFORMING SELECTION REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"SELECTED DIRECTIONS FOR ANALYSIS ({len(selected_directions)}):\n")
        f.write("-" * 40 + "\n")

        for i, direction in enumerate(selected_directions):
            score_data = score_lookup[direction]
            f.write(f"\n{i+1}. {direction.upper()}\n")
            f.write(f"   File: {exported_files[i].name}\n")
            f.write(f"   Total Uniqueness Score: {score_data['total_score']:.4f}\n")
            f.write(f"   RMS Power Level: {score_data['rms_power']:.4f}\n")
            f.write("   Component Scores:\n")
            f.write(f"     • Spectral Activity (Hf): {score_data['activity_variation']:.4f}\n")
            f.write(f"     • Frequency Diversity (ADI): {score_data['frequency_diversity']:.4f}\n")
            f.write(f"     • Temporal Complexity (1 - Ht): {score_data['temporal_complexity']:.4f}\n")
            f.write(f"     • Acoustic Complexity (ACI): {score_data['acoustic_complexity']:.4f}\n")
            f.write(f"     • Spatial Uniqueness: {score_data['spatial_uniqueness']:.4f}\n")

        f.write("\n\nCORRELATION ANALYSIS:\n")
        f.write("-" * 25 + "\n")
        f.write("Correlation matrix between selected directions:\n")
        selected_indices = [dir_to_idx[direction] for direction in selected_directions]
        selected_corr = correlation_matrix[np.ix_(selected_indices, selected_indices)]
        f.write(f"{'':>12}")
        for direction in selected_directions:
            f.write(f"{direction:>8}")
        f.write("\n")
        for i, direction in enumerate(selected_directions):
            f.write(f"{direction:>12}")
            for j in range(len(selected_directions)):
                f.write(f"{selected_corr[i, j]:>8.3f}")
            f.write("\n")

        f.write("\n\nREJECTED DIRECTIONS:\n")
        f.write("-" * 20 + "\n")
        rejected_directions = [d for d in all_directions if d not in selected_directions]
        for direction in rejected_directions:
            score_data = score_lookup[direction]
            # The thresholds in the report reflect the ones used in selection
            if score_data["total_score"] < MIN_UNIQUENESS_THRESHOLD:
                reason = f"Low uniqueness ({score_data['total_score']:.3f})"
            else:
                dir_idx = dir_to_idx[direction]
                max_corr = 0
                corr_with = ""
                for sel_dir in selected_directions:
                    sel_idx = dir_to_idx[sel_dir]
                    corr = abs(correlation_matrix[dir_idx, sel_idx])
                    if corr > max_corr:
                        max_corr = corr
                        corr_with = sel_dir
                if max_corr > CORRELATION_THRESHOLD:
                    reason = f"Too correlated with {corr_with} (r={max_corr:.3f})"
                else:
                    reason = "Exceeded max exports"
            f.write(f"   {direction}: {reason}\n")

        f.write("\n\nRECOMMENDATIONS FOR ECOACOUSTIC ANALYSIS:\n")
        f.write("-" * 45 + "\n")
        f.write("1. Start analysis with the highest-ranked direction.\n")
        f.write("2. Use maad_indices.csv to compare ACI, ADI, AEI, BI, NDSI, Hf, Ht across beams.\n")
        f.write("3. Compare results between directions to understand spatial patterns.\n")
        f.write("4. Look for temporal patterns in high complexity directions.\n")
        f.write("5. Focus frequency analysis on directions with high frequency diversity.\n")

    print(f"Selection report saved to: {report_file}")


def create_selection_visualization(
    beamformed_audio,
    directions,
    selected_directions,
    uniqueness_scores,
    correlation_matrix,
    output_path,
    sample_rate,
):
    """Create comprehensive visualization of the selection process."""
    fig = plt.figure(figsize=(20, 16))
    dir_to_idx = {direction: i for i, direction in enumerate(directions)}
    score_lookup = {s["direction"]: s for s in uniqueness_scores}
    colors = ["red" if direction in selected_directions else "lightblue" for direction in directions]

    # Plot 1: Uniqueness scores overview
    ax1 = plt.subplot(3, 3, 1)
    scores = [score_lookup[direction]["total_score"] for direction in directions]
    ax1.bar(directions, scores, color=colors, alpha=0.8)
    ax1.set_title("Uniqueness Scores (Red = Selected)", fontweight="bold")
    ax1.set_ylabel("Uniqueness Score")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.axhline(y=MIN_UNIQUENESS_THRESHOLD, color="black", linestyle="--", alpha=0.5, label="Min Threshold")
    ax1.legend()

    # Plot 2: Correlation matrix heatmap
    ax2 = plt.subplot(3, 3, 2)
    im = ax2.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax2.set_title("Direction Correlation Matrix")
    ax2.set_xticks(range(len(directions)))
    ax2.set_yticks(range(len(directions)))
    ax2.set_xticklabels(directions, rotation=45)
    ax2.set_yticklabels(directions)
    plt.colorbar(im, ax=ax2, shrink=0.8)
    selected_indices = [dir_to_idx[direction] for direction in selected_directions]
    for i in selected_indices:
        for j in selected_indices:
            ax2.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2))

    # Plot 3: Power vs Uniqueness scatter
    ax3 = plt.subplot(3, 3, 3)
    powers = [score_lookup[direction]["rms_power"] for direction in directions]
    ax3.scatter(powers, scores, c=colors, s=100, alpha=0.7)
    for i, direction in enumerate(directions):
        ax3.annotate(direction, (powers[i], scores[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax3.set_xlabel("RMS Power")
    ax3.set_ylabel("Uniqueness Score")
    ax3.set_title("Power vs Uniqueness")
    ax3.grid(True, alpha=0.3)

    # Plot 4–6: Component breakdowns for selected directions (up to 3 for clarity)
    if len(selected_directions) >= 3:
        component_names = [
            "activity_variation",
            "frequency_diversity",
            "temporal_complexity",
            "acoustic_complexity",
            "spatial_uniqueness",
        ]
        component_labels = ["Spectral\nActivity", "Frequency\nDiversity", "Temporal\nComplexity", "Acoustic\nComplexity", "Spatial\nUniqueness"]
        for plot_idx, direction in enumerate(selected_directions[:3]):
            ax = plt.subplot(3, 3, 4 + plot_idx)
            score_data = score_lookup[direction]
            values = [score_data[comp] for comp in component_names]
            bars = ax.bar(component_labels, values, color="red", alpha=0.7)
            ax.set_title(f"{direction.upper()} Components")
            ax.set_ylabel("Component Score")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3, axis="y")
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height + height * 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    # Plot 7: Spectrogram of first selected direction
    ax7 = plt.subplot(3, 3, 7)
    if len(selected_directions) > 0:
        direction = selected_directions[0]
        idx = dir_to_idx[direction]
        audio_signal = beamformed_audio[:, idx]
        f, t, Sxx = signal.spectrogram(audio_signal, sample_rate, nperseg=1024)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        ax7.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="viridis")
        ax7.set_title(f"Spectrogram: {direction.upper()}")
        ax7.set_xlabel("Time (s)")
        ax7.set_ylabel("Frequency (Hz)")
        ax7.set_ylim(0, min(8000, sample_rate // 2))

    # Plot 8: Selection summary
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis("off")
    summary_text = "SELECTION SUMMARY\n\n"
    summary_text += f"Total Directions: {len(directions)}\n"
    summary_text += f"Selected: {len(selected_directions)}\n"
    summary_text += f"Rejected: {len(directions) - len(selected_directions)}\n\n"
    summary_text += "SELECTED FOR ANALYSIS:\n"
    for i, direction in enumerate(selected_directions):
        score = score_lookup[direction]["total_score"]
        summary_text += f"{i+1}. {direction} (score: {score:.3f})\n"
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, verticalalignment="top", fontfamily="monospace", fontsize=10)

    # Plot 9: Frequency-band comparison for up to 3 selections
    ax9 = plt.subplot(3, 3, 9)
    if len(selected_directions) > 0:
        freq_bands = [(0, 500), (500, 1500), (1500, 3500), (3500, 6000), (6000, 12000)]
        band_names = ["0-0.5k", "0.5-1.5k", "1.5-3.5k", "3.5-6k", "6-12k"]
        x = np.arange(len(band_names))
        width = 0.8 / max(1, len(selected_directions[:3]))
        for i, direction in enumerate(selected_directions[:3]):
            idx = dir_to_idx[direction]
            audio_signal = beamformed_audio[:, idx]
            f, t, Sxx = signal.spectrogram(audio_signal, sample_rate, nperseg=1024)
            band_powers = []
            for f_low, f_high in freq_bands:
                freq_mask = (f >= f_low) & (f <= f_high)
                if np.any(freq_mask):
                    band_power = np.mean(Sxx[freq_mask, :])
                    band_powers.append(10 * np.log10(band_power + 1e-10))
                else:
                    band_powers.append(-60)
            ax9.bar(x + i * width, band_powers, width, label=direction, alpha=0.8)
        ax9.set_xlabel("Frequency Bands (Hz)")
        ax9.set_ylabel("Average Power (dB)")
        ax9.set_title("Frequency Content Comparison")
        ax9.set_xticks(x + width)
        ax9.set_xticklabels(band_names)
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    viz_file = output_path / "selection_analysis.png"
    plt.savefig(viz_file, dpi=300, bbox_inches="tight")
    print(f"Selection visualization saved to: {viz_file}")
    plt.close()


def plot_exported_spectrograms(exported_files, output_path, sample_rate):
    """
    Plot spectrograms of all exported audio files side-by-side for comparison.
    """
    n = len(exported_files)
    if n == 0:
        print("No exported files to plot.")
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for i, file in enumerate(exported_files):
        audio, sr = sf.read(file)
        if audio.ndim > 1:
            audio = audio[:, 0]
        f, t, Sxx = signal.spectrogram(audio, sr, nperseg=1024)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        ax = axes[0, i]
        ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="viridis")
        ax.set_title(os.path.basename(file), fontsize=10)
        ax.set_xlabel("Time (s)")
        if i == 0:
            ax.set_ylabel("Frequency (Hz)")
        else:
            ax.set_yticklabels([])
        ax.set_ylim(0, min(8000, sr // 2))
    plt.tight_layout()
    out_png = output_path / "exported_spectrograms.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Exported spectrogram comparison saved to: {out_png}")


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch ecoacoustic beamforming analysis.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing 9-channel, 2nd-order ambisonic WAV files to process. If not set, processes a single file."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Single 9-channel, 2nd-order ambisonic WAV file to process (overrides --input_dir if set)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ecoacoustic_analysis",
        help="Output directory for analysis results."
    )
    args = parser.parse_args()

    # Determine files to process
    if args.input_file:
        files_to_process = [args.input_file]
    elif args.input_dir:
        wav_dir = Path(args.input_dir)
        files_to_process = sorted([str(f) for f in wav_dir.glob("*.wav")])
        if not files_to_process:
            print(f"No WAV files found in {args.input_dir}")
            exit(1)
    else:
        # Default: process geese-call-octo.wav if present
        default_file = "geese-call-octo.wav"
        if Path(default_file).exists():
            files_to_process = [default_file]
        else:
            print("No input file or directory specified, and geese-call-octo.wav not found.")
            exit(1)

    print(f"Batch processing {len(files_to_process)} file(s)...")
    for input_file in files_to_process:
        print(f"\n=== Processing: {input_file} ===")
        try:
            exported_files = analyze_and_export_best_directions(
                input_file=input_file,
                output_dir=args.output_dir,
                duration_seconds=DEFAULT_DURATION_SECONDS,
                start_time=DEFAULT_START_TIME,
                max_exports=MAX_EXPORTS,
                min_uniqueness_threshold=MIN_UNIQUENESS_THRESHOLD,
                correlation_threshold=CORRELATION_THRESHOLD,
                use_correlation_filter=USE_CORRELATION_FILTER,
                use_min_angle_filter=USE_MIN_ANGLE_FILTER,
                min_angular_separation_deg=MIN_ANGULAR_SEPARATION_DEG,
                grid_mode=GRID_MODE,
            )

            if exported_files:
                print(f"\nSUCCESS: Exported {len(exported_files)} optimized files for ecoacoustic analysis.")
                for file in exported_files:
                    print(f"   {file.name}")
                print("\nNext steps:")
                print("   • Inspect maad_indices_all_directions.csv for full-field indices.")
                print("   • Inspect maad_indices.csv for selected/exported beams.")
                print("   • Review selection_report.txt and selection_analysis.png.")
            else:
                print("\nNo directions met the selection criteria.")
                print("Try lowering min_uniqueness_threshold, relaxing correlation, or disabling angular spacing.")

        except Exception as e:
            print(f"Error: {e}")
            print("Ensure you have a 9-channel 2nd-order ambisonic WAV and ambisonic_beamforming.py present.")
