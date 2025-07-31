import numpy as np
import pandas as pd
import warnings
from scipy import signal
from maad import sound as maad_sound
from maad import features as maad_features

from beamforming_utils.beamforming_grid import  angular_separation_deg

from beamforming_utils.config import (
    W_ACTIVITY, W_FREQDIV, W_TEMP, W_ACI, W_SPATIAL,
    ADI_AEI_DB_THRESHOLD, NDSI_BIO, NDSI_ANTH,
)

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

def calculate_uniqueness_metrics(beamformed_audio, directions, sample_rate, profile_weights=None, ADI_dB_threshold=None, preproc=None):
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
                ADI_dB_threshold=ADI_dB_threshold if ADI_dB_threshold is not None else ADI_AEI_DB_THRESHOLD,
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

    # Weighted total score (profile-aware)
    w = profile_weights or {"Hf": W_ACTIVITY, "ADI": W_FREQDIV, "TEMP": W_TEMP, "ACI": W_ACI, "SPATIAL": W_SPATIAL}
    total_score = (
        w["Hf"]      * activity_variation +
        w["ADI"]     * frequency_diversity +
        w["TEMP"]    * temporal_complexity +
        w["ACI"]     * acoustic_complexity +
        w["SPATIAL"] * spatial_uniqueness
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
