import numpy as np
import soundfile as sf

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


def export_non_selected_directions(
    beamformed_audio, directions, non_selected_directions, output_path, duration_seconds, sample_rate
):
    """Export the non-selected directions as mono WAV files with 'rejected_' prefix."""
    exported_files = []
    dir_to_idx = {direction: i for i, direction in enumerate(directions)}
    for direction in non_selected_directions:
        idx = dir_to_idx[direction]
        filename = f"rejected_{direction}_{duration_seconds}s.wav"
        filepath = output_path / filename
        direction_audio = beamformed_audio[:, idx]
        max_val = np.max(np.abs(direction_audio))
        if max_val > 0:
            direction_audio = direction_audio / max_val * 0.95
        sf.write(filepath, direction_audio, sample_rate)
        exported_files.append(filepath)
        rms_level = np.sqrt(np.mean(direction_audio**2))
        print(f"  {direction}: {filename} (RMS: {rms_level:.4f}) [REJECTED]")
    return exported_files
