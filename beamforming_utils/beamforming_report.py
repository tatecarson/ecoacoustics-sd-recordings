import os 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal
import numpy as np
import soundfile as sf
from pathlib import Path

from beamforming_utils.config import (
    W_ACTIVITY, W_FREQDIV, W_TEMP, W_ACI, W_SPATIAL, MIN_UNIQUENESS_THRESHOLD, CORRELATION_THRESHOLD, GRID_MODE, MAX_EXPORTS
)

def create_selection_report(
    selected_directions, uniqueness_scores, correlation_matrix, all_directions, output_path, exported_files,
    preproc_mode=None, profile_params=None, report_lines=None
):
    """Create a detailed report explaining the selection decisions, including preprocessing provenance."""
    report_file = output_path / "selection_report.txt"
    dir_to_idx = {direction: i for i, direction in enumerate(all_directions)}
    score_lookup = {s["direction"]: s for s in uniqueness_scores}

    with open(report_file, "w") as f:
        # --- Provenance block ---
        f.write("PREPROCESSING PROVENANCE\n")
        f.write("-" * 50 + "\n")
        f.write(f"PREPROC_MODE: {preproc_mode}\n")
        if profile_params is not None:
            f.write(f"PREPROC_PARAMS: hpf_hz={profile_params.get('hpf_hz')}, envelope_median_ms={profile_params.get('envelope_median_ms')}\n")
        if report_lines:
            for line in report_lines:
                if 'AUTO-PREPROC' in line:
                    f.write(line + "\n")
        f.write("\n")
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


def generate_individual_spectrograms(exported_files, non_selected_files, output_path, sample_rate):
    """
    Generate individual spectrogram images for each exported file (selected and rejected).
    Returns a dictionary mapping file paths to their spectrogram image paths.
    """
    spectrograms = {}
    all_files = exported_files + non_selected_files
    
    if not all_files:
        return spectrograms
    
    print(f"Generating {len(all_files)} individual spectrograms...")
    
    # Create spectrograms subdirectory
    spec_dir = output_path / "spectrograms"
    spec_dir.mkdir(exist_ok=True)
    
    for file_path in all_files:
        try:
            # Load audio
            audio, sr = sf.read(file_path)
            if audio.ndim > 1:
                audio = audio[:, 0]
            
            # Generate spectrogram
            f, t, Sxx = signal.spectrogram(audio, sr, nperseg=1024, noverlap=512)
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            im = ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="viridis")
            
            # Styling
            file_stem = file_path.stem
            ax.set_title(f"Spectrogram: {file_stem}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel("Frequency (Hz)", fontsize=12)
            ax.set_ylim(0, min(8000, sr // 2))
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Power (dB)", fontsize=12)
            
            # Save individual spectrogram
            spec_filename = f"{file_stem}_spectrogram.png"
            spec_path = spec_dir / spec_filename
            plt.savefig(spec_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Store relative path for HTML
            spectrograms[file_path] = f"spectrograms/{spec_filename}"
            
        except Exception as e:
            print(f"Error generating spectrogram for {file_path.name}: {e}")
    
    print(f"Individual spectrograms saved to: {spec_dir}")
    return spectrograms


def create_html_report(
    input_file,
    output_path,
    exported_files,
    selected_directions,
    uniqueness_scores,
    indices_df_all,
    correlation_matrix,
    directions,
    duration_seconds,
    sample_rate,
    non_selected_files=None,
    individual_spectrograms=None,
):
    """
    Create an interactive HTML report for the analysis results.
    """
    from datetime import datetime
    import base64
    
    html_file = output_path / "analysis_report.html"
    input_stem = Path(input_file).stem
    score_lookup = {s["direction"]: s for s in uniqueness_scores}
    non_selected_files = non_selected_files or []
    non_selected_directions = [d for d in directions if d not in selected_directions]
    individual_spectrograms = individual_spectrograms or {}
    
    # Read and encode images as base64
    def encode_image(image_path):
        if image_path.exists():
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        return None
    
    selection_analysis_b64 = encode_image(output_path / "selection_analysis.png")
    spectrograms_b64 = encode_image(output_path / "exported_spectrograms.png")
    
    # Generate HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ecoacoustic Beamforming Analysis - {input_stem}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .analysis-params {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .audio-section {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .audio-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .audio-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .audio-item h4 {{
            margin-top: 0;
            color: #e74c3c;
            border: none;
            padding: 0;
        }}
        audio {{
            width: 100%;
            margin: 10px 0;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .metrics-table th, .metrics-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .metrics-table th {{
            background-color: #34495e;
            color: white;
        }}
        .metrics-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .image-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #3498db;
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card h4 {{
            margin: 0 0 10px 0;
            border: none;
            padding: 0;
            color: white;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #7f8c8d;
            border-top: 1px solid #bdc3c7;
            padding-top: 20px;
        }}
        .param-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .param-item {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .param-item strong {{
            color: #2c3e50;
        }}
        .direction-badge {{
            display: inline-block;
            background: #e74c3c;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
            margin: 2px;
        }}
        .rejected-badge {{
            display: inline-block;
            background: #95a5a6;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
            margin: 2px;
        }}
        .rejected-section {{
            background: #f8f8f8;
            border-left: 4px solid #95a5a6;
            padding: 20px;
            margin: 30px 0;
            border-radius: 8px;
        }}
        .rejected-item {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border: 1px solid #bdc3c7;
        }}
        .rejected-item h4 {{
            margin-top: 0;
            color: #7f8c8d;
            border: none;
            padding: 0;
        }}
        .reason-text {{
            color: #e74c3c;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .spectrogram-container {{
            margin: 15px 0;
            text-align: center;
        }}
        .spectrogram-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        .spectrogram-toggle {{
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
            margin: 10px 0;
        }}
        .spectrogram-toggle:hover {{
            background: #2980b9;
        }}
        .spectrogram-content {{
            display: none;
            margin-top: 10px;
        }}
        .spectrogram-content.show {{
            display: block;
        }}
    </style>
    <script>
        function toggleSpectrogram(id) {{
            const content = document.getElementById(id);
            const button = content.previousElementSibling;
            if (content.classList.contains('show')) {{
                content.classList.remove('show');
                button.textContent = 'Show Spectrogram';
            }} else {{
                content.classList.add('show');
                button.textContent = 'Hide Spectrogram';
            }}
        }}
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 Ecoacoustic Beamforming Analysis Report</h1>
            <h2>{input_stem}</h2>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="analysis-params">
            <h3>📊 Analysis Parameters</h3>
            <div class="param-grid">
                <div class="param-item">
                    <strong>Duration:</strong> {duration_seconds}s
                </div>
                <div class="param-item">
                    <strong>Grid Mode:</strong> {GRID_MODE}
                </div>
                <div class="param-item">
                    <strong>Max Exports:</strong> {MAX_EXPORTS}
                </div>
                <div class="param-item">
                    <strong>Min Uniqueness:</strong> {MIN_UNIQUENESS_THRESHOLD}
                </div>
                <div class="param-item">
                    <strong>Correlation Threshold:</strong> {CORRELATION_THRESHOLD}
                </div>
                <div class="param-item">
                    <strong>Sample Rate:</strong> {sample_rate} Hz
                </div>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h4>Total Directions</h4>
                <div class="value">{len(directions)}</div>
            </div>
            <div class="stat-card">
                <h4>Selected Directions</h4>
                <div class="value">{len(selected_directions)}</div>
            </div>
            <div class="stat-card">
                <h4>Exported Files</h4>
                <div class="value">{len(exported_files)}</div>
            </div>
            <div class="stat-card">
                <h4>Analysis Duration</h4>
                <div class="value">{duration_seconds}s</div>
            </div>
        </div>

        <div class="audio-section">
            <h3>🔊 Selected Direction Audio Files</h3>
            <p>Listen to the exported beamformed audio for each selected direction:</p>
            <div class="audio-grid">"""

    # Add audio players for each exported file
    for i, file_path in enumerate(exported_files):
        direction = selected_directions[i]
        score_data = score_lookup[direction]
        spec_id = f"spec_selected_{i}"
        
        html_content += f"""
                <div class="audio-item">
                    <h4>Direction: {direction.upper()}</h4>
                    <audio controls preload="metadata">
                        <source src="{file_path.name}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    <p><strong>Uniqueness Score:</strong> {score_data['total_score']:.4f}</p>
                    <p><strong>RMS Power:</strong> {score_data['rms_power']:.4f}</p>
                    <div>
                        <small>
                            <strong>Components:</strong><br>
                            Activity: {score_data['activity_variation']:.3f} | 
                            Frequency: {score_data['frequency_diversity']:.3f} | 
                            Temporal: {score_data['temporal_complexity']:.3f} | 
                            Acoustic: {score_data['acoustic_complexity']:.3f} | 
                            Spatial: {score_data['spatial_uniqueness']:.3f}
                        </small>
                    </div>"""
        
        # Add spectrogram if available
        if file_path in individual_spectrograms:
            spec_path = individual_spectrograms[file_path]
            html_content += f"""
                    <div class="spectrogram-container">
                        <button class="spectrogram-toggle" onclick="toggleSpectrogram('{spec_id}')">Show Spectrogram</button>
                        <div id="{spec_id}" class="spectrogram-content">
                            <img src="{spec_path}" alt="Spectrogram for {direction}">
                        </div>
                    </div>"""
        
        html_content += """
                </div>"""

    html_content += """
            </div>
        </div>"""

    # Add rejected directions section if they exist
    if non_selected_files:
        html_content += """
        <div class="rejected-section">
            <h3>❌ Rejected Direction Audio Files</h3>
            <p>These directions were not selected due to low uniqueness scores, high correlation with selected directions, or exceeding the maximum export limit:</p>
            <div class="audio-grid">"""

        # Get rejection reasons for each non-selected direction
        rejection_reasons = {}
        for direction in non_selected_directions:
            score_data = score_lookup[direction]
            if score_data["total_score"] < MIN_UNIQUENESS_THRESHOLD:
                rejection_reasons[direction] = f"Low uniqueness score ({score_data['total_score']:.3f})"
            else:
                # Check correlation with selected directions
                dir_to_idx = {d: i for i, d in enumerate(directions)}
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
                    rejection_reasons[direction] = f"Too correlated with {corr_with} (r={max_corr:.3f})"
                else:
                    rejection_reasons[direction] = "Exceeded maximum exports limit"

        # Add audio players for each rejected file
        for i, file_path in enumerate(non_selected_files):
            # Extract direction from filename (remove 'rejected_' prefix and duration suffix)
            filename = file_path.name
            direction = filename.replace('rejected_', '').replace(f'_{duration_seconds}s.wav', '')
            score_data = score_lookup[direction]
            reason = rejection_reasons.get(direction, "Unknown reason")
            spec_id = f"spec_rejected_{i}"
            
            html_content += f"""
                <div class="rejected-item">
                    <h4>Direction: {direction.upper()} <span class="rejected-badge">REJECTED</span></h4>
                    <audio controls preload="metadata">
                        <source src="{file_path.name}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    <p class="reason-text">Rejection Reason: {reason}</p>
                    <p><strong>Uniqueness Score:</strong> {score_data['total_score']:.4f}</p>
                    <p><strong>RMS Power:</strong> {score_data['rms_power']:.4f}</p>
                    <div>
                        <small>
                            <strong>Components:</strong><br>
                            Activity: {score_data['activity_variation']:.3f} | 
                            Frequency: {score_data['frequency_diversity']:.3f} | 
                            Temporal: {score_data['temporal_complexity']:.3f} | 
                            Acoustic: {score_data['acoustic_complexity']:.3f} | 
                            Spatial: {score_data['spatial_uniqueness']:.3f}
                        </small>
                    </div>"""
            
            # Add spectrogram if available
            if file_path in individual_spectrograms:
                spec_path = individual_spectrograms[file_path]
                html_content += f"""
                    <div class="spectrogram-container">
                        <button class="spectrogram-toggle" onclick="toggleSpectrogram('{spec_id}')">Show Spectrogram</button>
                        <div id="{spec_id}" class="spectrogram-content">
                            <img src="{spec_path}" alt="Spectrogram for {direction}">
                        </div>
                    </div>"""
            
            html_content += """
                </div>"""

        html_content += """
            </div>
        </div>"""

    html_content += """

        <h3>📈 Analysis Results Overview</h3>"""

    # Add selection analysis visualization
    if selection_analysis_b64:
        html_content += f"""
        <div class="image-container">
            <h4>Selection Analysis Visualization</h4>
            <img src="data:image/png;base64,{selection_analysis_b64}" alt="Selection Analysis">
        </div>"""

    # Add exported spectrograms
    if spectrograms_b64:
        html_content += f"""
        <div class="image-container">
            <h4>Exported Direction Spectrograms</h4>
            <img src="data:image/png;base64,{spectrograms_b64}" alt="Exported Spectrograms">
        </div>"""

    # Add detailed metrics table for selected directions
    if selected_directions:
        selected_df = indices_df_all[indices_df_all["direction"].isin(selected_directions)].copy()
        html_content += """
        <h3>📋 Detailed Metrics (Selected Directions)</h3>
        <table class="metrics-table">
            <tr>
                <th>Direction</th>
                <th>ACI</th>
                <th>ADI</th>
                <th>Hf</th>
                <th>Ht</th>
                <th>Total Score</th>
                <th>RMS Power</th>
            </tr>"""
        
        for _, row in selected_df.iterrows():
            html_content += f"""
            <tr>
                <td><span class="direction-badge">{row['direction']}</span></td>
                <td>{row['ACI']:.4f}</td>
                <td>{row['ADI']:.4f}</td>
                <td>{row['Hf']:.4f}</td>
                <td>{row['Ht']:.4f}</td>
                <td>{row['total_score']:.4f}</td>
                <td>{row['rms_power']:.4f}</td>
            </tr>"""
        
        html_content += """
        </table>"""

    # Add weight configuration info
    html_content += f"""
        <h3>⚖️ Uniqueness Score Weights</h3>
        <div class="param-grid">
            <div class="param-item">
                <strong>Activity (Hf):</strong> {W_ACTIVITY:.2f}
            </div>
            <div class="param-item">
                <strong>Frequency Diversity (ADI):</strong> {W_FREQDIV:.2f}
            </div>
            <div class="param-item">
                <strong>Temporal Complexity (1-Ht):</strong> {W_TEMP:.2f}
            </div>
            <div class="param-item">
                <strong>Acoustic Complexity (ACI):</strong> {W_ACI:.2f}
            </div>
            <div class="param-item">
                <strong>Spatial Uniqueness:</strong> {W_SPATIAL:.2f}
            </div>
        </div>

        <h3>📁 Generated Files</h3>
        <ul>
            <li>📊 <code>maad_indices_all_directions.csv</code> - Complete indices for all directions</li>
            <li>📊 <code>maad_indices.csv</code> - Indices for selected/exported directions</li>
            <li>📈 <code>selection_analysis.png</code> - Comprehensive analysis visualization</li>
            <li>📈 <code>exported_spectrograms.png</code> - Spectrogram comparison</li>
            <li>📄 <code>selection_report.txt</code> - Detailed text report</li>
            <li>📊 <code>correlation_matrix.csv</code> - Direction correlation data</li>"""

    for file_path in exported_files:
        html_content += f"""
            <li>🎵 <code>{file_path.name}</code> - Beamformed audio file (SELECTED)</li>"""

    # Add rejected files to the listing if they exist
    for file_path in non_selected_files:
        html_content += f"""
            <li>🎵 <code>{file_path.name}</code> - Beamformed audio file (REJECTED)</li>"""

    html_content += f"""
        </ul>

        <div class="footer">
            <p>Generated by Ecoacoustic Beamforming Analysis Script</p>
            <p>Input file: <code>{input_file}</code></p>
        </div>
        </div>
    </body>
    </html>"""

    # Write HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive HTML report saved to: {html_file}")
    return html_file