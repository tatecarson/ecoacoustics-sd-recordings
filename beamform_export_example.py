import numpy as np
import soundfile as sf
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal

# Import the beamformer class (assumes ambisonic_beamforming.py is in same directory)
from ambisonic_beamforming import AmbisonicBeamformer


def analyze_and_export_best_directions(input_file, output_dir="ecoacoustic_analysis", 
                                     duration_seconds=30, start_time=0, 
                                     max_exports=5, min_uniqueness_threshold=0.3,
                                     correlation_threshold=0.7):
    """
    Analyze ambisonic audio and export only the most distinctive directions
    for ecoacoustic analysis, combining correlation and uniqueness metrics.
    
    Args:
        input_file: Path to 9-channel 2nd-order ambisonic WAV file
        output_dir: Directory to save selected files
        duration_seconds: Length of audio to export (default 30 seconds)  
        start_time: Start time in seconds to begin extraction
        max_exports: Maximum number of directions to export (default 5)
        min_uniqueness_threshold: Minimum uniqueness score to consider (default 0.3)
        correlation_threshold: Max correlation between exported directions (default 0.7)
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"🎯 SMART ECOACOUSTIC BEAMFORMING ANALYSIS")
    print(f"=" * 50)
    print(f"Loading ambisonic file: {input_file}")
    
    # Load the ambisonic audio file
    try:
        audio, sample_rate = sf.read(input_file)
        print(f"Loaded: {audio.shape[0]/sample_rate:.1f}s, {audio.shape[1]} channels, {sample_rate} Hz")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return []
    
    # Validate it's 9-channel 2nd order ambisonic
    if audio.shape[1] != 9:
        print(f"❌ Error: Expected 9 channels for 2nd order ambisonics, got {audio.shape[1]}")
        return []
    
    # Create beamformer
    beamformer = AmbisonicBeamformer(sample_rate)
    
    # Calculate sample indices for extraction
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + duration_seconds) * sample_rate)
    
    # Make sure we don't exceed file length
    if end_sample > audio.shape[0]:
        end_sample = audio.shape[0]
        actual_duration = (end_sample - start_sample) / sample_rate
        print(f"Note: Using {actual_duration:.1f}s (less than requested {duration_seconds}s)")
    
    # Extract the segment
    audio_segment = audio[start_sample:end_sample]
    print(f"Analyzing {audio_segment.shape[0]/sample_rate:.1f} seconds starting at {start_time}s\n")
    
    # Beamform to all 3D directions
    print("🔄 Beamforming to all 3D directions...")
    beamformed_audio, directions = beamformer.beamform_3d_directions(audio_segment)
    print(f"Beamformed to {len(directions)} directions: {directions}\n")
    
    # Step 1: Calculate correlation matrix
    print("📊 Step 1: Calculating correlation matrix...")
    correlation_matrix = calculate_correlation_analysis(beamformed_audio, directions, output_path)
    
    # Step 2: Calculate uniqueness metrics
    print("🎯 Step 2: Analyzing uniqueness metrics...")
    uniqueness_scores = calculate_uniqueness_metrics(beamformed_audio, directions, sample_rate)
    
    # Step 3: Smart selection algorithm
    print("🧠 Step 3: Selecting best directions for ecoacoustic analysis...")
    selected_directions = smart_direction_selection(
        beamformed_audio, directions, correlation_matrix, uniqueness_scores,
        max_exports, min_uniqueness_threshold, correlation_threshold
    )
    
    # Step 4: Export only selected directions
    print(f"💾 Step 4: Exporting {len(selected_directions)} selected directions...")
    exported_files = export_selected_directions(
        beamformed_audio, directions, selected_directions, 
        output_path, duration_seconds, sample_rate
    )

    # Plot spectrograms of exported files for comparison
    plot_exported_spectrograms(exported_files, output_path, sample_rate)
    
    # Step 5: Create analysis report
    create_selection_report(
        selected_directions, uniqueness_scores, correlation_matrix, 
        directions, output_path, exported_files
    )
    
    # Step 6: Create visualization
    create_selection_visualization(
        beamformed_audio, directions, selected_directions, 
        uniqueness_scores, correlation_matrix, output_path, sample_rate
    )
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {output_path.absolute()}")
    print(f"🎵 {len(exported_files)} optimized files ready for ecoacoustic analysis")
    
    return exported_files


def calculate_correlation_analysis(beamformed_audio, directions, output_path):
    """Calculate correlation matrix and identify redundant directions."""
    correlation_matrix = np.corrcoef(beamformed_audio, rowvar=False)
    
    # Save correlation matrix
    correlation_file = output_path / "correlation_matrix.csv"
    np.savetxt(correlation_file, correlation_matrix, delimiter=",", 
               header=",".join(directions), comments="")
    
    return correlation_matrix


def calculate_uniqueness_metrics(beamformed_audio, directions, sample_rate):
    """Calculate comprehensive uniqueness metrics for each direction."""
    
    # Define ecoacoustic frequency bands (in Hz)
    eco_freq_bands = [
        (0, 500),
        (500, 1500),
        (1500, 3500),
        (3500, 6000),
        (6000, 12000)
    ]

    uniqueness_scores = []
    spectrograms = []
    print(f"Calculating spectrograms for {beamformed_audio.shape[1]} directions...")
    
    for i in range(beamformed_audio.shape[1]):
        try:
            f, t, Sxx = signal.spectrogram(beamformed_audio[:, i], sample_rate, 
                                         nperseg=2048, noverlap=1024)
            spectrograms.append((f, t, Sxx))
            print(f"  Direction {directions[i]}: Spectrogram calculated ({len(f)} frequency bins, {len(t)} time slices)")
        except Exception as e:
            print(f"❌ Error calculating spectrogram for direction {directions[i]}: {e}")
            continue
    
    print("Analyzing uniqueness metrics for each direction...")
    for i, direction in enumerate(directions):
        try:
            f, t, Sxx = spectrograms[i]
            
            # Metric 1: Ecoacoustic Activity Index (spectral variation)
            spectral_centroids = []
            spectral_rolloffs = []
            for time_slice in range(Sxx.shape[1]):
                power_spectrum = Sxx[:, time_slice]
                if np.sum(power_spectrum) > 0:
                    # Spectral centroid
                    centroid = np.sum(f * power_spectrum) / np.sum(power_spectrum)
                    spectral_centroids.append(centroid)
                    
                    # Spectral rolloff (85% of energy)
                    cumsum = np.cumsum(power_spectrum)
                    rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
                    if len(rolloff_idx) > 0:
                        spectral_rolloffs.append(f[rolloff_idx[0]])
            
            activity_variation = (np.std(spectral_centroids) + np.std(spectral_rolloffs)) / 2 if spectral_centroids else 0
            print(f"  Direction {direction}: Activity variation calculated ({activity_variation:.4f})")
            
            # Metric 2: Ecological frequency band diversity
            band_powers = []
            for f_low, f_high in eco_freq_bands:
                freq_mask = (f >= f_low) & (f <= f_high)
                if np.any(freq_mask):
                    band_power = np.mean(Sxx[freq_mask, :])
                    band_powers.append(band_power)
            
            # Shannon diversity index for frequency bands
            if band_powers and sum(band_powers) > 0:
                band_probs = np.array(band_powers) / sum(band_powers)
                band_probs = band_probs[band_probs > 0]  # Remove zeros
                frequency_diversity = -np.sum(band_probs * np.log(band_probs))
            else:
                frequency_diversity = 0
            print(f"  Direction {direction}: Frequency diversity calculated ({frequency_diversity:.4f})")
            
            # Metric 3: Temporal complexity (using FFT for efficiency)
            audio_signal = beamformed_audio[:, i]

            try:
                # Use FFT for efficient autocorrelation
                max_samples = min(len(audio_signal), int(sample_rate * 5))  # 5 seconds
                audio_segment = audio_signal[:max_samples]
                
                # Remove DC component
                audio_segment = audio_segment - np.mean(audio_segment)
                
                # Compute autocorrelation using FFT (much faster)
                fft = np.fft.fft(audio_segment, n=2*len(audio_segment))
                power = np.abs(fft) ** 2
                autocorr = np.fft.ifft(power).real
                autocorr = autocorr[:len(audio_segment)]
                
                # Normalize
                if autocorr[0] > 0:
                    autocorr_norm = autocorr / autocorr[0]
                else:
                    autocorr_norm = autocorr
                
                # Detect periodicity
                peaks, properties = signal.find_peaks(
                    autocorr_norm[int(sample_rate*0.05):int(sample_rate*0.5)],  # Look between 50ms and 500ms
                    height=0.1,
                    prominence=0.05
                )
                
                if len(peaks) > 0:
                    peak_prominence = np.max(properties['peak_heights'])
                else:
                    peak_prominence = 0
                    
                print(f"  Direction {direction}: Temporal complexity calculated ({peak_prominence:.4f})")
                
            except Exception as e:
                print(f"  Direction {direction}: Error in temporal analysis: {e}")
                peak_prominence = 0
            
            print(f"  Processing acoustic complexity for direction {direction}...")
            # Metric 4: Acoustic complexity index (ACI-like)
            # Temporal variation in spectral intensity
            aci_score = 0
            for freq_bin in range(len(f)):
                if f[freq_bin] <= 12000:  # Focus on biologically relevant frequencies
                    intensity_series = Sxx[freq_bin, :]
                    if len(intensity_series) > 1:
                        aci_score += np.sum(np.abs(np.diff(intensity_series))) / np.sum(intensity_series) if np.sum(intensity_series) > 0 else 0
            print(f"  Direction {direction}: Acoustic complexity index calculated ({aci_score:.4f})")
            
            print(f"  Processing spatial uniqueness for direction {direction}...")
            # Metric 5: Spatial uniqueness (how different from other directions)
            spatial_uniqueness = 0
            for j in range(beamformed_audio.shape[1]):
                if i != j:
                    corr = np.corrcoef(beamformed_audio[:, i], beamformed_audio[:, j])[0, 1]
                    spatial_uniqueness += (1 - abs(corr))
            spatial_uniqueness /= (beamformed_audio.shape[1] - 1)
            print(f"  Direction {direction}: Spatial uniqueness calculated ({spatial_uniqueness:.4f})")
            
            print(f"  Combining metrics for direction {direction}...")
            # Combined ecoacoustic uniqueness score
            total_score = (
                activity_variation * 0.25 +      # Spectral activity
                frequency_diversity * 0.25 +     # Frequency diversity  
                peak_prominence * 0.15 +         # Temporal patterns
                aci_score * 0.20 +              # Acoustic complexity
                spatial_uniqueness * 0.15       # Spatial distinctiveness
            )
            print(f"  Direction {direction}: Total uniqueness score ({total_score:.4f})")
            
            uniqueness_scores.append({
                'direction': direction,
                'total_score': total_score,
                'activity_variation': activity_variation,
                'frequency_diversity': frequency_diversity,
                'temporal_complexity': peak_prominence,
                'acoustic_complexity': aci_score,
                'spatial_uniqueness': spatial_uniqueness,
                'rms_power': np.sqrt(np.mean(beamformed_audio[:, i]**2))
            })
        except Exception as e:
            print(f"❌ Error analyzing metrics for direction {direction}: {e}")
            continue
    
    # Sort by total score
    uniqueness_scores.sort(key=lambda x: x['total_score'], reverse=True)
    print("Uniqueness metrics analysis complete.")
    
    return uniqueness_scores


def smart_direction_selection(beamformed_audio, directions, correlation_matrix, 
                            uniqueness_scores, max_exports, min_uniqueness_threshold, 
                            correlation_threshold):
    """
    Smart algorithm to select the most valuable directions for ecoacoustic analysis.
    Combines uniqueness and correlation to avoid redundancy while maximizing information.
    """
    
    selected_directions = []
    selected_indices = []
    
    print(f"Selection criteria:")
    print(f"  • Max exports: {max_exports}")
    print(f"  • Min uniqueness: {min_uniqueness_threshold:.2f}")
    print(f"  • Max correlation between selections: {correlation_threshold:.2f}")
    print()
    
    # Create direction to index mapping
    dir_to_idx = {direction: i for i, direction in enumerate(directions)}
    
    # Start with the most unique direction
    for score_data in uniqueness_scores:
        direction = score_data['direction']
        uniqueness = score_data['total_score']
        
        # Check if meets minimum uniqueness threshold
        if uniqueness < min_uniqueness_threshold:
            print(f"❌ {direction}: Below uniqueness threshold ({uniqueness:.3f} < {min_uniqueness_threshold})")
            continue
        
        # Check if we've reached max exports
        if len(selected_directions) >= max_exports:
            print(f"⏹️ Reached maximum exports ({max_exports})")
            break
        
        # Check correlation with already selected directions
        current_idx = dir_to_idx[direction]
        too_correlated = False
        
        for selected_dir in selected_directions:
            selected_idx = dir_to_idx[selected_dir]
            correlation = abs(correlation_matrix[current_idx, selected_idx])
            
            if correlation > correlation_threshold:
                print(f"❌ {direction}: Too correlated with {selected_dir} (r={correlation:.3f})")
                too_correlated = True
                break
        
        if not too_correlated:
            selected_directions.append(direction)
            selected_indices.append(current_idx)
            print(f"✅ {direction}: Selected (uniqueness={uniqueness:.3f}, power={score_data['rms_power']:.4f})")
    
    print(f"\nSelected {len(selected_directions)} directions for export: {selected_directions}")
    
    return selected_directions


def export_selected_directions(beamformed_audio, directions, selected_directions, 
                             output_path, duration_seconds, sample_rate):
    """Export only the selected directions as WAV files."""
    
    exported_files = []
    dir_to_idx = {direction: i for i, direction in enumerate(directions)}
    
    for direction in selected_directions:
        idx = dir_to_idx[direction]
        
        # Create filename
        filename = f"ecoacoustic_{direction}_{duration_seconds}s.wav"
        filepath = output_path / filename
        
        # Get the beamformed audio for this direction
        direction_audio = beamformed_audio[:, idx]
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(direction_audio))
        if max_val > 0:
            direction_audio = direction_audio / max_val * 0.95
        
        # Export as mono WAV
        sf.write(filepath, direction_audio, sample_rate)
        exported_files.append(filepath)
        
        # Print export stats
        rms_level = np.sqrt(np.mean(direction_audio**2))
        print(f"  💾 {direction}: {filename} (RMS: {rms_level:.4f})")
    
    return exported_files


def create_selection_report(selected_directions, uniqueness_scores, correlation_matrix, 
                          all_directions, output_path, exported_files):
    """Create a detailed report explaining the selection decisions."""
    
    report_file = output_path / "selection_report.txt"
    
    # Create direction mappings
    dir_to_idx = {direction: i for i, direction in enumerate(all_directions)}
    score_lookup = {s['direction']: s for s in uniqueness_scores}
    
    with open(report_file, 'w') as f:
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
            f.write(f"   Component Scores:\n")
            f.write(f"     • Spectral Activity: {score_data['activity_variation']:.4f}\n")
            f.write(f"     • Frequency Diversity: {score_data['frequency_diversity']:.4f}\n")
            f.write(f"     • Temporal Complexity: {score_data['temporal_complexity']:.4f}\n")
            f.write(f"     • Acoustic Complexity: {score_data['acoustic_complexity']:.4f}\n")
            f.write(f"     • Spatial Uniqueness: {score_data['spatial_uniqueness']:.4f}\n")
        
        f.write(f"\n\nCORRELATION ANALYSIS:\n")
        f.write("-" * 25 + "\n")
        f.write("Correlation matrix between selected directions:\n")
        
        # Create correlation submatrix for selected directions
        selected_indices = [dir_to_idx[direction] for direction in selected_directions]
        selected_corr = correlation_matrix[np.ix_(selected_indices, selected_indices)]
        
        # Write correlation matrix
        f.write(f"{'':>12}")
        for direction in selected_directions:
            f.write(f"{direction:>8}")
        f.write("\n")
        
        for i, direction in enumerate(selected_directions):
            f.write(f"{direction:>12}")
            for j in range(len(selected_directions)):
                f.write(f"{selected_corr[i,j]:>8.3f}")
            f.write("\n")
        
        f.write(f"\n\nREJECTED DIRECTIONS:\n")
        f.write("-" * 20 + "\n")
        
        rejected_directions = [d for d in all_directions if d not in selected_directions]
        for direction in rejected_directions:
            score_data = score_lookup[direction]
            
            # Determine rejection reason
            if score_data['total_score'] < 0.3:  # Using default threshold
                reason = f"Low uniqueness ({score_data['total_score']:.3f})"
            else:
                # Check correlation with selected
                dir_idx = dir_to_idx[direction]
                max_corr = 0
                corr_with = ""
                for sel_dir in selected_directions:
                    sel_idx = dir_to_idx[sel_dir]
                    corr = abs(correlation_matrix[dir_idx, sel_idx])
                    if corr > max_corr:
                        max_corr = corr
                        corr_with = sel_dir
                
                if max_corr > 0.7:  # Using default threshold
                    reason = f"Too correlated with {corr_with} (r={max_corr:.3f})"
                else:
                    reason = "Exceeded max exports"
            
            f.write(f"   {direction}: {reason}\n")
        
        f.write(f"\n\nRECOMMENDATIONS FOR ECOACOUSTIC ANALYSIS:\n")
        f.write("-" * 45 + "\n")
        f.write("1. Start analysis with the highest-ranked direction\n")
        f.write("2. Use these files for acoustic indices (ACI, ADI, AEI, BI, NDSI)\n")
        f.write("3. Compare results between directions to understand spatial patterns\n")
        f.write("4. Look for temporal patterns in high complexity directions\n")
        f.write("5. Focus frequency analysis on directions with high frequency diversity\n")
    
    print(f"📄 Selection report saved to: {report_file}")


def create_selection_visualization(beamformed_audio, directions, selected_directions, 
                                 uniqueness_scores, correlation_matrix, output_path, sample_rate):
    """Create comprehensive visualization of the selection process."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create direction mappings
    dir_to_idx = {direction: i for i, direction in enumerate(directions)}
    score_lookup = {s['direction']: s for s in uniqueness_scores}
    
    # Colors: selected = red, rejected = blue
    colors = ['red' if direction in selected_directions else 'lightblue' 
              for direction in directions]
    
    # Plot 1: Uniqueness scores overview
    ax1 = plt.subplot(3, 3, 1)
    scores = [score_lookup[direction]['total_score'] for direction in directions]
    bars = ax1.bar(directions, scores, color=colors, alpha=0.8)
    ax1.set_title('Uniqueness Scores (Red = Selected)', fontweight='bold')
    ax1.set_ylabel('Uniqueness Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add threshold line
    ax1.axhline(y=0.3, color='black', linestyle='--', alpha=0.5, label='Min Threshold')
    ax1.legend()
    
    # Plot 2: Correlation matrix heatmap
    ax2 = plt.subplot(3, 3, 2)
    im = ax2.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_title('Direction Correlation Matrix')
    ax2.set_xticks(range(len(directions)))
    ax2.set_yticks(range(len(directions)))
    ax2.set_xticklabels(directions, rotation=45)
    ax2.set_yticklabels(directions)
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    # Highlight selected directions
    selected_indices = [dir_to_idx[direction] for direction in selected_directions]
    for i in selected_indices:
        for j in selected_indices:
            ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                      fill=False, edgecolor='red', linewidth=2))
    
    # Plot 3: Power vs Uniqueness scatter
    ax3 = plt.subplot(3, 3, 3)
    powers = [score_lookup[direction]['rms_power'] for direction in directions]
    scatter = ax3.scatter(powers, scores, c=colors, s=100, alpha=0.7)
    for i, direction in enumerate(directions):
        ax3.annotate(direction, (powers[i], scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax3.set_xlabel('RMS Power')
    ax3.set_ylabel('Uniqueness Score')
    ax3.set_title('Power vs Uniqueness')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4-6: Component breakdowns for selected directions
    if len(selected_directions) >= 3:
        component_names = ['activity_variation', 'frequency_diversity', 'temporal_complexity', 
                          'acoustic_complexity', 'spatial_uniqueness']
        component_labels = ['Spectral\nActivity', 'Frequency\nDiversity', 'Temporal\nComplexity',
                           'Acoustic\nComplexity', 'Spatial\nUniqueness']
        
        for plot_idx, direction in enumerate(selected_directions[:3]):
            ax = plt.subplot(3, 3, 4 + plot_idx)
            score_data = score_lookup[direction]
            
            values = [score_data[comp] for comp in component_names]
            bars = ax.bar(component_labels, values, color='red', alpha=0.7)
            ax.set_title(f'{direction.upper()} Components')
            ax.set_ylabel('Component Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 7: Spectrograms of selected directions
    ax7 = plt.subplot(3, 3, 7)
    if len(selected_directions) > 0:
        direction = selected_directions[0]
        idx = dir_to_idx[direction]
        audio_signal = beamformed_audio[:, idx]
        
        f, t, Sxx = signal.spectrogram(audio_signal, sample_rate, nperseg=1024)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        ax7.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        ax7.set_title(f'Spectrogram: {direction.upper()}')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Frequency (Hz)')
        ax7.set_ylim(0, min(8000, sample_rate//2))
    
    # Plot 8: Selection summary
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    summary_text = f"SELECTION SUMMARY\n\n"
    summary_text += f"Total Directions: {len(directions)}\n"
    summary_text += f"Selected: {len(selected_directions)}\n"
    summary_text += f"Rejected: {len(directions) - len(selected_directions)}\n\n"
    summary_text += "SELECTED FOR ANALYSIS:\n"
    
    for i, direction in enumerate(selected_directions):
        score = score_lookup[direction]['total_score']
        summary_text += f"{i+1}. {direction} (score: {score:.3f})\n"
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
            verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    # Plot 9: Frequency analysis
    ax9 = plt.subplot(3, 3, 9)
    if len(selected_directions) > 0:
        freq_bands = [(0, 500), (500, 1500), (1500, 3500), (3500, 6000), (6000, 12000)]
        band_names = ['0-0.5k', '0.5-1.5k', '1.5-3.5k', '3.5-6k', '6-12k']
        
        x = np.arange(len(band_names))
        width = 0.8 / len(selected_directions)
        
        for i, direction in enumerate(selected_directions[:3]):  # Max 3 for visibility
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
        
        ax9.set_xlabel('Frequency Bands (Hz)')
        ax9.set_ylabel('Average Power (dB)')
        ax9.set_title('Frequency Content Comparison')
        ax9.set_xticks(x + width)
        ax9.set_xticklabels(band_names)
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    viz_file = output_path / "selection_analysis.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"📊 Selection visualization saved to: {viz_file}")
    plt.close()


def plot_exported_spectrograms(exported_files, output_path, sample_rate):
    """
    Plot spectrograms of all exported audio files side-by-side for comparison.
    """
    n = len(exported_files)
    if n == 0:
        print("No exported files to plot.")
        return
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
    for i, file in enumerate(exported_files):
        audio, sr = sf.read(file)
        if audio.ndim > 1:
            audio = audio[:, 0]
        f, t, Sxx = signal.spectrogram(audio, sr, nperseg=1024)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        ax = axes[0, i]
        im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        ax.set_title(os.path.basename(file), fontsize=10)
        ax.set_xlabel('Time (s)')
        if i == 0:
            ax.set_ylabel('Frequency (Hz)')
        else:
            ax.set_yticklabels([])
        ax.set_ylim(0, min(8000, sr // 2))
    plt.tight_layout()
    out_png = output_path / "exported_spectrograms.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"📊 Exported spectrogram comparison saved to: {out_png}")


# Example usage
if __name__ == "__main__":
    input_file = "ambi-test_1.wav"  
    
    # Smart beamforming analysis and export
    try:
        exported_files = analyze_and_export_best_directions(
            input_file=input_file,
            output_dir="ecoacoustic_analysis",
            duration_seconds=30,
            start_time=5,
            max_exports=5,                    # Export max 5 most distinct directions
            min_uniqueness_threshold=0.3,     # Minimum uniqueness score
            correlation_threshold=0.7          # Max correlation between exported files
        )
        
        if exported_files:
            print(f"\n🎉 SUCCESS! Exported {len(exported_files)} optimized files for ecoacoustic analysis:")
            for file in exported_files:
                print(f"   🎵 {file.name}")
            
            print(f"\n📋 Next steps for ecoacoustic analysis:")
            print(f"   1. Run acoustic indices (ACI, ADI, AEI) on each exported file")
            print(f"   2. Compare biodiversity metrics between directions")
            print(f"   3. Check selection_report.txt for detailed analysis")
            print(f"   4. Review selection_analysis.png for visual insights")
        else:
            print(f"\n⚠️ No directions met the selection criteria.")
            print(f"Try lowering min_uniqueness_threshold or correlation_threshold.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have a 9-channel 2nd-order ambisonic WAV file.")