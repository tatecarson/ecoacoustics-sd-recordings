import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

class AdaptiveAmbisonicBeamformer:
    def __init__(self, sample_rate=48000):
        self.fs = sample_rate
        
    def load_bformat(self, filepath, order=2):
        """Load B-format ambisonic file"""
        sr, audio = wavfile.read(filepath)
        self.fs = sr
        self.order = order
        
        if audio.ndim == 1:
            raise ValueError("Expected multi-channel B-format file")
        
        # Expected channels for different orders
        expected_channels = {1: 4, 2: 9, 3: 16}
        if order not in expected_channels:
            raise ValueError(f"Unsupported ambisonic order: {order}")
        
        if audio.shape[1] != expected_channels[order]:
            raise ValueError(f"Expected {expected_channels[order]} channels for {order}nd order, got {audio.shape[1]}")
            
        # Split into B-format components
        if order == 1:
            # 1st order: W, X, Y, Z
            self.W = audio[:, 0]    # Omnidirectional
            self.X = audio[:, 1]    # Front-back
            self.Y = audio[:, 2]    # Left-right  
            self.Z = audio[:, 3]    # Up-down
            self.channels = {'W': 0, 'X': 1, 'Y': 2, 'Z': 3}
            
        elif order == 2:
            # 2nd order: W, X, Y, Z, V, T, R, S, U
            # ACN (Ambisonic Channel Number) ordering
            self.W = audio[:, 0]    # (0,0)  - Omnidirectional
            self.X = audio[:, 1]    # (1,-1) - Y direction
            self.Y = audio[:, 2]    # (1,0)  - Z direction  
            self.Z = audio[:, 3]    # (1,1)  - X direction
            self.V = audio[:, 4]    # (2,-2) 
            self.T = audio[:, 5]    # (2,-1)
            self.R = audio[:, 6]    # (2,0)
            self.S = audio[:, 7]    # (2,1)
            self.U = audio[:, 8]    # (2,2)
            
            self.channels = {'W': 0, 'X': 1, 'Y': 2, 'Z': 3, 
                           'V': 4, 'T': 5, 'R': 6, 'S': 7, 'U': 8}
        
        print(f"Loaded {order}nd order B-format audio: {len(self.W)/sr:.1f}s, {sr}Hz, {audio.shape[1]} channels")
        return audio
    
    def spherical_harmonics(self, azimuth, elevation, order=None):
        """Compute spherical harmonics for given direction"""
        if order is None:
            order = self.order
            
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Convert to standard spherical coordinates
        # Azimuth: 0° = front, 90° = left, 180° = back, 270° = right
        # Elevation: 0° = horizontal, +90° = up, -90° = down
        
        x = np.cos(el_rad) * np.cos(az_rad)  # Front-back
        y = np.cos(el_rad) * np.sin(az_rad)  # Left-right
        z = np.sin(el_rad)                   # Up-down
        
        if order == 1:
            # 1st order spherical harmonics
            Y = np.array([
                1.0,        # Y(0,0)  = W
                y,          # Y(1,-1) = Y  
                z,          # Y(1,0)  = Z
                x           # Y(1,1)  = X
            ])
            
        elif order == 2:
            # 2nd order spherical harmonics (ACN ordering, SN3D normalization)
            sqrt3 = np.sqrt(3.0)
            sqrt15 = np.sqrt(15.0)
            sqrt5 = np.sqrt(5.0)
            
            Y = np.array([
                1.0,                           # Y(0,0)   = W
                sqrt3 * y,                     # Y(1,-1)  = Y
                sqrt3 * z,                     # Y(1,0)   = Z  
                sqrt3 * x,                     # Y(1,1)   = X
                sqrt15 * x * y,                # Y(2,-2)  = V
                sqrt15 * y * z,                # Y(2,-1)  = T
                sqrt5 * (3*z**2 - 1) / 2,      # Y(2,0)   = R
                sqrt15 * x * z,                # Y(2,1)   = S
                sqrt15 * (x**2 - y**2) / 2     # Y(2,2)   = U
            ])
            
        else:
            raise ValueError(f"Order {order} not implemented")
        
        return Y
    
    def beamform_direction(self, azimuth, elevation, pattern='cardioid'):
        """Create beamformed signal for specific direction"""
        
        # Get spherical harmonics for this direction
        Y = self.spherical_harmonics(azimuth, elevation, self.order)
        
        if self.order == 1:
            # 1st order decoding
            if pattern == 'cardioid':
                # Standard cardioid decoder
                beamformed = (self.W + 
                            Y[1] * self.X +  # Y component
                            Y[2] * self.Y +  # Z component  
                            Y[3] * self.Z)   # X component
            else:
                raise ValueError(f"Pattern '{pattern}' not supported for 1st order")
                
        elif self.order == 2:
            # 2nd order decoding - much better directivity!
            if pattern == 'cardioid':
                # 2nd order cardioid has much sharper pattern
                beamformed = (self.W +
                            Y[1] * self.X +  # Y
                            Y[2] * self.Y +  # Z
                            Y[3] * self.Z +  # X
                            Y[4] * self.V +  # XY
                            Y[5] * self.T +  # YZ
                            Y[6] * self.R +  # 3Z²-1
                            Y[7] * self.S +  # XZ
                            Y[8] * self.U)   # X²-Y²
                            
            elif pattern == 'hypercardioid':
                # Even sharper pattern for 2nd order
                # Apply weighting to emphasize higher order components
                w0, w1, w2 = 0.5, 0.5, 1.0  # Weights for orders 0, 1, 2
                beamformed = (w0 * self.W +
                            w1 * (Y[1] * self.X + Y[2] * self.Y + Y[3] * self.Z) +
                            w2 * (Y[4] * self.V + Y[5] * self.T + Y[6] * self.R + 
                                 Y[7] * self.S + Y[8] * self.U))
            else:
                raise ValueError(f"Pattern '{pattern}' not supported")
                
        return beamformed
    
    def compute_srp(self, frame_length=4096, hop_length=2048, 
                    azimuth_range=None, elevation_range=None, 
                    angular_resolution=10, pattern='cardioid'):
        """Compute Steered Response Power across angular grid"""
        
        if azimuth_range is None:
            azimuth_range = (0, 360)
        if elevation_range is None:
            elevation_range = (-90, 90)
            
        # For 2nd order, we can use finer resolution since we have better spatial precision
        if self.order == 2 and angular_resolution > 10:
            print(f"Note: 2nd order ambisonics supports finer resolution. Consider using 5-10° instead of {angular_resolution}°")
            
        # Create angular grid
        azimuths = np.arange(azimuth_range[0], azimuth_range[1], angular_resolution)
        elevations = np.arange(elevation_range[0], elevation_range[1], angular_resolution)
        
        print(f"Testing {len(azimuths)} x {len(elevations)} = {len(azimuths)*len(elevations)} directions")
        print(f"Using {self.order}nd order {pattern} beamforming")
        
        # Storage for SRP values
        srp_map = np.zeros((len(elevations), len(azimuths)))
        direction_powers = []
        direction_coords = []
        
        for i, elevation in enumerate(elevations):
            for j, azimuth in enumerate(azimuths):
                # Beamform in this direction
                beamformed = self.beamform_direction(azimuth, elevation, pattern)
                
                # Compute power using STFT
                f, t, stft = signal.stft(beamformed, self.fs, 
                                       nperseg=frame_length, 
                                       noverlap=frame_length-hop_length)
                
                # Average power across time and frequency
                power = np.mean(np.abs(stft)**2)
                srp_map[i, j] = power
                
                direction_powers.append(power)
                direction_coords.append((azimuth, elevation))
        
        self.srp_map = srp_map
        self.azimuths = azimuths
        self.elevations = elevations
        self.direction_powers = np.array(direction_powers)
        self.direction_coords = direction_coords
        
        return srp_map, direction_powers, direction_coords
    
    def find_peak_directions(self, num_peaks=8, min_separation=30, 
                           max_correlation=0.6, frame_length=4096):
        """Find peak power directions with correlation filtering"""
        
        if not hasattr(self, 'srp_map'):
            raise ValueError("Run compute_srp() first")
        
        # Find local maxima in SRP map
        from scipy.ndimage import maximum_filter
        
        # Apply maximum filter to find local peaks
        local_maxima = maximum_filter(self.srp_map, size=3) == self.srp_map
        
        # Get coordinates of local maxima
        peak_coords = np.where(local_maxima)
        peak_powers = self.srp_map[peak_coords]
        
        # Convert to azimuth/elevation
        all_peak_directions = []
        for i in range(len(peak_coords[0])):
            el_idx = peak_coords[0][i]
            az_idx = peak_coords[1][i]
            azimuth = self.azimuths[az_idx]
            elevation = self.elevations[el_idx]
            power = peak_powers[i]
            all_peak_directions.append((azimuth, elevation, power))
        
        # Sort by power (highest first)
        all_peak_directions.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Found {len(all_peak_directions)} potential peaks, selecting decorrelated subset...")
        
        # Apply greedy selection with both spatial and correlation constraints
        selected_peaks = []
        candidate_beams = []  # Store beamformed signals for correlation testing
        
        for candidate_az, candidate_el, candidate_power in all_peak_directions:
            if len(selected_peaks) >= num_peaks:
                break
            
            # Check spatial separation from already selected peaks
            too_close_spatially = False
            for sel_az, sel_el, sel_power in selected_peaks:
                # Angular separation calculation
                separation = np.sqrt((candidate_az - sel_az)**2 + (candidate_el - sel_el)**2)
                if separation < min_separation:
                    too_close_spatially = True
                    break
            
            if too_close_spatially:
                continue
            
            # Create candidate beam for correlation testing
            candidate_beam = self.beamform_direction(candidate_az, candidate_el)
            
            # Test correlation with all already selected beams
            too_correlated = False
            if len(candidate_beams) > 0:
                # Compute spectrograms for correlation analysis
                f, t, candidate_stft = signal.stft(candidate_beam, self.fs, 
                                                 nperseg=frame_length,
                                                 noverlap=frame_length//2)
                candidate_psd = np.mean(np.abs(candidate_stft)**2, axis=1)
                
                for existing_beam in candidate_beams:
                    f, t, existing_stft = signal.stft(existing_beam, self.fs,
                                                    nperseg=frame_length,
                                                    noverlap=frame_length//2)
                    existing_psd = np.mean(np.abs(existing_stft)**2, axis=1)
                    
                    # Calculate correlation between power spectral densities
                    correlation = np.corrcoef(candidate_psd, existing_psd)[0, 1]
                    
                    if abs(correlation) > max_correlation:
                        too_correlated = True
                        print(f"  Rejected az={candidate_az:3.0f}°, el={candidate_el:+3.0f}° (correlation={correlation:.3f} with existing beam)")
                        break
            
            if not too_correlated:
                selected_peaks.append((candidate_az, candidate_el, candidate_power))
                candidate_beams.append(candidate_beam)
                print(f"  Selected az={candidate_az:3.0f}°, el={candidate_el:+3.0f}°, power={candidate_power:.2e}")
        
        self.peak_directions = selected_peaks
        print(f"\nSelected {len(selected_peaks)} decorrelated peak directions (max_correlation < {max_correlation})")
        
        return selected_peaks
    
    def create_adaptive_beams(self, directions=None, pattern='cardioid'):
        """Create beamformed signals for specified directions"""
        
        if directions is None:
            if not hasattr(self, 'peak_directions'):
                raise ValueError("Run find_peak_directions() first or provide directions")
            directions = [(az, el) for az, el, pow in self.peak_directions]
        
        beamformed_signals = []
        direction_labels = []
        
        for i, (azimuth, elevation) in enumerate(directions):
            beamformed = self.beamform_direction(azimuth, elevation, pattern)
            beamformed_signals.append(beamformed)
            direction_labels.append(f"az{azimuth:03.0f}_el{elevation:+03.0f}")
        
        self.adaptive_beams = beamformed_signals
        self.beam_labels = direction_labels
        
        print(f"Created {len(beamformed_signals)} adaptive {pattern} beamformed signals")
        return beamformed_signals, direction_labels
    
    def analyze_beam_independence(self, frame_length=4096, overlap=0.5):
        """Analyze correlation between adaptive beams"""
        
        if not hasattr(self, 'adaptive_beams'):
            raise ValueError("Run create_adaptive_beams() first")
        
        n_beams = len(self.adaptive_beams)
        
        # Handle single beam case
        if n_beams == 1:
            print("\nOnly 1 beam created - no correlations to compute")
            print("Consider relaxing correlation threshold or spatial separation")
            self.beam_correlations = np.array([[1.0]])
            return self.beam_correlations
        
        # Compute spectrograms for each beam
        spectrograms = []
        for beam in self.adaptive_beams:
            f, t, stft = signal.stft(beam, self.fs, 
                                   nperseg=frame_length,
                                   noverlap=int(frame_length * overlap))
            # Use power spectral density
            psd = np.mean(np.abs(stft)**2, axis=1)  # Average over time
            spectrograms.append(psd)
        
        # Compute correlation matrix
        spectrograms = np.array(spectrograms)
        correlation_matrix = np.corrcoef(spectrograms)
        
        self.beam_correlations = correlation_matrix
        
        # Print correlation summary
        correlations = []
        print("\nAdaptive beam correlations:")
        for i in range(n_beams):
            for j in range(i+1, n_beams):
                corr = correlation_matrix[i, j]
                correlations.append(abs(corr))
                print(f"  {self.beam_labels[i]} <-> {self.beam_labels[j]}: {corr:.3f}")
        
        if len(correlations) > 0:
            print(f"\nCorrelation statistics:")
            print(f"  Mean |correlation|: {np.mean(correlations):.3f}")
            print(f"  Max |correlation|: {np.max(correlations):.3f}")
            print(f"  Correlations < 0.6: {np.sum(np.array(correlations) < 0.6)}/{len(correlations)}")
        
        return correlation_matrix
    
    def export_adaptive_beams(self, output_dir="adaptive_beams", format="wav", 
                             normalize=True, bit_depth=16):
        """Export adaptive beamformed signals as individual audio files"""
        
        if not hasattr(self, 'adaptive_beams'):
            raise ValueError("Run create_adaptive_beams() first")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = []
        
        for i, (beam, label) in enumerate(zip(self.adaptive_beams, self.beam_labels)):
            # Prepare audio data
            audio_data = beam.copy()
            
            # Normalize if requested
            if normalize:
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    # Normalize to 90% of full scale to prevent clipping
                    audio_data = audio_data * (0.9 / max_val)
            
            # Convert to appropriate bit depth
            if bit_depth == 16:
                audio_data = (audio_data * 32767).astype(np.int16)
            elif bit_depth == 24:
                audio_data = (audio_data * 8388607).astype(np.int32)
            elif bit_depth == 32:
                audio_data = audio_data.astype(np.float32)
            else:
                raise ValueError("bit_depth must be 16, 24, or 32")
            
            # Create filename
            filename = f"beam_{i+1:02d}_{label}.{format}"
            filepath = os.path.join(output_dir, filename)
            
            # Export based on format
            if format.lower() == 'wav':
                wavfile.write(filepath, self.fs, audio_data)
            elif format.lower() in ['flac', 'aiff']:
                try:
                    import soundfile as sf
                    sf.write(filepath, audio_data, self.fs)
                except ImportError:
                    print("Warning: soundfile not installed. Using wav format instead.")
                    filepath = filepath.replace(f".{format}", ".wav")
                    wavfile.write(filepath, self.fs, audio_data)
            
            exported_files.append(filepath)
            
            # Get peak direction info if available
            direction_info = ""
            if hasattr(self, 'peak_directions') and i < len(self.peak_directions):
                az, el, power = self.peak_directions[i]
                direction_info = f" (az={az:3.0f}°, el={el:+3.0f}°, power={power:.2e})"
            
            print(f"Exported: {filename}{direction_info}")
        
        print(f"\n✅ Exported {len(exported_files)} beamformed audio files to '{output_dir}/'")
        self.exported_files = exported_files
        return exported_files
    
    def export_beam_metadata(self, output_dir="adaptive_beams", filename="beam_metadata.csv"):
        """Export metadata about the beams"""
        
        if not hasattr(self, 'adaptive_beams'):
            raise ValueError("Run create_adaptive_beams() first")
        
        import pandas as pd
        import os
        
        metadata = []
        
        for i, label in enumerate(self.beam_labels):
            row = {
                'beam_id': i + 1,
                'filename': f"beam_{i+1:02d}_{label}.wav",
                'label': label,
                'duration_seconds': len(self.adaptive_beams[i]) / self.fs
            }
            
            # Add direction info if available
            if hasattr(self, 'peak_directions') and i < len(self.peak_directions):
                az, el, power = self.peak_directions[i]
                row.update({
                    'azimuth_deg': az,
                    'elevation_deg': el, 
                    'srp_power': power,
                    'power_rank': i + 1
                })
            
            # Add RMS power of the beam
            rms_power = np.sqrt(np.mean(self.adaptive_beams[i]**2))
            row['rms_power'] = rms_power
            
            metadata.append(row)
        
        df = pd.DataFrame(metadata)
        
        # Add correlation summary if available
        if hasattr(self, 'beam_correlations'):
            n_beams = len(self.adaptive_beams)
            for i in range(n_beams):
                max_corr = 0
                for j in range(n_beams):
                    if i != j:
                        max_corr = max(max_corr, abs(self.beam_correlations[i, j]))
                df.loc[i, 'max_correlation'] = max_corr
        
        # Save metadata
        metadata_path = os.path.join(output_dir, filename)
        df.to_csv(metadata_path, index=False)
        print(f"📊 Exported beam metadata to '{metadata_path}'")
        
        return df
    
    def plot_srp_map(self, figsize=(12, 8)):
        """Plot the SRP map"""
        
        if not hasattr(self, 'srp_map'):
            raise ValueError("Run compute_srp() first")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # SRP heatmap
        im = ax1.imshow(self.srp_map, aspect='auto', origin='lower',
                       extent=[self.azimuths[0], self.azimuths[-1], 
                              self.elevations[0], self.elevations[-1]])
        ax1.set_xlabel('Azimuth (degrees)')
        ax1.set_ylabel('Elevation (degrees)')
        ax1.set_title('Steered Response Power Map')
        plt.colorbar(im, ax=ax1, label='Power')
        
        # Mark peak directions if available
        if hasattr(self, 'peak_directions'):
            for az, el, power in self.peak_directions:
                ax1.plot(az, el, 'r*', markersize=15, markeredgecolor='white')
        
        # Power vs direction index
        sorted_powers = sorted(self.direction_powers, reverse=True)
        ax2.plot(sorted_powers)
        ax2.set_xlabel('Direction rank')
        ax2.set_ylabel('Power')
        ax2.set_title('Power Distribution')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def run_full_pipeline(self, input_file, output_dir="adaptive_beams", 
                         angular_resolution=8, num_peaks=8, min_separation=25,
                         max_correlation=0.6, pattern='hypercardioid', 
                         elevation_range=(-60, 60), export_audio=True,
                         fallback_to_fixed_grid=True):
        """Complete pipeline with correlation-based beam selection"""
        
        print("🎵 Starting correlation-aware adaptive beamforming pipeline...")
        
        # 1. Load audio
        print("\n1️⃣ Loading audio...")
        self.load_bformat(input_file, order=2)
        
        # 2. Compute SRP with limited elevation range
        print(f"\n2️⃣ Computing SRP (elevation range: {elevation_range})...")
        self.compute_srp(angular_resolution=angular_resolution, 
                        elevation_range=elevation_range,
                        pattern=pattern)
        
        # 3. Find decorrelated peaks
        print(f"\n3️⃣ Finding decorrelated peaks (max correlation: {max_correlation})...")
        self.find_peak_directions(num_peaks=num_peaks, 
                                min_separation=min_separation,
                                max_correlation=max_correlation)
        
        # 4. Fallback to fixed grid if too few beams
        if len(self.peak_directions) < 3 and fallback_to_fixed_grid:
            print(f"\n⚠️  Only found {len(self.peak_directions)} decorrelated beams.")
            print("🔄 Falling back to fixed grid approach...")
            
            # Use your original successful fixed directions
            fixed_directions = [
                (180, -30),  # down-south
                (0, 45),     # up-north
                (90, 0),     # east
                (270, 0),    # west
                (315, 15),   # northwest elevated
            ]
            
            # Filter directions that fit in elevation range
            valid_fixed = [(az, el) for az, el in fixed_directions 
                          if elevation_range[0] <= el <= elevation_range[1]]
            
            print(f"Using {len(valid_fixed)} fixed grid directions")
            self.create_adaptive_beams(directions=valid_fixed, pattern=pattern)
            
        else:
            # 5. Create adaptive beams
            print("\n4️⃣ Creating adaptive beams...")
            self.create_adaptive_beams(pattern=pattern)
        
        # 6. Verify independence
        print("\n5️⃣ Verifying beam independence...")
        correlations = self.analyze_beam_independence()
        
        # 7. Export results
        if export_audio:
            print("\n6️⃣ Exporting audio files...")
            self.export_adaptive_beams(output_dir=output_dir)
            self.export_beam_metadata(output_dir=output_dir)
        
        # 8. Plot results
        print("\n7️⃣ Generating plots...")
        fig = self.plot_srp_map()
        
        import os
        plot_path = os.path.join(output_dir, "srp_analysis.png")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Saved SRP plot to '{plot_path}'")
        
        # Summary of results
        if len(self.adaptive_beams) > 1:
            max_corr = np.max(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))
        else:
            max_corr = 0.0
        
        print(f"\n✅ Pipeline complete!")
        print(f"   📊 Created {len(self.adaptive_beams)} beams")
        if len(self.adaptive_beams) > 1:
            print(f"   📈 Maximum correlation: {max_corr:.3f}")
        
        if max_corr < 0.6:
            print(f"   🎯 Good independence achieved!")
        elif max_corr < 0.8:
            print(f"   ⚠️  Moderate correlations - acceptable for most analysis")
        else:
            print(f"   🚨 High correlations - but still better than single recording")
        
        return {
            'num_beams': len(self.adaptive_beams),
            'max_correlation': max_corr,
            'beam_files': self.exported_files if export_audio else None,
            'correlations': correlations,
            'peak_directions': getattr(self, 'peak_directions', [])
        }

# Example usage
if __name__ == "__main__":
    beamformer = AdaptiveAmbisonicBeamformer()
    
    # OPTION 1: Try adaptive first, fallback to fixed grid
    results = beamformer.run_full_pipeline(
        input_file='ambi-test_1.wav',
        output_dir='hybrid_beams',
        max_correlation=0.8,         # More permissive
        min_separation=30,           # Smaller separation
        elevation_range=(-45, 45),
        num_peaks=6,
        fallback_to_fixed_grid=True  # Safety net
    )
    
    # OPTION 2: Just use fixed grid (reliable)
    # beamformer.load_bformat('your_recording.wav', order=2)
    # fixed_directions = [
    #     (180, -30), (0, 45), (90, 0), (270, 0), (315, 15)
    # ]
    # beamformer.create_adaptive_beams(directions=fixed_directions)
    # beamformer.analyze_beam_independence()
    # beamformer.export_adaptive_beams()
    
    print(f"Final result: {results['num_beams']} beams with max correlation {results['max_correlation']:.3f}")