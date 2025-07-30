from pathlib import Path
import warnings
import soundfile as sf

from ambisonic_beamforming import AmbisonicBeamformer
from beamforming_utils.beamforming_grid import latlong_grid, fibonacci_sphere_grid, unit_vector_from_azel
from beamforming_utils.beamforming_analysis import calculate_correlation_analysis, smart_direction_selection, calculate_uniqueness_metrics
from beamforming_utils.beamforming_export import export_selected_directions, export_non_selected_directions
from beamforming_utils.beamforming_report import (
    create_selection_report, create_selection_visualization, plot_exported_spectrograms,
    generate_individual_spectrograms, create_html_report
)

from beamforming_utils.config import (
    GRID_MODE, AZ_STEP_DEG, EL_PLANES_DEG, INCLUDE_POLES, FIBONACCI_COUNT,
    USE_CORRELATION_FILTER, CORRELATION_THRESHOLD, USE_MIN_ANGLE_FILTER, MIN_ANGULAR_SEPARATION_DEG,
    DEFAULT_DURATION_SECONDS, DEFAULT_START_TIME, MAX_EXPORTS, MIN_UNIQUENESS_THRESHOLD,
    GENERATE_HTML_REPORT, EXPORT_ALL_DIRECTIONS
)

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
    generate_html_report=GENERATE_HTML_REPORT,
    export_all_directions=EXPORT_ALL_DIRECTIONS,
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

    # Step 4b: Export non-selected directions if requested
    all_exported_files = exported_files.copy()
    non_selected_files = []
    if export_all_directions:
        non_selected_directions = [d for d in directions if d not in selected_directions]
        if non_selected_directions:
            print(f"Step 4b: Exporting {len(non_selected_directions)} non-selected directions...")
            non_selected_files = export_non_selected_directions(
                beamformed_audio, directions, non_selected_directions, nested_output_path, duration_seconds, sample_rate
            )
            all_exported_files.extend(non_selected_files)

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

    # Step 5b: Generate individual spectrograms for HTML if needed
    individual_spectrograms = {}
    if generate_html_report:
        print("Step 5b: Generating individual spectrograms for HTML...")
        individual_spectrograms = generate_individual_spectrograms(
            exported_files, non_selected_files if export_all_directions else [], nested_output_path, sample_rate
        )

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
    
    # Step 8: Generate HTML report if requested
    if generate_html_report:
        print("Step 8: Generating interactive HTML report...")
        create_html_report(
            input_file,
            nested_output_path,
            exported_files,
            selected_directions,
            uniqueness_scores,
            indices_df_all,
            correlation_matrix,
            directions,
            duration_seconds,
            sample_rate,
            non_selected_files if export_all_directions else [],
            individual_spectrograms,
        )
    
    return exported_files

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
    parser.add_argument(
        "--html_report",
        action="store_true",
        help="Generate interactive HTML report for each processed file."
    )
    parser.add_argument(
        "--export_all",
        action="store_true",
        help="Export both selected and rejected directions. Rejected files are prefixed with 'rejected_'."
    )
    parser.add_argument(
        "--window",
        type=float,
        default=30.0,
        help="Window size in seconds for analysis (default: 30.0)."
    )
    parser.add_argument(
        "--hop",
        type=float,
        default=15.0,
        help="Hop size in seconds for windowed analysis (default: 15.0)."
    )
    parser.add_argument(
        "--pool",
        type=str,
        choices=["mean", "p80", "meanp80"],
        default="meanp80",
        help="Pooling method for windowed indices: mean, p80, or meanp80 (default: meanp80)."
    )
    parser.add_argument(
        "--pool_alpha",
        type=float,
        default=None,
        help="Alpha parameter for meanp80 pooling (0..1, default: 0.5, or 0.3 for 'water' profile)."
    )
    parser.add_argument(
        "--highlights",
        type=int,
        default=0,
        help="Number of highlight windows to export (default: 0, disables highlights)."
    )
    parser.add_argument(
        "--min_rms_db",
        type=float,
        default=-60.0,
        help="Minimum RMS (dB) threshold for window inclusion (default: -60.0)."
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional profile name (e.g., 'water') to adjust defaults."
    )
    args = parser.parse_args()

    # Set pool_alpha default for 'water' profile if not explicitly provided
    if args.profile == "water" and args.pool_alpha is None:
        args.pool_alpha = 0.3
    elif args.pool_alpha is None:
        args.pool_alpha = 0.5

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
                generate_html_report=args.html_report,
                export_all_directions=args.export_all,
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
