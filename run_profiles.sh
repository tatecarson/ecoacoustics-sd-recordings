PROFILES="wind rain surf_river thunder geophony_general none"
INPUT_DIR="profile-audio-test"

for P in $PROFILES; do
  python beamforming-export-scikit-maad.py \
    --input_dir "$INPUT_DIR" \
    --profile "$P" \
    --output_dir "out_${P}" \
    --html_report
done