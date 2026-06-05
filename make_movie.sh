make_movie() {
  local input_glob="$1"
  local outdir="$2"
  local prestring="$3"

  # --- checks ---
  if [[ -z "$input_glob" || -z "$outdir" || -z "$prestring" ]]; then
    echo "Usage: make_movie '<input_glob>' <output_dir> <prestring>"
    return 1
  fi

  mkdir -p "$outdir"

  # --- derive a suffix from the glob (remove path + wildcard prefix) ---
  local base_pattern
  base_pattern=$(basename "$input_glob")

  # strip everything up to first occurrence of "panel_" (customize if needed)
  local suffix="${base_pattern#*panel_}"
  suffix="${suffix%.png}"

  # --- output filenames ---
  local outfile="${outdir}/${prestring}_panel_${suffix}.mp4"
  local tmpfile="${outdir}/${prestring}_panel_${suffix}_tmp.mp4"

  echo "Creating movie: $outfile"

  # --- create movie ---
  ffmpeg -pattern_type glob -i "$input_glob" \
    -framerate 24 \
    -c:v libx265 \
    -pix_fmt yuv420p \
    -vf "scale=2*trunc(iw/2):2*trunc(ih/2),setsar=1" \
    "$outfile"

  # --- fix for Apple compatibility (hvc1 tag) ---
  ffmpeg -i "$outfile" -c copy -tag:v hvc1 "$tmpfile"

  mv "$tmpfile" "$outfile"

  echo "Done: $outfile"
}

# If script is executed (not sourced), run the function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  make_movie "$@"
fi
