#!/bin/sh
# vox file-manager helper for Linux. Runs vox on files passed by a right-click
# action and shows a desktop notification when done. Written by `vox setup files`.
#
# Usage: vox-files.sh transcribe FILE...   (writes FILE.txt next to each file)
#        vox-files.sh speak FILE...        (writes FILE.wav next to each file)

# File managers may start actions with a minimal PATH, so use absolute paths.
run_vox() { __VOX__ "$@"; }
export PATH="__PATH__:/usr/local/bin:/usr/bin:/bin"

notify() {
  if command -v notify-send >/dev/null 2>&1; then
    notify-send "$1" "$2"
  fi
}

mode="$1"
[ "$#" -gt 0 ] && shift
case "$mode" in
  transcribe) ext="txt"; verb="Transcribing" ;;
  speak) ext="wav"; verb="Speaking" ;;
  *) echo "usage: vox-files.sh transcribe|speak FILE..." >&2; exit 1 ;;
esac
if [ "$#" -eq 0 ]; then
  notify "vox" "No files were selected."
  exit 1
fi
if [ "$#" -eq 1 ]; then
  notify "vox" "$verb $(basename "$1")"
else
  notify "vox" "$verb $# files"
fi

total=$#
ok=0
failed=0
first_error=""
last_written=""
for file in "$@"; do
  if output=$(run_vox "$mode" "$file" 2>&1 </dev/null); then
    ok=$((ok + 1))
    name=$(basename "$file")
    last_written="${name%.*}.$ext"
  else
    failed=$((failed + 1))
    if [ -z "$first_error" ]; then
      first_error="$(basename "$file"): $(printf '%s\n' "$output" | grep -v '^[[:space:]]*$' | tail -n 2 | tr '\n' ' ')"
    fi
  fi
done

if [ "$failed" -eq 0 ]; then
  if [ "$ok" -eq 1 ]; then
    notify "vox" "Saved $last_written"
  else
    notify "vox" "Done: $ok files"
  fi
  exit 0
fi
notify "vox: $failed of $total failed" "$first_error"
exit 1
