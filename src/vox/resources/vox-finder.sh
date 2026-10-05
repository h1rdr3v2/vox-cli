#!/bin/zsh
# vox Finder helper. Runs vox on files passed by a Finder Quick Action or a
# Shortcut and shows a notification when done. Written by `vox setup finder`.
#
# Usage: vox-finder.sh transcribe FILE...   (writes FILE.txt next to each file)
#        vox-finder.sh speak FILE...        (writes FILE.wav next to each file)

# Quick Actions and Shortcuts run with a minimal PATH, so use absolute paths.
VOX=(__VOX__)
export PATH="__PATH__:/usr/bin:/bin:/usr/sbin:/sbin"

notify() {
  /usr/bin/osascript - "$1" "$2" <<'APPLESCRIPT' >/dev/null 2>&1
on run argv
  display notification (item 2 of argv) with title (item 1 of argv)
end run
APPLESCRIPT
}

mode="$1"
shift
case "$mode" in
  transcribe) ext="txt" ;;
  speak) ext="wav" ;;
  *) echo "usage: vox-finder.sh transcribe|speak FILE..." >&2; exit 1 ;;
esac
if [ "$#" -eq 0 ]; then
  notify "vox" "No files were selected."
  exit 1
fi

ok=0
failed=0
first_error=""
last_written=""
for file in "$@"; do
  if output=$("${VOX[@]}" "$mode" "$file" 2>&1 </dev/null); then
    ok=$((ok + 1))
    name="${file:t}"
    last_written="${name:r}.$ext"
  else
    failed=$((failed + 1))
    if [ -z "$first_error" ]; then
      first_error="${file:t}: $(printf '%s\n' "$output" | grep -v '^[[:space:]]*$' | tail -n 2 | tr '\n' ' ')"
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
notify "vox: $failed of $# failed" "$first_error"
exit 1
