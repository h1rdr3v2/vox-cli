#!/bin/zsh
# vox Finder helper. Runs vox on files passed by a Finder Quick Action or a
# Shortcut and shows a notification when done. Written by `vox setup finder`.
#
# Usage: vox-finder.sh transcribe FILE...   (writes FILE.txt, or your default_format, next to each file)
#        vox-finder.sh speak FILE...        (writes FILE.wav next to each file)
#        vox-finder.sh step 1 MODE FILE...  (Quick Action step 1: start, wait for 10%)
#        vox-finder.sh step N JOB           (Quick Action steps 2 to 10)
#
# macOS shows a Quick Action's progress by counting its finished steps, so the
# Quick Actions have ten steps: the first starts the job in the background and
# each one returns when the next tenth of the work is done.

# Quick Actions and Shortcuts run with a minimal PATH, so use absolute paths.
VOX=(__VOX__)
export PATH="__PATH__:/usr/bin:/bin:/usr/sbin:/sbin"
STEPS=10

notify() {
  if [[ -n $VOX_FINDER_NOTIFY_LOG ]]; then  # tests read notifications from a file
    print -r -- "$1|$2" >> "$VOX_FINDER_NOTIFY_LOG"
    return
  fi
  /usr/bin/osascript - "$1" "$2" <<'APPLESCRIPT' >/dev/null 2>&1
on run argv
  display notification (item 2 of argv) with title (item 1 of argv)
end run
APPLESCRIPT
}

usage() {
  echo "usage: vox-finder.sh transcribe|speak FILE... | step N ..." >&2
  exit 1
}

# Run MODE on each file, keeping JOB/position ("index total") and
# JOB/progress (progress within the current file) up to date.
run_all() {
  local mode=$1 job=$2 ext output name
  shift 2
  local total=$# ok=0 failed=0 index=0 first_error="" last_written=""
  local verb
  case $mode in
    transcribe)
      # Transcripts use the default_format setting: txt, srt, vtt or json.
      ext=$("${VOX[@]}" config get default_format 2>/dev/null)
      [[ $ext == (txt|srt|vtt|json) ]] || ext="txt"
      verb="Transcribing"
      ;;
    speak) ext="wav"; verb="Speaking" ;;
  esac
  if (( total == 1 )); then
    notify "vox" "$verb ${1:t}"
  else
    notify "vox" "$verb $total files"
  fi
  for file in "$@"; do
    print -r -- 0 > "$job/progress"
    print -r -- "$index $total" > "$job/position"
    if output=$("${VOX[@]}" "$mode" "$file" --progress-file "$job/progress" 2>&1 </dev/null); then
      ok=$((ok + 1))
      name="${file:t}"
      last_written="${name:r}.$ext"
    else
      failed=$((failed + 1))
      if [[ -z $first_error ]]; then
        first_error="${file:t}: $(print -r -- "$output" | grep -v '^[[:space:]]*$' | tail -n 2 | tr '\n' ' ')"
      fi
    fi
    index=$((index + 1))
  done
  print -r -- "$total $total" > "$job/position"
  if (( failed == 0 )); then
    if (( ok == 1 )); then
      notify "vox" "Saved $last_written"
    else
      notify "vox" "Done: $ok files"
    fi
  else
    notify "vox: $failed of $total failed" "$first_error"
  fi
  print -r -- $failed > "$job/done"
}

# Fraction of the whole job done, 0 to 1.
fraction() {
  local job=$1 index=0 total=1 within=0
  [[ -r $job/position ]] && read index total < "$job/position"
  [[ -r $job/progress ]] && read within < "$job/progress"
  (( total > 0 )) || total=1
  print -r -- $(( (index + within) / (total + 0.0) ))
}

# Delete a job folder, and nothing that is not one.
remove_job() {
  [[ $1 == */vox-job.* && -d $1 ]] && rm -rf -- "$1"
}

kill_tree() {
  local child
  for child in $(pgrep -P "$1"); do kill_tree "$child"; done
  kill "$1" 2>/dev/null
}

# Return once the job has done STEP tenths of its work (the last step waits
# for the end). Cancelling the Quick Action stops the job.
wait_for() {
  local job=$1 step=$2 pid
  read pid < "$job/pid"
  trap 'kill_tree $pid; remove_job "$job"; exit 1' TERM INT HUP
  while [[ ! -e $job/done ]]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      notify "vox" "The job stopped before it finished."
      remove_job "$job"
      exit 1
    fi
    if (( step < STEPS )) && (( $(fraction "$job") * STEPS >= step )); then
      break
    fi
    sleep 0.5
  done
  if (( step < STEPS )); then
    print -r -- "$job"  # the next step's input
  else
    remove_job "$job"
  fi
}

new_job() {
  mktemp -d "${TMPDIR:-/tmp}/vox-job.XXXXXX"
}

case $1 in
  transcribe|speak)
    mode=$1
    shift
    if [[ $# -eq 0 ]]; then
      notify "vox" "No files were selected."
      exit 1
    fi
    job=$(new_job)
    print -r -- $$ > "$job/pid"
    run_all "$mode" "$job" "$@"
    read failed < "$job/done"
    remove_job "$job"
    (( failed == 0 ))
    ;;
  step)
    step=$2
    [[ $step == <1-> ]] || usage
    shift 2
    if (( step == 1 )); then
      mode=$1
      [[ $mode == (transcribe|speak) ]] || usage
      shift
      if [[ $# -eq 0 ]]; then
        notify "vox" "No files were selected."
        exit 1
      fi
      job=$(new_job)
      # In the background, detached from this step, with no handle on its output.
      run_all "$mode" "$job" "$@" </dev/null >/dev/null 2>&1 &!
      print -r -- $! > "$job/pid"
    else
      job=$1
      [[ -n $job && -d $job && $job == */vox-job.* ]] || exit 0
    fi
    wait_for "$job" "$step"
    ;;
  *)
    usage
    ;;
esac
