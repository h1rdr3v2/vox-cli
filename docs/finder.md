# Finder Quick Actions

Two right-click actions for Finder:

- **Transcribe with vox**: for audio and video files. Runs `vox transcribe` on each file and writes the transcript next to it, in your `default_format` (`.txt` unless you changed it with `vox config set default_format srt`).
- **Speak with vox**: for `.txt` and `.md` files. Runs `vox speak` on each file and writes a `.wav` next to it.

Both show a notification when they start ("Transcribing interview.m4a") and when they finish ("Saved interview.txt", or what went wrong). While they run, the gear in the menu bar shows real progress in 10% steps; click its ✕ to cancel.

Both call a small helper script, `~/.config/vox/finder/vox-finder.sh`. Quick Actions and Shortcuts run with a minimal PATH, so the helper uses the absolute paths of `vox` and `ffmpeg`, found when you run `vox setup finder`.

Before you start, install the models you want (Quick Actions cannot show the first-run picker):

```bash
vox models pull whisper-large-v3-turbo
vox models pull kokoro-82m
```

## Option 1: let vox create them (recommended)

```bash
vox setup finder
```

This writes the helper script and creates two Quick Actions in `~/Library/Services`:

- `Transcribe with vox.workflow`
- `Speak with vox.workflow`

Right-click a file in Finder and look under **Quick Actions** (or **Services**). If they do not appear, open **System Settings > General > Login Items & Extensions**, click the info button next to **Finder** (or **Actions** on older macOS), and turn them on.

Run `vox setup finder` again after reinstalling vox or ffmpeg somewhere else, so the paths stay correct. To see the paths without changing anything: `vox setup finder --print-only`.

## Removing them

```bash
vox setup finder --uninstall
```

This deletes both Quick Actions from `~/Library/Services` and the helper script. `vox uninstall` does the same as part of removing vox. Shortcuts you built yourself in the Shortcuts app (option 2) live in the Shortcuts app: right-click them there and choose **Delete**.

## Option 2: build them in Shortcuts

Run `vox setup finder` once anyway: it writes the helper script and prints the exact command line for each shortcut.

### Transcribe with vox

1. Open **Shortcuts** and create a new shortcut named **Transcribe with vox**.
2. Open the shortcut's details (the info button) and turn on **Use as Quick Action**, with **Finder** checked.
3. At the top, set it to receive **Files** (or **Media**) from **Quick Actions**.
4. Add the action **Run Shell Script**:
   - Shell: `zsh`
   - Input: **Shortcut Input**
   - Pass input: **as arguments**
   - Script (use the line `vox setup finder` printed; it looks like this):

     ```
     /Users/you/.config/vox/finder/vox-finder.sh transcribe "$@"
     ```

5. Save. Right-click a video in Finder, choose **Quick Actions > Transcribe with vox**, and a `.txt` appears next to it.

The first run may ask you to allow Shortcuts to run scripts: **Shortcuts > Settings > Advanced > Allow Running Scripts**.

### Speak with vox

Same steps, named **Speak with vox**, receiving **Text files** (or **Files**), with the script:

```
/Users/you/.config/vox/finder/vox-finder.sh speak "$@"
```

## What the helper does

```
vox-finder.sh transcribe FILE...    # FILE.txt (or your default_format) next to each file
vox-finder.sh speak FILE...         # FILE.wav next to each file
```

It runs vox once per file with stdin closed, so vox never stops to ask a question, and notifies you when it starts and when it finishes, for example "Saved interview.txt".

macOS shows a Quick Action's progress by counting its finished steps, and ignores progress reported from inside a step. So the generated Quick Actions have ten steps: `vox-finder.sh step 1 transcribe FILE...` starts the job in the background, and steps 2 to 10 (`vox-finder.sh step N JOB`) each return once the next tenth of the work is done. vox reports how far it is through `--progress-file`. Shortcuts run the one-step form above, so they show no percentage. If something fails, the notification shows the file and vox's error with its hint, for example: `No STT model installed. Run: vox models pull whisper-large-v3-turbo`.

You can call the helper from anywhere, for example Folder Actions or a Hazel rule.

## Troubleshooting

- **Nothing happens**: run the helper by hand in Terminal with a file to see its output:
  `~/.config/vox/finder/vox-finder.sh transcribe ~/Desktop/clip.mp4`
- **"ffmpeg is not installed"**: `brew install ffmpeg`, then run `vox setup finder` again.
- **Slow first file**: the first request starts the vox server and loads the model (a few seconds). Files after that are fast until the server exits after 5 idle minutes.
- The server log is at `~/.cache/vox/run/server.log`.
