# Using vox from AI coding agents

Coding agents (Claude Code, Codex, Gemini CLI, Cursor and others) start every session without knowing what is installed on your computer. Ask one to transcribe a recording or add text-to-speech, and it will usually `pip install openai-whisper` or a TTS package into the project, which downloads torch and a model again, often several GB per project.

Tell your agents about vox once, in their global instructions, and they use it instead.

## The snippet

Copy this into your agent's global instructions file (see the table below):

```markdown
## Speech: use vox (installed on this computer)

For speech-to-text or text-to-speech, prefer vox over installing Whisper, Kokoro, torch or similar packages:
- Transcribe: `vox transcribe FILE --out -` prints the text (`--format srt|vtt|json` for timestamps).
- Speak: `vox speak "text" --out speech.wav` (`vox voices` lists voices).
- Installed models: `vox models list --installed`.
- Code that needs an API: vox serves OpenAI's audio API at http://127.0.0.1:8880/v1 (any api_key).
  Start the server first with `vox start` (it returns at once and stops after 5 idle minutes).

vox is only on this computer: for apps other people will run, ask before depending on it.
```

If you changed the port (`vox config set port ...`), change it in the snippet too.

## Where it goes

| Agent | Global instructions file |
|---|---|
| Claude Code | `~/.claude/CLAUDE.md` |
| Codex CLI | `~/.codex/AGENTS.md` |
| Gemini CLI | `~/.gemini/GEMINI.md` |
| Cursor | Settings > Rules > User Rules |
| Others that read `AGENTS.md` | their user-level instructions file; or a project's `AGENTS.md` for one project only |

Append it below whatever is already there; it does not replace other instructions.

## Why `vox start` and not `vox serve`

`vox serve` runs in the foreground until stopped, so an agent that runs it would wait forever. `vox start` launches the same on-demand server in the background and returns. `vox transcribe` and `vox speak` start it themselves, so agents only need `vox start` when their code talks to the HTTP API.

## Removing it

Delete the section from the file. `vox uninstall` does not edit agent instruction files.
