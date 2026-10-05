# Running vox as an always-on server (launchd)

By default vox needs no setup: `vox transcribe` and `vox speak` start a server when needed, and it exits after 5 idle minutes so it uses no memory while idle.

If other apps call vox's OpenAI-compatible API at `http://127.0.0.1:8880/v1`, you may want a server that is always there. `vox serve --persistent` never exits on idle. It still loads each model only when the first request needs it, and holds it after that. vox does not install a launch agent for you; here is how to add one.

## 1. Find the vox path

```bash
which vox
```

For example `/Users/you/.local/bin/vox`.

## 2. Create the launch agent

Save this as `~/Library/LaunchAgents/local.vox.server.plist`, replacing both `/Users/you` paths:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>local.vox.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/you/.local/bin/vox</string>
        <string>serve</string>
        <string>--persistent</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <!-- So vox can find ffmpeg (Homebrew) for mp3 output and video input. -->
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>/Users/you/.cache/vox/run/launchd.log</string>
</dict>
</plist>
```

## 3. Load it

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.vox.server.plist
```

Check it:

```bash
vox status
```

`vox status` reports `persistent` mode. The CLI uses this server too.

## Stopping and removing

`vox stop` stops the server, but with `KeepAlive` launchd starts it again. To stop it for good:

```bash
launchctl bootout gui/$(id -u)/local.vox.server
rm ~/Library/LaunchAgents/local.vox.server.plist
```

## Notes

- Only one vox server runs at a time. If the CLI started an on-demand server first, run `vox stop` before loading the agent.
- To use another port, add `<string>--port</string><string>9000</string>` to `ProgramArguments`, and set it for the CLI too with `vox config set port 9000`.
- Memory: a persistent server keeps every model it has loaded until it restarts. `launchctl kickstart -k gui/$(id -u)/local.vox.server` restarts it and frees that memory.
