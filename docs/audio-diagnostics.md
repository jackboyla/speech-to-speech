# Audio diagnostics

`local` and `talk` automatically log a JSON support report with the first microphone
or speaker warning, audio setup/session failure, or server error, failed transcription
or failed response. No extra flag or rerun is needed. Healthy sessions print no
report, and repeated problems share one report per session. Report generation runs
outside the audio callback.

For example, the log for a microphone overflow includes:

```text
Microphone status: input overflow
Audio diagnostic report (copy with the warning/error for support):
...
```

Copy the warning/error and report into a support issue. The report includes
source and installed package versions, Python, OS/Linux distro,
CPU architecture, selected backends and configured model devices, audio rates and
buffer settings, and the selected sounddevice devices and host APIs. It omits API
keys, endpoint URLs, prompts, tools, raw configuration, conversations and audio.

`configured` contains settings, including defaults and global device overrides.
A null input/output device selects the sounddevice default. A zero callback block
size lets PortAudio vary the size. `observed.audio_devices` contains queried
device metadata; its default sample rates and latencies are not measurements of
an open stream. Missing PortAudio or an unavailable device leaves the rest of the
report usable. When streams have opened, `observed.streams` contains their
PortAudio-reported device, rate, block size and estimated latency, plus whether
they started. An input overflow trigger records that at least one overflow
occurred; the client keeps no cumulative count. This is a snapshot at the first
warning or failure, not a running monitor.

For a report without starting a session, put `audio-report` before your usual
command. It accepts the same flags or YAML file options:

```bash
speech-to-speech audio-report local > audio-report.json
speech-to-speech audio-report local -f pipeline.yaml --name assistant
speech-to-speech audio-report talk --input-device 2 --output-device 3
speech-to-speech audio-report serve
```

The manual command does not load models or tool modules, connect, or open audio
streams, so runtime information is unavailable. Neither report measures CPU load
or actual model placement after automatic device selection or fallback. `talk`
cannot inspect remote server settings; use `audio-report serve` there if needed.
`serve` has no local microphone or speakers. Reports collect support context;
they do not diagnose or fix the cause of an overflow.
