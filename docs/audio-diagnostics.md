# Audio diagnostics

Put `audio-report` before your usual command to print a JSON report, then exit:

```bash
speech-to-speech audio-report local
speech-to-speech audio-report local --device cpu --local_audio_chunk_size 2048
speech-to-speech audio-report local -f pipeline.yaml --name assistant
speech-to-speech audio-report talk --input-device 2 --output-device 3
speech-to-speech audio-report serve
```

Copy the output into a support issue or redirect it to a file with `> audio-report.json`.
The report includes source and installed package versions, Python, OS/Linux distro,
CPU architecture, selected backends and configured model devices, audio rates and
buffer settings, and the selected sounddevice devices and host APIs. It omits API
keys, endpoint URLs, prompts, tools, raw configuration, conversations and audio.

`configured` contains settings, including defaults and global device overrides.
A null input/output device selects the sounddevice default. A zero callback block
size lets PortAudio choose the size. `observed.audio_devices` contains queried
device metadata; its default sample rates and latencies are not measurements of
an open stream. Missing PortAudio or an unavailable device leaves the rest of the
report usable.

The command does not load models or tool modules, connect to a server, or open
audio streams. It cannot report actual model placement after device fallback,
negotiated stream settings, CPU load, or previous microphone overflow counts:
the client logs overflow warnings but keeps no counter. `talk` cannot inspect
remote server settings; run a separate `audio-report serve` on that server.
`serve` has no local microphone or speaker selection. This report helps collect
support context; it does not diagnose or fix the cause of an overflow.
