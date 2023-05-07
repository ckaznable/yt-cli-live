# Youtube Text Live Streaming in CLI

This project is currently a work in progress (WIP). It aims to enable streaming YouTube videos and converting the audio into text, displaying it in the command line interface (CLI). The project utilizes the [whisper-rs](https://github.com/tazz4843/whisper-rs), [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and [yt-dlp](https://github.com/yt-dlp/yt-dlp) libraries and is being developed in Rust.

Please note that the project is still under active development, and certain features or functionalities may be incomplete or subject to change. Contributions, suggestions, and bug reports are welcome.

## Requirement

- yt-dlp

Using yt-dlp for youtube streaming

## Usage

```text
youtube text streaming in cli

Usage: yt-cli-live [OPTIONS] --model <MODEL> <URL>

Arguments:
  <URL>  youtube url or youtube video id

Options:
  -m, --model <MODEL>      path of whisper model
  -t, --threads <THREADS>  usage thread number for whisper [default: 1]
  -l, --lang <LANG>        whisper parse target language [default: en]
  -v, --verbose            show log of runtime
  -h, --help               Print help
  -V, --version            Print version
```

## LICENSE

MIT
