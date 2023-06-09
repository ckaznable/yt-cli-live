# Youtube Text Live Streaming in CLI

This project aims to enable streaming YouTube videos and converting the audio into text, displaying it in the command line interface (CLI). The project utilizes the [whisper-rs](https://github.com/tazz4843/whisper-rs), [whisper.cpp](https://github.com/ggerganov/whisper.cpp), [silero-vad](https://github.com/snakers4/silero-vad) and [yt-dlp](https://github.com/yt-dlp/yt-dlp) libraries and is being developed in Rust.

## Requirement

- [yt-dlp](https://github.com/yt-dlp/yt-dlp)

This project using yt-dlp for youtube streaming

- whisper model

This project using whisper for ASR(Automatic Speech Recognition)

then you can following [whisper.cpp](https://github.com/ggerganov/whisper.cpp) README to download whisper models

Suggested use of base or small model

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

### Example

```shell
yt-cli-live -t 8 -m <model path> -l ja <youtube streaming id or url>
```

## Build Dependencies

- rustc

yt-cli-live is written in Rust, so you'll need to grab a [Rust installation](https://www.rust-lang.org/) in order to compile it.

- libclang-dev

you need `libclang-dev` in `linux`

## Building

```shell
git clone https://github.com/ckaznable/yt-cli-live
cd yt-cli-live
cargo build --release
```

## LICENSE

MIT
