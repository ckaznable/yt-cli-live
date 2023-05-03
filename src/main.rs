use clap::Parser;
use speech::SpeechConfig;
use std::{
    error::Error,
    ffi::c_int,
    io::{BufRead, BufReader},
    process::{Child, ChildStdout, Command, Stdio},
    time::{Duration, Instant},
};
use wait_timeout::ChildExt;
use whisper_rs::WhisperContext;

use util::Log;

mod util;
mod audio;
mod speech;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// path of whisper model
    #[arg(short, long)]
    model: String,

    /// usage thread number for whisper
    #[arg(short, long, default_value_t = 1)]
    threads: u8,

    /// whisper parse target language
    #[arg(short, long, default_value = "en")]
    lang: String,

    /// youtube url or youtube video id
    #[arg()]
    url: String,

    /// show log of runtime
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let logger = Log::new(args.verbose);

    let ctx = WhisperContext::new(&args.model).expect("failed to load model");
    let mut state = ctx.create_state().expect("failed to create state");
    let speech_config = SpeechConfig::new(args.threads as c_int, Some(&args.lang));

    let mut buffer: Vec<u8> = vec![];
    let mut last_processed = Instant::now();
    let collect_time = Duration::from_secs(5);

    let (mut child, stdout) = get_yt_dlp_stdout(&args.url);
    let mut reader = BufReader::new(stdout);

    let mut process = |buffer: &[u8]| {
        if buffer.is_empty() {
            return;
        }

        match audio::get_audio_data(buffer) {
            Ok(audio_data) => {
                logger.verbose(format!(
                    "Get {}kb audio data from ts",
                    audio_data.len() / 1024
                ));

                speech::process(
                    &mut state,
                    &audio_data,
                    &speech_config,
                    |segment, start, end| {
                        println!("[{} - {}] {}", start, end, segment);
                    },
                );
            }
            Err(str) => {
                logger.error(str);
            }
        }
    };

    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            break;
        }

        let len = buf.len();
        buffer.extend_from_slice(buf);

        if last_processed.elapsed() >= collect_time {
            logger.verbose(format!(
                "Reading {}kb data from yt-dlp",
                buffer.len() / 1024
            ));

            process(&buffer);

            last_processed = Instant::now();
            buffer.clear();
        }

        reader.consume(len);
    }

    process(&buffer);
    child
        .wait_timeout(Duration::from_secs(3))
        .expect("failed to wait on yt-dlp");

    Ok(())
}

fn get_yt_dlp_stdout(url: &str) -> (Child, ChildStdout) {
    let mut cmd = Command::new("yt-dlp");
    cmd.arg(url)
        .args(["-f", "w"])
        .args(["--quiet"])
        .args(["-o", "-"]);

    let mut child = cmd
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to execute yt-dlp");

    let stdout = child.stdout.take().expect("invalid stdout stream");

    (child, stdout)
}
