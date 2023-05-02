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
    #[arg(long, default_value_t = 1)]
    threads: u8,

    /// whisper parse target language
    #[arg(short, long, default_value = "en")]
    lang: String,

    #[arg()]
    url: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let ctx = WhisperContext::new(&args.model).expect("failed to load model");
    let mut state = ctx.create_state().expect("failed to create state");
    let speech_config = SpeechConfig::new(args.threads as c_int, Some(&args.lang));

    let mut buffer: Vec<u8> = vec![];
    let mut last_processed = Instant::now();
    let collect_time = Duration::from_secs(5);

    let (mut child, stdout) = get_yt_dlp_stdout(&args.url);
    let mut reader = BufReader::new(stdout);

    let mut process = |buffer: &[u8]| -> Result<(), Box<dyn Error>> {
        if let Ok(audio_data) = audio::get_audio_data(buffer) {
            speech::process(
                &mut state,
                &audio_data,
                &speech_config,
                |segment, start, end| {
                    println!("[{} - {}] {}", start, end, segment);
                },
            );
        }

        Ok(())
    };

    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            break;
        }

        let len = buf.len();
        buffer.extend_from_slice(buf);

        if last_processed.elapsed() >= collect_time {
            if !buffer.is_empty() {
                process(&buffer).expect("failed to process");
            }

            last_processed = Instant::now();
            buffer.clear();
        }

        reader.consume(len);
    }

    if !buffer.is_empty() {
        process(&buffer).expect("failed to process");
    }

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
