use clap::Parser;
use ringbuf::{Consumer, LocalRb, SharedRb};
use speech::{SpeechConfig, WhisperPayload};
use std::{
    error::Error,
    ffi::c_int,
    io::{BufRead, BufReader},
    mem::MaybeUninit,
    process::{Child, ChildStdout, Command, Stdio},
    sync::{
        mpsc::{self, Receiver},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Instant,
};
use whisper_rs::WhisperContext;

use util::Log;

mod audio;
mod speech;
mod util;
mod vad;

type U8Consumer = Consumer<u8, Arc<SharedRb<u8, Vec<MaybeUninit<u8>>>>>;

enum ThreadState {
    End,
    Sync,
}

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
    let (tx, rx) = mpsc::channel::<ThreadState>();

    // 1Mb buffer
    let rb_size = 1024 * 1024;
    let mut rb = LocalRb::<u8, Vec<_>>::new(rb_size);
    let (mut ts_prod, mut ts_cons) = rb.split_ref();

    let (mut child, stdout) = get_yt_dlp_stdout(&args.url);
    let mut reader = BufReader::new(stdout);

    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            break;
        }

        let len = buf.len();
        ts_prod.push_slice(buf);

        if ts_prod.is_full() {
            todo!();
        }

        reader.consume(len);
    }

    child.kill().expect("failed to wait on yt-dlp");
    let _ = tx.send(ThreadState::End);

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

fn evoke_whisper_thread(
    args: Args,
    rx: Receiver<ThreadState>,
    mut cons: U8Consumer,
) -> JoinHandle<()> {
    let ctx = WhisperContext::new(&args.model).expect("failed to load model");
    let logger = Log::new(args.verbose);

    thread::spawn(move || {
        let mut state = ctx.create_state().expect("failed to create state");
        let mut streaming_time = 0.0f64;

        while let Ok(ThreadState::Sync) = rx.recv() {
            if cons.is_empty() {
                logger.error(audio::Error::Empty.to_string());
                continue;
            }

            let data = cons.pop_iter().collect::<Vec<u8>>();
            logger.verbose(format!("Reading {}kb data from yt-dlp", data.len() / 1024));

            match audio::get_audio_data(&data) {
                Ok((audio_data, dur)) => {
                    logger.verbose(format!(
                        "Get {}kb audio data and duration {:.3}s from ts",
                        audio_data.len() / 1024,
                        dur
                    ));

                    let config = SpeechConfig::new(args.threads as c_int, Some(&args.lang));
                    let mut payload: WhisperPayload = WhisperPayload::new(&audio_data, config);
                    let running_calc = Instant::now();

                    let segment_time = (streaming_time * 1000.0) as i64;
                    streaming_time += dur;

                    speech::process(&mut state, &mut payload, &mut |segment, start| {
                        println!(
                            "[{}] {}",
                            util::format_timestamp_to_time(segment_time + start),
                            segment
                        );
                    });

                    logger.verbose(format!(
                        "whisper process time: {}s",
                        running_calc.elapsed().as_secs()
                    ));
                }
                Err(err) => {
                    logger.error(err.to_string());
                }
            }
        }
    })
}
