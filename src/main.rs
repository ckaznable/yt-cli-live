use audio::{resample_to_16k, YOUTUBE_TS_SAMPLE_RATE};
use clap::Parser;
use owo_colors::OwoColorize;
use ringbuf::{Consumer, HeapRb, LocalRb, Producer, Rb, SharedRb};
use speech::{SpeechConfig, WhisperPayload};
use std::{
    error::Error,
    ffi::c_int,
    io::{BufRead, BufReader},
    mem::MaybeUninit,
    process::{Child, ChildStdout, Command, Stdio},
    sync::{
        mpsc::{self, Receiver, SyncSender},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Instant,
};
use vad::{split_audio_data_with_window_size, VadState, WINDOW_SIZE_SAMPLES};
use whisper_rs::WhisperContext;

use util::Log;

mod audio;
mod speech;
mod util;
mod vad;

type F32Consumer = Consumer<f32, Arc<SharedRb<f32, Vec<MaybeUninit<f32>>>>>;
type SegmentProducer =
    Producer<vad::VadSegment, Arc<SharedRb<vad::VadSegment, Vec<MaybeUninit<vad::VadSegment>>>>>;
type SegmentConsumer =
    Consumer<vad::VadSegment, Arc<SharedRb<vad::VadSegment, Vec<MaybeUninit<vad::VadSegment>>>>>;

enum ThreadState {
    End,
    Sync,
}

#[derive(Parser, Debug, Clone)]
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

    let (mut child, stdout) = get_yt_dlp_stdout(&args.url);
    let mut reader = BufReader::new(stdout);

    // local buffer for ts file in 1Mb
    let rb_size = 1024 * 1024;
    let rb = LocalRb::<u8, Vec<_>>::new(rb_size);
    let (mut prod, mut cons) = rb.split();

    // shared buffer f32 transformed pcm in 30s audio data
    let rb_size = YOUTUBE_TS_SAMPLE_RATE as usize * 30;
    let rb = HeapRb::<f32>::new(rb_size);
    let (mut ts_prod, ts_cons) = rb.split();

    // shared buffer for vad output in 20 segment
    let rb = HeapRb::<vad::VadSegment>::new(20);
    let (vad_prod, vad_cons) = rb.split();

    let (tx, rx) = mpsc::sync_channel::<ThreadState>(1);
    let (vad_tx, vad_rx) = mpsc::sync_channel::<ThreadState>(1);

    let handle_vad = evoke_vad_thread(args.clone(), (vad_tx.clone(), rx), (vad_prod, ts_cons));
    let handle_whisper = evoke_whisper_thread(args, vad_rx, vad_cons);

    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            break;
        }

        let len = buf.len();
        prod.push_slice(buf);

        if prod.is_full() || prod.len() > 128000 {
            let data = cons.pop_iter().collect::<Vec<u8>>();
            logger.verbose(format!("Reading {}kb data from yt-dlp", data.len() / 1024));

            match audio::get_audio_data(&data) {
                Ok((audio_data, dur)) => {
                    logger.verbose(format!(
                        "Get {}kb audio data and duration {:.3}s from ts",
                        audio_data.len() / 1024,
                        dur
                    ));

                    ts_prod.push_slice(&audio_data);
                    if let Err(e) = tx.try_send(ThreadState::Sync) {
                        match e {
                            mpsc::TrySendError::Full(_) => (),
                            mpsc::TrySendError::Disconnected(_) => break,
                        }
                    }
                }
                Err(err) => {
                    logger.error(err.to_string());
                }
            }
        }

        reader.consume(len);
    }

    tx.send(ThreadState::End).unwrap();
    vad_tx.send(ThreadState::End).unwrap();
    child.kill().expect("failed to kill yt-dlp process");
    handle_vad.join().unwrap();
    handle_whisper.join().unwrap();

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

fn evoke_vad_thread(
    args: Args,
    channel: (SyncSender<ThreadState>, Receiver<ThreadState>),
    rb: (SegmentProducer, F32Consumer),
) -> JoinHandle<()> {
    let logger = Log::new(args.verbose);
    let (tx, rx) = channel;
    let (mut prod, mut cons) = rb;

    thread::spawn(move || {
        let mut vad_state = VadState::new().unwrap();
        let mut rb = LocalRb::<f32, Vec<_>>::new(WINDOW_SIZE_SAMPLES);

        while let Ok(ThreadState::Sync) = rx.recv() {
            if cons.is_empty() {
                logger.error("empty pcm data");
                continue;
            }

            let data = cons.pop_iter().collect::<Vec<f32>>();
            let mut data = resample_to_16k(&data, YOUTUBE_TS_SAMPLE_RATE as f64);

            if rb.len() > 0 {
                data.splice(0..0, rb.pop_iter().collect::<Vec<f32>>());
            }

            let (left, right) = split_audio_data_with_window_size(data);
            if let Some(d) = right {
                d.iter().for_each(|d| {
                    rb.push_overwrite(*d);
                })
            }

            if let Some(data) = left {
                let mut buf = vec![];

                let running_calc = Instant::now();
                data.chunks(WINDOW_SIZE_SAMPLES).for_each(|data| {
                    let _ = vad::vad(&mut vad_state, data.to_vec(), &mut buf);
                });

                logger.verbose(format!(
                    "vad process time: {}s, detect {} segment",
                    running_calc.elapsed().as_secs(),
                    buf.len()
                ));

                if !buf.is_empty() {
                    prod.push_iter(&mut buf.into_iter());
                    if let Err(e) = tx.try_send(ThreadState::Sync) {
                        match e {
                            mpsc::TrySendError::Full(_) => (),
                            mpsc::TrySendError::Disconnected(_) => break,
                        }
                    }
                }
            }
        }
    })
}

fn evoke_whisper_thread(
    args: Args,
    rx: Receiver<ThreadState>,
    mut cons: SegmentConsumer,
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

            cons.pop_iter().for_each(|segment| {
                let config = SpeechConfig::new(args.threads as c_int, Some(&args.lang));
                let mut payload: WhisperPayload = WhisperPayload::new(&segment.data, config);
                let running_calc = Instant::now();

                let segment_time = (streaming_time * 1000.0) as i64;
                streaming_time += segment.duration as f64;

                speech::process(&mut state, &mut payload, &mut |segment, start| {
                    println!(
                        "[{}] {}",
                        util::format_timestamp_to_time(segment_time + start).bright_yellow(),
                        segment
                    );
                });

                logger.verbose(format!(
                    "whisper process time: {}s",
                    running_calc.elapsed().as_secs()
                ));
            });
        }
    })
}
