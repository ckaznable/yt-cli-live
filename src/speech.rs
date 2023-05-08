use std::os::raw::c_int;
use whisper_rs::{FullParams, SamplingStrategy, WhisperState};

pub struct SpeechConfig<'a> {
    pub threads: c_int,
    pub lang: Option<&'a str>,
}

impl<'a> Default for SpeechConfig<'a> {
    fn default() -> Self {
        SpeechConfig {
            threads: 4,
            lang: Some("en"),
        }
    }
}

impl<'a> SpeechConfig<'a> {
    pub fn new(threads: c_int, lang: Option<&'a str>) -> SpeechConfig<'a> {
        SpeechConfig { threads, lang }
    }
}

pub struct WhisperPayload<'a> {
    audio_data: &'a [f32],
    config: SpeechConfig<'a>,
}

impl<'a> WhisperPayload<'a> {
    pub fn new<A: AsRef<[f32]>>(audio_data: &'a A, config: SpeechConfig<'a>) -> WhisperPayload<'a> {
        WhisperPayload {
            audio_data: audio_data.as_ref(),
            config,
        }
    }
}

pub fn process<F: FnMut(&str, i64)>(state: &mut WhisperState<'_>, payload: &mut WhisperPayload, f: &mut F) {
    let WhisperPayload {
        audio_data,
        config,
    } = payload;

    let params = get_params(config);

    state.full(params, audio_data).expect("failed to run model");

    // fetch the results
    let num_segments = state
        .full_n_segments()
        .expect("failed to get number of segments");

    let mut last_segment = String::from("");
    for i in 0..num_segments {
        if let (Ok(segment), Ok(start_timestamp)) = (
            state.full_get_segment_text(i),
            state.full_get_segment_t0(i),
        ) {
            if last_segment != segment {
                f(segment.as_ref(), start_timestamp);
            }

            last_segment = segment;
        }
    }
}

fn get_params<'a, 'b>(config: &SpeechConfig<'a>) -> FullParams<'a, 'b> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(config.threads);
    params.set_language(config.lang);
    params.set_suppress_blank(true);
    params.set_no_speech_thold(0.5);
    params.set_audio_ctx(768);

    // disable anything that prints to stdout
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    params
}
