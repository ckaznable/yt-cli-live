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

pub fn process<F: Fn(String, i64, i64)>(
    state: &mut WhisperState,
    audio_data: &[f32],
    config: &SpeechConfig<'_>,
    f: F,
) {
    let params = get_params(config);

    state.full(params, audio_data).expect("failed to run model");

    // fetch the results
    let num_segments = state
        .full_n_segments()
        .expect("failed to get number of segments");
    for i in 0..num_segments {
        if let (Ok(segment), Ok(start_timestamp), Ok(end_timestamp)) = (
            state.full_get_segment_text(i),
            state.full_get_segment_t0(i),
            state.full_get_segment_t1(i),
        ) {
            f(segment, start_timestamp, end_timestamp);
        }
    }
}

fn get_params<'a, 'b>(config: &SpeechConfig<'a>) -> FullParams<'a, 'b> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(config.threads);
    params.set_language(config.lang);

    // disable anything that prints to stdout
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    params
}
