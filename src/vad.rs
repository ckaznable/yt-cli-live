use std::{io::Cursor, mem::MaybeUninit, rc::Rc};

use ringbuf::{Consumer, LocalRb, Producer};
use tract_onnx::{
    prelude::{tract_itertools::Itertools, *},
    tract_hir::tract_ndarray::Array,
};

// vad sample rate
pub const SAMPLE_RATE: f32 = 16000.0;
// 30ms chunk size
pub const WINDOW_SIZE_SAMPLES: usize = (SAMPLE_RATE * 0.03) as usize;

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;
type F32RingBufProducer = Producer<f32, Rc<LocalRb<f32, Vec<MaybeUninit<f32>>>>>;
type F32RingBufConsumer = Consumer<f32, Rc<LocalRb<f32, Vec<MaybeUninit<f32>>>>>;

pub struct VadState {
    /// vad model required parameter
    model: OnnxModel,
    h: Tensor,
    c: Tensor,

    /// vad detection state
    triggered: bool,
    speech_start_ts: u32,
    speech_end_ts: u32,
    window_count: u32,

    /// 15s audio data ring buffer
    rb_prod: F32RingBufProducer,
    rb_cons: F32RingBufConsumer,
}

impl VadState {
    pub fn new() -> TractResult<VadState> {
        let bytes = include_bytes!("../models/silero_vad.onnx");
        let mut cursor = Cursor::new(bytes);
        let model = onnx()
            .model_for_read(&mut cursor)?
            .with_input_names(["input", "h0", "c0"])?
            .with_output_names(["output", "hn", "cn"])?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, WINDOW_SIZE_SAMPLES)),
            )?
            .with_input_fact(
                1,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(2, 1, 64)),
            )?
            .with_input_fact(
                2,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(2, 1, 64)),
            )?
            .into_optimized()?
            .into_runnable()?;

        // 15s audio data ring buffer
        let (prod, cons) = LocalRb::<f32, Vec<_>>::new(WINDOW_SIZE_SAMPLES * 500).split();

        Ok(VadState {
            model,
            h: Tensor::zero::<f32>(&[2, 1, 64])?,
            c: Tensor::zero::<f32>(&[2, 1, 64])?,
            triggered: false,
            speech_start_ts: 0,
            speech_end_ts: 0,
            window_count: 0,
            rb_prod: prod,
            rb_cons: cons,
        })
    }

    fn init(&mut self) {
        self.window_count = 0;
        self.speech_end_ts = 0;
        self.speech_start_ts = 0;
    }
}

#[derive(Default)]
pub struct VadSegment {
    pub data: Vec<f32>,
    pub duration: f32,
}

pub fn vad(
    state: &mut VadState,
    audio_data: Vec<f32>,
    buf: &mut Vec<VadSegment>,
) -> TractResult<()> {
    state.rb_prod.push_slice(&audio_data);

    if state.rb_prod.is_full() {
        state.init();
        buf.push(VadSegment {
            data: state.rb_cons.pop_iter().collect_vec(),
            duration: 15.,
        });
        return Ok(());
    }

    let pcm = Array::from_shape_vec((1, audio_data.len()), audio_data).unwrap();
    let pcm = pcm.into_arc_tensor();
    let samples = pcm.shape()[1];
    let chunk_len = samples.min(WINDOW_SIZE_SAMPLES);

    let mut x = Tensor::zero::<f32>(&[1, WINDOW_SIZE_SAMPLES])?;
    x.assign_slice(0..chunk_len, &pcm, 0..chunk_len, 1)?;

    let mut outputs = state
        .model
        .run(tvec!(x, state.h.clone(), state.c.clone()))?;
    state.c = outputs.remove(2).into_tensor();
    state.h = outputs.remove(1).into_tensor();

    let speech_prob = outputs[0].as_slice::<f32>()?[1];

    const MIN_SILENCE_DURATION_MS: u32 = 800;
    const MIN_SPEECH_DURATION_MS: u32 = 300;
    const THRESHOLD: f32 = 0.5;
    const NEG_THRESHOLD: f32 = 0.1;
    const MIN_SILENCE_SAMPLES: u32 = MIN_SILENCE_DURATION_MS * SAMPLE_RATE as u32 / 1000;
    const MIN_SPEECH_SAMPLES: u32 = MIN_SPEECH_DURATION_MS * SAMPLE_RATE as u32 / 1000;

    let speech_sample_offset = state.window_count * WINDOW_SIZE_SAMPLES as u32;
    state.window_count += 1;

    if speech_prob >= THRESHOLD && state.speech_end_ts != 0 {
        state.speech_end_ts = 0;
    }

    if speech_prob >= THRESHOLD && !state.triggered {
        state.triggered = true;
        state.speech_start_ts = speech_sample_offset;
    } else if speech_prob < NEG_THRESHOLD && state.triggered {
        if state.speech_end_ts == 0 {
            state.speech_end_ts = speech_sample_offset;
        }

        if speech_sample_offset - state.speech_end_ts >= MIN_SILENCE_SAMPLES {
            if state.speech_end_ts - state.speech_start_ts > MIN_SPEECH_SAMPLES {
                state.init();
                buf.push(VadSegment {
                    data: state.rb_cons.pop_iter().collect_vec(),
                    duration: speech_sample_offset as f32 / SAMPLE_RATE,
                });
            }

            state.triggered = false
        }
    }

    Ok(())
}

pub fn split_audio_data_with_window_size(
    audio_data: Vec<f32>,
) -> (Option<Vec<f32>>, Option<Vec<f32>>) {
    let len = audio_data.len();

    if len < WINDOW_SIZE_SAMPLES {
        (None, Some(audio_data))
    } else if len % WINDOW_SIZE_SAMPLES == 0 {
        (Some(audio_data), None)
    } else {
        let chunk_num = len / WINDOW_SIZE_SAMPLES;
        let last_offset = chunk_num * WINDOW_SIZE_SAMPLES;
        let (left, right) = audio_data.split_at(last_offset);

        (Some(left.to_vec()), Some(right.to_vec()))
    }
}
