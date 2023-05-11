use std::time::Duration;

use tract_onnx::{
    prelude::*,
    tract_hir::{internal::DimLike, tract_ndarray::Array},
};

// vad sample rate
const SAMPLE_RATE: f32 = 16000.0;
// 30ms chunk size
pub const WINDOW_SIZE_SAMPLES: usize = (SAMPLE_RATE * 0.03) as usize;

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct VadState {
    model: OnnxModel,
    h: Tensor,
    c: Tensor,
}

impl VadState {
    pub fn new() -> TractResult<VadState> {
        let model = onnx()
            .model_for_path("models/silero_vad.onnx")?
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

        Ok(VadState {
            model,
            h: Tensor::zero::<f32>(&[2, 1, 64])?,
            c: Tensor::zero::<f32>(&[2, 1, 64])?,
        })
    }
}

pub struct VadOutput {
    segment: Vec<VadSegment>,
    duration: Duration,
}

pub struct VadSegment {
    start: u64,
    end: u64,
    data: Vec<f32>,
}

pub fn vad(state: &mut VadState, audio_data: Vec<f32>) -> TractResult<bool> {
    let pcm = Array::from_shape_vec((1, audio_data.len()), audio_data).unwrap();
    let pcm = pcm.into_arc_tensor();
    let samples = pcm.shape()[1];

    let offset = WINDOW_SIZE_SAMPLES;
    let chunk_len = (samples - offset).min(WINDOW_SIZE_SAMPLES);

    let mut x = Tensor::zero::<f32>(&[1, WINDOW_SIZE_SAMPLES])?;
    x.assign_slice(0..chunk_len, &pcm, offset..offset + chunk_len, 1)?;

    let mut outputs = state.model.run(tvec!(x, state.h.clone(), state.c.clone()))?;
    state.c = outputs.remove(2).into_tensor();
    state.h = outputs.remove(1).into_tensor();

    let speech_prob = outputs[0].as_slice::<f32>()?[1];

    Ok(speech_prob > 0.5)
}

pub fn split_audio_data_with_window_size(audio_data: Vec<f32>) -> (Option<Vec<f32>>, Option<Vec<f32>>) {
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
