use std::{fmt::Display, io::Cursor};

use mpeg2ts::{
    es::StreamType,
    ts::{ReadTsPacket, TsPacketReader, TsPayload},
};
use rubato::{InterpolationParameters, InterpolationType, Resampler, SincFixedIn, WindowFunction};
use symphonia::core::{
    audio::AudioBuffer,
    codecs::{DecoderOptions, CODEC_TYPE_NULL},
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

pub const YOUTUBE_TS_SAMPLE_RATE: u16 = 22050;

#[derive(Debug)]
pub enum Error {
    Format,
    Decoder,
    Track,
    Empty,
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Error::*;

        write!(
            f,
            "{}",
            match self {
                Format => "unsupported format",
                Decoder => "no supported audio tracks",
                Track => "unsupported codec",
                Empty => "empty audio data",
            }
        )
    }
}

pub fn get_audio_data(data: &[u8]) -> Result<(Vec<f32>, f64), Error> {
    let ts_audio = extract_ts_audio(data);

    if ts_audio.is_empty() {
        Err(Error::Empty)
    } else {
        get_mono_f32(ts_audio)
    }
}

fn get_mono_f32(raw: Vec<u8>) -> Result<(Vec<f32>, f64), Error> {
    let src = Cursor::new(raw);
    // Create the media source stream.
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    // Create a probe hint using the file's extension. [Optional]
    let mut hint = Hint::new();
    hint.with_extension("aac");

    // Use the default options for metadata and format readers.
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();
    let dec_opts: DecoderOptions = Default::default();

    // Probe the media source.
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .map_err(|_| Error::Format)?;

    // Get the instantiated format reader.
    let mut format = probed.format;

    // Find the first audio track with a known (decodeable) codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or(Error::Track)?;

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .map_err(|_| Error::Decoder)?;

    // Store the track identifier, it will be used to filter packets.
    let track_id = track.id;

    let mut data: Vec<f32> = vec![];
    let mut dur = 0.0f64;
    let mut rate = 0.0f64;
    let mut planes_num = 1.0f64;

    // The decode loop.
    loop {
        // Get the next packet from the media format.
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::ResetRequired) => {
                continue;
            }
            Err(_) => {
                break;
            }
        };

        // Consume any new metadata that has been read since the last packet.
        while !format.metadata().is_latest() {
            // Pop the old head of the metadata queue.
            format.metadata().pop();
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples.
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let mut buf = AudioBuffer::<f32>::new(packet.dur, *audio_buf.spec());
                audio_buf.convert(&mut buf);

                let planes = buf.planes();
                let planes = planes.planes();
                data.extend_from_slice(planes[0]);
                dur += packet.dur as f64;

                if rate == 0.0 {
                    rate = audio_buf.spec().rate as f64;
                }

                if planes_num == 1.0 {
                    planes_num = planes.len() as f64;
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => (),
            _ => {
                break;
            }
        }
    }

    Ok((data, dur / (rate * planes_num)))
}

fn extract_ts_audio(raw: &[u8]) -> Vec<u8> {
    let cursor = Cursor::new(raw);
    let mut reader = TsPacketReader::new(cursor);

    let mut data: Vec<u8> = vec![];
    let mut audio_pid: u16 = 0;

    loop {
        match reader.read_ts_packet() {
            Ok(Some(packet)) => {
                use TsPayload::*;

                let pid = packet.header.pid.as_u16();
                let is_audio_pid = pid == audio_pid;

                if let Some(payload) = packet.payload {
                    match payload {
                        Pmt(pmt) => {
                            if let Some(el) = pmt
                                .table
                                .iter()
                                .find(|el| el.stream_type == StreamType::AdtsAac)
                            {
                                audio_pid = el.elementary_pid.as_u16();
                            }
                        }
                        Pes(pes) => {
                            if pes.header.stream_id.is_audio() && is_audio_pid {
                                data.extend_from_slice(&pes.data);
                            }
                        }
                        Raw(bytes) => {
                            if is_audio_pid {
                                data.extend_from_slice(&bytes);
                            }
                        }
                        _ => (),
                    }
                }
            }
            Ok(None) => break,
            _ => (),
        }
    }

    data
}

pub fn resample_to_16k(input: &[f32], input_sample_rate: f64) -> Vec<f32> {
    let params = InterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: InterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler =
        SincFixedIn::<f32>::new(16000. / input_sample_rate, 2.0, params, input.len(), 1).unwrap();

    let waves_in = vec![input.to_vec()];
    let mut output = resampler.process(&waves_in, None).unwrap();
    output.remove(0)
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Read};

    use super::extract_ts_audio;

    fn read_file_to_bytes(path: &str) -> Result<Vec<u8>, std::io::Error> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Ok(buffer)
    }

    #[test]
    fn extract_ts_must_work() {
        let file_path = "tests/test.ts";
        if let Ok(bytes) = read_file_to_bytes(file_path) {
            let data = extract_ts_audio(bytes.as_slice());
            assert!(!data.is_empty());
            assert!(data.len() < bytes.len());
        }
    }
}
