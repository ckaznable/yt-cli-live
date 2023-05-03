use std::io::Cursor;

use mpeg2ts::{
    es::StreamType,
    ts::{ReadTsPacket, TsPacketReader, TsPayload},
};
use symphonia::core::{
    audio::AudioBuffer,
    codecs::{DecoderOptions, CODEC_TYPE_NULL},
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

pub fn get_audio_data(data: &[u8]) -> Result<Vec<f32>, &'static str> {
    get_mono_f32(extract_ts_audio(data))
}

fn get_mono_f32(raw: Vec<u8>) -> Result<Vec<f32>, &'static str> {
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
        .map_err(|_| "unsupported format")?;

    // Get the instantiated format reader.
    let mut format = probed.format;

    // Find the first audio track with a known (decodeable) codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("no supported audio tracks")?;

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .map_err(|_| "unsupported codec")?;

    // Store the track identifier, it will be used to filter packets.
    let track_id = track.id;

    let mut data: Vec<f32> = vec![];

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
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => (),
            _ => {
                break;
            }
        }
    }

    Ok(data)
}

fn extract_ts_audio(raw: &[u8]) -> Vec<u8> {
    let cursor = Cursor::new(raw);
    let mut reader = TsPacketReader::new(cursor);

    let mut data: Vec<u8> = vec![];
    let mut audio_pid: u16 = 0;

    while let Ok(Some(packet)) = reader.read_ts_packet() {
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

    data
}
