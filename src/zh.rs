use opencc_rust::{DefaultConfig, OpenCC};

pub struct ZHTransformer {
    convert: OpenCC,
}

impl ZHTransformer {
    pub fn from(s: &str) -> Result<Self, &'static str> {
        let s = s.to_string().to_lowercase();

        if s == "zh" {
            return Err("auto detection chinese language code");
        }

        if !s.starts_with("zh") {
            return Err("is not chinese language code");
        }

        let config = if matches!(s.as_str(), "zh_tw" | "zh-tw") {
            DefaultConfig::S2TW
        } else if matches!(s.as_str(), "zh_hk" | "zh-hk") {
            DefaultConfig::S2HK
        } else {
            DefaultConfig::T2S
        };

        Ok(Self {
            convert: OpenCC::new(config)?,
        })
    }

    pub fn convert<S: AsRef<str>>(&self, s: S) -> String {
        self.convert.convert(s)
    }
}
