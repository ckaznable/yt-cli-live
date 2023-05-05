pub struct Log {
    enable: bool,
}

impl Log {
    pub fn new(enable: bool) -> Log {
        Log { enable }
    }

    pub fn verbose<S: AsRef<str>>(&self, msg: S) {
        if self.enable {
            println!("[verbose] {}", msg.as_ref());
        }
    }

    pub fn error<S: AsRef<str>>(&self, msg: S) {
        if self.enable {
            println!("[error] {}", msg.as_ref());
        }
    }
}

pub fn format_timestamp_to_time(ms: i64) -> String {
    let minutes = (ms / 60000) % 60;
    let seconds = (ms / 1000) % 60;
    let milliseconds = ms % 1000;
    format!("{:02}:{:02}:{:03}", minutes, seconds, milliseconds)
}
