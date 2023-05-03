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