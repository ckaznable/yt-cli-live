use owo_colors::OwoColorize;

#[derive(Clone, Default)]
pub struct Log {
    enable: bool,
}

impl Log {
    pub fn new(enable: bool) -> Log {
        Log { enable }
    }

    pub fn verbose<S: AsRef<str>>(&self, msg: S) {
        if self.enable {
            println!("{} {}", "[verbose]".green(), msg.as_ref().green());
        }
    }

    pub fn error<S: AsRef<str>>(&self, msg: S) {
        if self.enable {
            println!("{} {}", "[error]".red(), msg.as_ref().red().bold());
        }
    }
}

pub fn format_timestamp_to_time(ms: i64) -> String {
    let minutes = (ms / 60000) % 60;
    let seconds = (ms / 1000) % 60;
    let milliseconds = ms % 1000;
    format!("{:02}:{:02}:{:03}", minutes, seconds, milliseconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_timestamp_to_time() {
        let time = 1000 * 60 * 60 * 3 + 1000 * 60 * 3 + 3;
        let result = format_timestamp_to_time(time);
        assert_eq!(result, "03:00:003");
    }
}
