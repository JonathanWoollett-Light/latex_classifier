use std::time::Instant;

#[allow(dead_code)]
// #[cfg(debug_assertions)]
pub fn time(instant: Instant) -> String {
    let millis = instant.elapsed().as_millis() % 1000;
    let seconds = instant.elapsed().as_secs() % 60;
    let mins = (instant.elapsed().as_secs() as f32 / 60f32).floor();
    format!("{:#02}:{:#02}:{:#03}", mins, seconds, millis)
}
