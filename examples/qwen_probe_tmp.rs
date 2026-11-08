//! Temporary probe: run the Qwen backend against an arbitrary checkpoint dir.

use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let wav = PathBuf::from(args.next().expect("wav"));
    let dir = PathBuf::from(args.next().expect("model dir"));
    let lang = args.next().unwrap_or_else(|| "ru".to_string());

    let samples = lc_rs::audio::read_samples(&wav).expect("wav");
    let mut model = lc_rs::ml::qwen::Qwen::load(&dir).expect("load");
    model.set_language(lc_rs::ml::qwen::language_name(&lang));
    let result = model.transcribe(&samples).expect("transcribe");
    println!("TEXT: {}", result.text);
    println!("LANG: {:?}", result.language);
}
