mod app;
#[cfg(not(target_arch = "wasm32"))]
mod gui;

fn main() -> anyhow::Result<()> {
    app::run()?;
    Ok(())
}
