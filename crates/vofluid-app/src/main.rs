mod app;
#[cfg(not(target_arch = "wasm32"))]
mod gui;
#[cfg(all(not(target_arch = "wasm32"), feature = "profile"))]
mod profiling_export;
#[cfg(all(not(target_arch = "wasm32"), feature = "profile"))]
mod profiling_gui;
#[cfg(all(not(target_arch = "wasm32"), feature = "profile"))]
mod profiling_sink;

fn main() -> anyhow::Result<()> {
    app::run()?;
    Ok(())
}
