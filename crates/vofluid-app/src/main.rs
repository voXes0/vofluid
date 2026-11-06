mod app;
mod gui;
#[cfg(feature = "profile")]
mod profiling_export;
#[cfg(feature = "profile")]
mod profiling_gui;
#[cfg(feature = "profile")]
mod profiling_sink;

fn main() -> anyhow::Result<()> {
    app::run()?;
    Ok(())
}
