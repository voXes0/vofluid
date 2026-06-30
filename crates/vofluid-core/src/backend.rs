// backend.rs
use crate::backends::parallel::ParallelBackend;
use crate::backends::single::SingleThread;
use crate::config::SimConfig;
use crate::particle::Particles;

pub enum Backend {
    Single(SingleThread),
    Parallel(ParallelBackend),
    // Gpu(GpuBackend),
}

impl Backend {
    pub fn step(&mut self, particles: &mut Particles, config: &SimConfig) {
        match self {
            Backend::Single(b) => b.step(particles, config),
            Backend::Parallel(b) => b.step(particles, config),
        }
    }
}
