use crate::config::SimConfig;
use crate::particle::Particles;

pub struct ParallelBackend;

impl ParallelBackend {
    pub fn new() -> Self {
        Self
    }
    pub fn step(&mut self, _particles: &mut Particles, config: &SimConfig) {
        // TODO: rayon-parallelised Clavet SPH step
    }
}
