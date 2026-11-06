use crate::config::SimConfig;
use crate::particle::Particles;
use glam::*;

pub struct ParallelBackend;

#[allow(clippy::all)]
impl ParallelBackend {
    pub fn new() -> Self {
        Self
    }
    pub fn step(&mut self, _particles: &mut Particles, _config: &SimConfig) {
        // TODO: rayon-parallelised Clavet SPH step
    }
    pub fn resize_grid_particles(&mut self, _new_particle_count: usize) {}
    pub fn resize_grid_bounds(&mut self, _new_bounds: Vec2) {}
}
