use crate::config::SimConfig;
use crate::particle::Particles;
use glam::*;

pub struct ParallelBackend;

impl ParallelBackend {
    pub fn new() -> Self {
        Self
    }
    pub fn step(&mut self, _particles: &mut Particles, config: &SimConfig) {
        // TODO: rayon-parallelised Clavet SPH step
    }
    pub fn resize_grid_particles(&mut self, new_particle_count: usize) {}
    pub fn resize_grid_bounds(&mut self, new_bounds: Vec2) {}
}
