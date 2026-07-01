// backend.rs
use crate::backends::parallel::ParallelBackend;
use crate::backends::single::SingleThread;
use crate::config::SimConfig;
use crate::particle::Particles;
use glam::*;

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
    pub fn resize_grid_particles(&mut self, new_particle_count: usize) {
        match self {
            Backend::Single(b) => b.resize_grid_particles(new_particle_count),
            Backend::Parallel(b) => b.resize_grid_particles(new_particle_count),
        }
    }
    pub fn resize_bounds(&mut self, new_bounds: Vec2) {
        match self {
            Backend::Single(b) => b.resize_grid_bounds(new_bounds),
            Backend::Parallel(b) => b.resize_grid_bounds(new_bounds),
        }
    }
}
