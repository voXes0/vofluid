use glam::{Vec2, vec2};

use crate::{backend::Backend, config::SimConfig, particle::Particles};

/// Top-level simulation container.
pub struct Simulation {
    pub particles: Particles,
    pub config: SimConfig,
    backend: Backend,
}

impl Simulation {
    pub fn new(backend: Backend, config: SimConfig) -> Self {
        let particles = Particles::new(config.particle_count, 128);
        let mut sim = Self {
            particles,
            config,
            backend,
        };
        sim.initialize_particles();
        sim
    }

    pub fn step(&mut self) {
        self.backend.step(&mut self.particles, &self.config);
    }

    pub fn set_backend(&mut self, backend: Backend) {
        self.backend = backend;
    }

    pub fn initialize_particles(&mut self) {
        self.particles.positions.fill(Vec2::ZERO);

        for pos in self.particles.positions.iter_mut() {
            let x = rand::random_range(-self.config.bounds[0]..self.config.bounds[0]);
            let y = rand::random_range(-self.config.bounds[1]..self.config.bounds[1]);
            *pos = vec2(x, y);
        }
    }
}
