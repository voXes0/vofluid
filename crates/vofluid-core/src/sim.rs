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
        let diff = (self.config.particle_count) as i32 - (self.particles.positions.len()) as i32;
        if diff > 0 {
            for _ in 0..diff {
                self.particles.push(Vec2::ZERO, Vec2::ZERO);
            }
        } else if diff < 0 {
            for i in (0..diff.unsigned_abs()).rev() {
                self.particles.swap_remove(i as usize);
            }
        }
        self.backend
            .resize_grid_particles(self.config.particle_count);

        for pos in self.particles.positions.iter_mut() {
            let x = rand::random_range(-self.config.bounds.x..self.config.bounds.x);
            let y = rand::random_range(-self.config.bounds.y..self.config.bounds.y);
            *pos = vec2(x, y);
        }
    }

    pub fn ensure_particle_count(&mut self) {
        let diff = (self.config.particle_count) as i32 - (self.particles.positions.len()) as i32;
        if diff > 0 {
            for _ in 0..diff {
                self.particles.push(
                    Vec2 {
                        x: rand::random_range(-self.config.bounds.x..self.config.bounds.x),
                        y: rand::random_range(-self.config.bounds.y..self.config.bounds.y),
                    },
                    Vec2::ZERO,
                );
            }
        } else if diff < 0 {
            for i in (0..diff.unsigned_abs()).rev() {
                self.particles.swap_remove(i as usize);
            }
        }
        self.backend
            .resize_grid_particles(self.config.particle_count);
    }

    pub fn resize_bounds(&mut self, new_bounds: Vec2) {
        self.backend.resize_bounds(new_bounds);
    }
}
