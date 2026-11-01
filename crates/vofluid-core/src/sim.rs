use glam::{Vec2, vec2};

use crate::{backend::Backend, config::SimConfig, particle::Particles};

pub struct Simulation {
    pub particles: Particles,
    pub config: SimConfig,
    backend: Backend,
    accumulator: f32,
    /// The config the profiler last saw. A difference at the next step starts
    /// a new profiling epoch (see `profiling::begin_epoch`).
    #[cfg(feature = "profile")]
    profiled_config: Option<SimConfig>,
}

const MAX_FRAME_TIME_SECS: f32 = 0.25;

#[derive(Clone, Copy, Debug, Default)]
pub struct AdvanceReport {
    /// How many internal `step()` calls ran to catch simulated time up to
    /// the requested `elapsed_secs`. Usually 0 or 1; can be >1 after a
    /// slow frame, 0 on a frame faster than `config.dt`.
    pub steps_run: u32,
    /// Total wall-clock time spent inside `step()` across all steps this
    /// call, in seconds. `0.0` if `steps_run` is `0`. Divide by `steps_run`
    /// for an average per-step cost.
    pub step_time_secs: f32,
}

impl AdvanceReport {
    /// Average per-step wall-clock cost, in seconds. `None` if no steps ran
    /// this call (nothing to average).
    pub fn avg_step_time_secs(&self) -> Option<f32> {
        (self.steps_run > 0).then(|| self.step_time_secs / self.steps_run as f32)
    }
}

impl Simulation {
    pub fn new(backend: Backend, config: SimConfig) -> Self {
        let particles = Particles::new(config.particle_count, 128);
        let mut sim = Self {
            particles,
            config,
            backend,
            accumulator: 0.0,
            #[cfg(feature = "profile")]
            profiled_config: None,
        };
        sim.initialize_particles();
        sim
    }

    pub fn step(&mut self) {
        #[cfg(feature = "profile")]
        let _step = self.begin_profiled_step();
        self.backend.step(&mut self.particles, &self.config);
    }

    /// Starts a profiling epoch if the config changed since the previous
    /// step, then returns the guard that times this step as a whole.
    #[cfg(feature = "profile")]
    fn begin_profiled_step(&mut self) -> crate::profiling::StepGuard {
        if self.profiled_config.as_ref() != Some(&self.config) {
            crate::profiling::begin_epoch(&self.config);
            self.profiled_config = Some(self.config.clone());
        }
        crate::profiling::StepGuard::new()
    }

    pub fn advance(&mut self, elapsed_secs: f32) -> AdvanceReport {
        let dt = self.config.dt;
        if dt <= 0.0 {
            return AdvanceReport::default();
        }

        let elapsed_secs = elapsed_secs.clamp(0.0, MAX_FRAME_TIME_SECS);
        self.accumulator += elapsed_secs;

        let steps = (self.accumulator / dt).floor();
        self.accumulator -= steps * dt;
        let steps_run = steps as u32;

        let start = std::time::Instant::now();
        for _ in 0..steps_run {
            self.step();
        }
        let step_time_secs = start.elapsed().as_secs_f32();

        AdvanceReport {
            steps_run,
            step_time_secs,
        }
    }

    pub fn set_backend(&mut self, backend: Backend) {
        self.backend = backend;
    }

    pub fn initialize_particles(&mut self) {
        self.ensure_particle_count();

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
