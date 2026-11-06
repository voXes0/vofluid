//! Single-threaded solver: the substep loop and the neighbour grid, generic
//! over the SPH interaction that runs inside it (`Relaxation`).
//!
//! The solver is position based. Per step the particles get gravity, then per
//! substep: integrate positions, re-sort the grid, let the `Relaxation` push
//! positions apart, clamp to the bounds, and rebuild velocities from the
//! displacement. Swapping the SPH algorithm means implementing `Relaxation`;
//! the loop, the grid and the profiling spans stay as they are.

mod double_density;
mod probe;
mod simd;

pub use double_density::DoubleDensity;

use crate::config::SimConfig;
use crate::grid::UniformGrid;
use crate::particle::{Particles, Positions};
use glam::Vec2;

/// Grid cells per interaction radius. 2 gives a 5x5 cell stencil, which
/// covers 2x the interaction circle; finer cells cut that waste but add
/// stencil rows (more per-row overhead), coarser ones add waste.
const CELLS_PER_RADIUS: f32 = 2.0;

/// The neighbour interaction run once per substep, after the grid is rebuilt
/// from the integrated positions.
///
/// An implementation reads positions from the grid's sorted arrays and
/// adds its displacements to `next_positions`, which starts out as a copy
/// of the current positions and is indexed by original particle index
/// (`UniformGrid::particle_index` maps slots to it). It keeps any per-particle
/// scratch itself, in slot order, and must size it `SORTED_PAD` slots beyond
/// the particle count if its kernels load full-width vectors.
pub trait Relaxation {
    fn new(config: &SimConfig) -> Self;

    fn resize_particles(&mut self, particle_count: usize);

    /// Applies one substep of length `dt`.
    fn relax(
        &mut self,
        grid: &UniformGrid,
        next_positions: &mut [Vec2],
        config: &SimConfig,
        dt: f32,
    );
}

pub struct SingleThread<R: Relaxation = DoubleDensity> {
    grid: UniformGrid,
    relaxation: R,
}

impl SingleThread {
    /// A solver using the default interaction, `DoubleDensity`.
    pub fn new(config: &SimConfig) -> Self {
        Self::with_relaxation(config)
    }
}

impl<R: Relaxation> SingleThread<R> {
    /// A solver using the interaction `R`.
    pub fn with_relaxation(config: &SimConfig) -> Self {
        Self {
            grid: UniformGrid::new(
                config.bounds,
                config.interaction_radius / CELLS_PER_RADIUS,
                config.particle_count,
            ),
            relaxation: R::new(config),
        }
    }

    // Whole-step timing is `Simulation::step`'s job (it brackets every
    // backend the same way), so there is no span for it here.
    pub fn step(&mut self, particles: &mut Particles, config: &SimConfig) {
        if config.substeps < 1 {
            return;
        }

        let sub_dt = config.dt / config.substeps as f32;

        particles.apply_uniform_acceleration(config.gravity, config.dt);
        for _ in 0..config.substeps {
            particles.save_prev_positions();
            particles.integrate_positions(sub_dt);
            self.grid.rebuild(&particles.positions);
            separate_coincident(particles, &self.grid, config.interaction_radius * 1e-3);
            particles.begin_scatter_pass();
            self.relaxation
                .relax(&self.grid, &mut particles.next_positions, config, sub_dt);
            particles.end_scatter_pass();
            clamp_boundary(&mut particles.positions, config.bounds);
            particles.reconstruct_velocities(sub_dt);
        }
    }

    pub fn resize_grid_bounds(&mut self, new_bounds: Vec2) {
        self.grid.resize_bounds(new_bounds);
    }

    pub fn resize_grid_particles(&mut self, new_particle_count: usize) {
        self.grid.resize_particles(new_particle_count);
        self.relaxation.resize_particles(new_particle_count);
    }
}

#[inline]
fn hash01(i: u32, salt: u32) -> f32 {
    let mut h = i.wrapping_mul(0x9E37_79B9) ^ salt;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    (h >> 8) as f32 / (1u32 << 24) as f32 // [0, 1)
}

fn separate_coincident(particles: &mut Particles, grid: &UniformGrid, min_dist: f32) {
    crate::prof_span!("single::separate_coincident");

    let (sx, sy) = grid.sorted_xy();
    let particle_index = grid.particle_index();
    for (_, _, slots) in grid.occupied_cells() {
        for s in slots.clone() {
            if (slots.start..s).any(|t| sx[t] == sx[s] && sy[t] == sy[s]) {
                let i = particle_index[s];
                let a = hash01(i as u32, 0x51ED) * std::f32::consts::TAU;
                let nudge = glam::vec2(a.cos(), a.sin()) * min_dist;
                particles.positions[i] += nudge;
                particles.prevs[i] += nudge; // same shift in prevs, so no velocity kick
            }
        }
    }
}

fn clamp_boundary(positions: &mut Positions, bounds: Vec2) {
    crate::prof_span!("single::clamp_boundary");

    for pos in positions.iter_mut() {
        *pos = pos.clamp(-bounds, bounds);
    }
}
