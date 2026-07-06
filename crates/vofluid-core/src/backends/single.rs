use crate::config::SimConfig;
use crate::grid::UniformGrid;
use crate::neighbor::NeighbourCache;
use crate::particle::{Densities, Particles, Positions};
use glam::*;

pub struct SingleThread {
    grid: UniformGrid,
    neighbour_cache: NeighbourCache,
    /// Scratch buffer: densities permuted into sorted-grid order after pass
    /// 1 of `double_density`, so pass 2 can read neighbour densities
    /// sequentially via `CachedNeighbour::n_slot`. Reused across steps to
    /// avoid reallocating every substep.
    sorted_densities: Vec<Vec2>,
}
impl SingleThread {
    pub fn new(config: &SimConfig) -> Self {
        #[cfg(feature = "profile")]
        tracy_client::Client::start();

        Self {
            grid: UniformGrid::new(
                config.bounds,
                config.interaction_radius * 0.5,
                config.particle_count,
            ),
            neighbour_cache: NeighbourCache::with_capacity(config.particle_count, 64),
            sorted_densities: Vec::with_capacity(config.particle_count),
        }
    }
    // Conventions: p is the iteration particle, n is the neighbor particle of the iteration particle
    pub fn step(&mut self, particles: &mut Particles, config: &SimConfig) {
        #[cfg(feature = "profile")]
        let _ = tracy_client::span!();
        let dt = config.dt;
        let sub_dt = dt / config.substeps as f32;

        particles.apply_uniform_acceleration(config.gravity, dt);
        for _ in 0..config.substeps {
            particles.save_prev_positions();
            particles.integrate_positions(sub_dt);
            self.grid.rebuild(&particles.positions);
            particles.begin_scatter_pass();
            double_density(
                &particles.positions,
                &mut particles.next_positions,
                &mut particles.densities,
                config,
                &self.grid,
                sub_dt,
                &mut self.neighbour_cache,
                &mut self.sorted_densities,
            );
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
        self.sorted_densities
            .reserve(new_particle_count.saturating_sub(self.sorted_densities.capacity()));
    }
}

fn double_density(
    positions: &Positions,
    next_positions: &mut Positions,
    densities: &mut Densities,
    config: &SimConfig,
    grid: &UniformGrid,
    dt: f32,
    cache: &mut NeighbourCache,
    sorted_densities: &mut Vec<Vec2>,
) {
    #[cfg(feature = "profile")]
    let _ = tracy_client::span!();

    let dt_sq = dt * dt;

    let i_radius = config.interaction_radius;
    let i_radius_inv = 1.0 / i_radius;
    let i_radius_sq = i_radius * i_radius;
    let min_dist = i_radius * 1e-3;

    densities.reset();
    cache.clear();

    for ((p_indice, p_pos), p_densities) in positions.iter().enumerate().zip(densities.iter_mut()) {
        grid.query_neighbours_sorted(*p_pos, i_radius, |n_slot, n_indice, n_pos| {
            #[cfg(feature = "profile")]
            let _ = tracy_client::span!("Neighbor query closure");

            if n_indice == p_indice {
                return;
            }
            let offset = n_pos - p_pos;

            let distance_sq = offset.length_squared();
            if distance_sq > i_radius_sq {
                return;
            }

            let distance = distance_sq.sqrt().max(min_dist);
            let distance_inv = distance.recip();
            let inf = 1.0 - distance * i_radius_inv;
            let inf_sq = inf * inf;

            p_densities.x += inf_sq;
            p_densities.y += inf_sq * inf;

            let dir_p_to_n = offset * distance_inv;

            cache.push(n_slot, inf, dir_p_to_n);
        });

        cache.finish_particle();
    }

    // Permute the finished densities into sorted-grid order so pass 2 can
    // read neighbour densities sequentially via `n_slot` instead of
    // scatter-reading `densities[n_indice]` at arbitrary original indices —
    // same trick as `sorted_positions`, applied to the other hot lookup.
    let particle_index = grid.particle_index();
    sorted_densities.clear();
    sorted_densities.extend(particle_index.iter().map(|&orig| densities[orig]));

    for ((p_indice, p_densities), next_p_pos) in
        densities.iter().enumerate().zip(next_positions.iter_mut())
    {
        #[cfg(feature = "profile")]
        let _ = tracy_client::span!("Second pass");

        let pressure_p = config.stiffness * (p_densities.x - config.rest_density);
        let pressure_near_p = config.stiffness_near * p_densities.y;

        let mut disp_p = Vec2::ZERO;

        for cached in cache.neighbours_of(p_indice) {
            let n_slot = cached.n_slot as usize;
            let n_densities = sorted_densities[n_slot];
            let pressure_n = config.stiffness * (n_densities.x - config.rest_density);
            let pressure_near_n = config.stiffness_near * n_densities.y;

            let sum_pressure = pressure_p + pressure_n;
            let sum_pressure_near = pressure_near_p + pressure_near_n;
            let inf = cached.inf;

            let displacement =
                dt_sq * (sum_pressure * inf + sum_pressure_near * inf * inf) * (-cached.dir_p_to_n);

            disp_p += displacement;
        }

        *next_p_pos += disp_p;
    }
}

fn clamp_boundary(positions: &mut Positions, bounds: Vec2) {
    for pos in positions.iter_mut() {
        *pos = pos.clamp(vec2(-bounds.x, -bounds.y), vec2(bounds.x, bounds.y));
    }
}
