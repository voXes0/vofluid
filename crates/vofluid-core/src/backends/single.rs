use crate::config::SimConfig;
use crate::grid::UniformGrid;
use crate::particle::{Densities, Particles, Positions};
use glam::*;

pub struct SingleThread {
    grid: UniformGrid,
}
impl SingleThread {
    pub fn new(config: &SimConfig) -> Self {
        Self {
            grid: UniformGrid::new(
                config.bounds,
                config.interaction_radius,
                config.particle_count,
            ),
        }
    }
    // Conventions: p is the iteration particle, n is the neighbor particle of the iteration particle
    pub fn step(&mut self, particles: &mut Particles, config: &SimConfig) {
        let dt = config.dt / config.substeps as f32;

        for _ in 0..config.substeps {
            particles.apply_uniform_acceleration(config.gravity, dt);
            particles.save_prev_positions();
            particles.integrate_positions(dt);
            self.grid.rebuild(&particles.positions);
            particles.begin_scatter_pass();
            double_density(
                &particles.positions,
                &mut particles.next_positions,
                &mut particles.densities,
                config,
                &self.grid,
                dt,
            );
            particles.end_scatter_pass();
            clamp_boundary(&mut particles.positions, config);
            particles.reconstruct_velocities(dt);
        }
    }

    pub fn resize_grid_bounds(&mut self, new_bounds: Vec2) {
        self.grid.resize_bounds(new_bounds);
    }
    pub fn resize_grid_particles(&mut self, new_particle_count: usize) {
        self.grid.resize_particles(new_particle_count);
    }
}

fn double_density(
    positions: &Positions,
    next_positions: &mut Positions,
    densities: &mut Densities,
    config: &SimConfig,
    grid: &UniformGrid,
    dt: f32,
) {
    let dt_sq = dt * dt;
    let i_radius = config.interaction_radius;
    let i_radius_inv = 1.0 / i_radius;
    let i_radius_sq = i_radius * i_radius;
    let min_dist = i_radius * 1e-4;

    densities.reset();

    for ((p_indice, p_pos), p_densities) in positions.iter().enumerate().zip(densities.iter_mut()) {
        grid.query_neighbours(*p_pos, |n_indice| {
            if n_indice == p_indice {
                return;
            }
            let n_pos = positions[n_indice];
            let offset = n_pos - p_pos;

            let distance_sq = offset.length_squared();
            if distance_sq > i_radius_sq {
                return;
            }

            let distance = distance_sq.sqrt();
            let inf = 1.0 - distance * i_radius_inv;
            let inf_sq = inf * inf;

            p_densities.x += inf_sq;
            p_densities.y += inf_sq * inf;
        });
    }

    for (((p_indice, p_pos), p_densities), next_p_pos) in positions
        .iter()
        .enumerate()
        .zip(densities.iter())
        .zip(next_positions.iter_mut())
    {
        let pressure_p = config.stiffness * (p_densities.x - config.rest_density);
        let pressure_near_p = config.stiffness_near * p_densities.y;

        let mut disp_p = Vec2::ZERO;

        grid.query_neighbours(*p_pos, |n_indice| {
            if n_indice == p_indice {
                return;
            }

            let n_pos = positions[n_indice];
            let offset = n_pos - p_pos; // p -> n

            let distance_sq = offset.length_squared();
            if distance_sq > i_radius_sq {
                return;
            }

            let distance = distance_sq.sqrt();
            let dir_p_to_n = if distance > min_dist {
                offset / distance
            } else {
                Vec2::ZERO
            };
            let inf = 1.0 - distance * i_radius_inv;

            let n_densities = densities[n_indice];
            let pressure_n = config.stiffness * (n_densities.x - config.rest_density);
            let pressure_near_n = config.stiffness_near * n_densities.y;

            let sum_pressure = pressure_p + pressure_n;
            let sum_pressure_near = pressure_near_p + pressure_near_n;

            let displacement =
                dt_sq * (sum_pressure * inf + sum_pressure_near * inf * inf) * (-dir_p_to_n);

            disp_p += displacement;
        });

        *next_p_pos += disp_p;
    }
}

fn clamp_boundary(positions: &mut Positions, config: &SimConfig) {
    for pos in positions.iter_mut() {
        *pos = pos.clamp(
            vec2(-config.bounds[0], -config.bounds[1]),
            vec2(config.bounds[0], config.bounds[1]),
        );
    }
}
