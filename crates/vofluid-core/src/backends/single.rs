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
                config.bounds * 1.5,
                config.interaction_radius,
                config.particle_count,
            ),
        }
    }
    pub fn step(&mut self, particles: &mut Particles, config: &SimConfig) {
        for _ in 0..config.substeps {
            particles.apply_uniform_acceleration(config.gravity, config.dt);
            particles.save_prev_positions();
            particles.integrate_positions(config.dt);
            self.grid.rebuild(&particles.positions);
            particles.begin_scatter_pass();
            double_density(
                &particles.positions,
                &mut particles.next_positions,
                &mut particles.densities,
                config,
                &self.grid,
            );
            particles.end_scatter_pass();
            clamp_boundary(&mut particles.positions, config);
            particles.reconstruct_velocities(config.dt);
        }
    }
}

fn double_density(
    positions: &Positions,
    next_positions: &mut Positions,
    densities: &mut Densities,
    config: &SimConfig,
    grid: &UniformGrid,
) {
    let dt = config.dt;
    let dt_sq = dt * dt;
    let i_radius = config.interaction_radius;
    let i_radius_inv = 1.0 / i_radius;
    let i_radius_sq = i_radius * i_radius;

    densities.reset();

    for ((p_indice, pos), densities) in positions.iter().enumerate().zip(densities.iter_mut()) {
        grid.query_neighbours(*pos, |n_indice| {
            if n_indice == p_indice {
                return;
            }
            let n_pos = positions[n_indice];
            let offset = n_pos - pos;

            let distance_sq = offset.length_squared();
            if distance_sq > i_radius_sq {
                return;
            }

            let distance = distance_sq.sqrt();
            let inf = 1.0 - distance * i_radius_inv; // influence of the particle on neighbor and vice versa, 0 at the edge of the radius and 1 at the center
            let inf_sq = inf * inf;

            densities.x += inf_sq; // x is density
            densities.y += inf_sq * inf; // y in near density
        });

        let pressure = config.stiffness * (densities.x - config.rest_density);
        let pressure_near = config.stiffness_near * densities.y;

        let mut displacement_p = Vec2::ZERO;

        grid.query_neighbours(*pos, |n_indice| {
            if n_indice == p_indice {
                return;
            }

            let n_pos = positions[n_indice];
            let offset = n_pos - pos;

            let distance_sq = offset.length_squared();
            if distance_sq > i_radius_sq {
                return;
            }

            let distance = distance_sq.sqrt();
            let dir = if distance > 0.0 {
                offset / distance
            } else {
                Vec2::ZERO
            };
            let inf = 1.0 - distance * i_radius_inv;

            let displacement = dt_sq * (pressure * inf + pressure_near * inf * inf) * dir;

            next_positions[n_indice] += displacement;
            displacement_p -= displacement;
        });

        next_positions[p_indice] += displacement_p;
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
