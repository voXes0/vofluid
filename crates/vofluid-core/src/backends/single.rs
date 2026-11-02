use crate::config::SimConfig;
use crate::grid::UniformGrid;
use crate::particle::{Densities, Particles, Positions};
use glam::*;

pub struct SingleThread {
    grid: UniformGrid,
    /// Scaled pressure `dt²·stiffness·(rho - rest)` per particle, in
    /// sorted-grid order (same slot order as `grid.sorted_xy()`). Built after
    /// pass 1 so pass 2 reads neighbour pressures as contiguous slices.
    sorted_pressure: Vec<f32>,
    /// Scaled near pressure `dt²·stiffness_near·rho_near`, same layout.
    sorted_pressure_near: Vec<f32>,
}
impl SingleThread {
    pub fn new(config: &SimConfig) -> Self {
        Self {
            grid: UniformGrid::new(
                config.bounds,
                config.interaction_radius * 0.5,
                config.particle_count,
            ),
            sorted_pressure: vec![0.0; config.particle_count],
            sorted_pressure_near: vec![0.0; config.particle_count],
        }
    }
    // Conventions: p is the iteration particle, n is the neighbor particle of the iteration particle
    pub fn step(&mut self, particles: &mut Particles, config: &SimConfig) {
        // Whole-step timing is `Simulation::step`'s job (it brackets every
        // backend the same way), so there's no span for it here.
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
                &mut self.sorted_pressure,
                &mut self.sorted_pressure_near,
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
        self.sorted_pressure.resize(new_particle_count, 0.0);
        self.sorted_pressure_near.resize(new_particle_count, 0.0);
    }
}

/// Per-step constants shared by both passes. `dt²` is folded into the
/// stiffnesses once here instead of being multiplied in per pair.
#[derive(Clone, Copy)]
struct Consts {
    radius: f32,
    inv_r: f32,
    min_dist: f32,
    k: f32,      // stiffness      * dt²
    k_near: f32, // stiffness_near * dt²
    rest: f32,
}

impl Consts {
    fn new(config: &SimConfig, dt: f32) -> Self {
        let dt_sq = dt * dt;
        let r = config.interaction_radius;
        Self {
            radius: r,
            inv_r: 1.0 / r,
            min_dist: r * 1e-3,
            k: config.stiffness * dt_sq,
            k_near: config.stiffness_near * dt_sq,
            rest: config.rest_density,
        }
    }
}

/// Plain compare-select. Verified in assembly: compiles to a single
/// `maxss`/`vmaxss` (`maxps`/`vmaxps` when vectorized), unlike `f32::max`,
/// which adds NaN handling on top. Returns `b` if either input is NaN.
#[inline(always)]
fn fmax(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// Pass 1: density / near-density per particle. Nothing but the densities is
/// stored; pass 2 recomputes the pair distances instead of reading them back
/// from memory.
///
/// For every stencil candidate (contiguous runs, one per stencil row)
/// `inf = max(0, 1 - d/r)`, which is exactly 0 outside the radius, so
/// out-of-range candidates add nothing to the sums.
fn gather_pass(positions: &[Vec2], densities: &mut [Vec2], grid: &UniformGrid, c: &Consts) {
    crate::prof_span!("single::double_density::gather_pass");

    // Profile-only bookkeeping: work counters, plus sampled per-particle
    // timing (`profiling_hw::PhaseSampler`: one particle in 32). On sampled
    // particles the real neighbours are also counted, after the timed part,
    // to estimate how many candidate tests are wasted.
    #[cfg(feature = "profile")]
    let (mut candidates, mut rows_seen) = (0u64, 0u64);
    #[cfg(feature = "profile")]
    let (mut candidates_sampled, mut neighbours_sampled) = (0u64, 0u64);
    #[cfg(feature = "profile")]
    let mut sampler = crate::profiling_hw::PhaseSampler::new([
        "single::double_density::gather_pass::particle",
    ]);

    let (sx, sy) = grid.sorted_xy();

    for (p, out) in positions.iter().zip(densities.iter_mut()) {
        #[cfg(feature = "profile")]
        let mut tick = sampler.start();
        #[cfg(feature = "profile")]
        let mut particle_candidates = 0usize;

        let mut rho = 0.0f32;
        let mut rho_near = 0.0f32;

        for r in grid.stencil_rows(*p, c.radius) {
            #[cfg(feature = "profile")]
            {
                particle_candidates += r.len();
                rows_seen += 1;
            }
            let xs = &sx[r.clone()];
            let ys = &sy[r];
            for (&x, &y) in xs.iter().zip(ys) {
                let ox = x - p.x;
                let oy = y - p.y;
                let d = fmax((ox * ox + oy * oy).sqrt(), c.min_dist);
                let f = fmax(1.0 - d * c.inv_r, 0.0);
                let f2 = f * f;
                rho = rho.algebraic_add(f2);
                rho_near = rho_near.algebraic_add(f2 * f);
            }
        }

        *out = vec2(rho, rho_near);

        #[cfg(feature = "profile")]
        {
            sampler.lap(&mut tick, 0, particle_candidates as f64);
            candidates += particle_candidates as u64;
            if tick.is_sampled() {
                // After the last lap, so none of this is timed.
                candidates_sampled += particle_candidates as u64;
                neighbours_sampled += count_in_radius(grid, sx, sy, *p, c);
            }
        }
    }

    #[cfg(feature = "profile")]
    sampler.finish();
    crate::prof_count!(
        "single::double_density::gather_pass::candidates",
        candidates
    );
    crate::prof_count!("single::double_density::gather_pass::rows", rows_seen);
    // `_sampled` counters are estimates from the sampled particles only.
    crate::prof_count!(
        "single::double_density::gather_pass::candidates_sampled",
        candidates_sampled
    );
    crate::prof_count!(
        "single::double_density::gather_pass::neighbours_sampled",
        neighbours_sampled
    );
}

/// Profile-only: how many of `p`'s stencil candidates lie inside the
/// interaction radius, i.e. are real neighbours (`inf > 0` in the passes).
/// Squared-distance compare, no sqrt, so it is cheap even on sampled
/// particles.
#[cfg(feature = "profile")]
fn count_in_radius(grid: &UniformGrid, sx: &[f32], sy: &[f32], p: Vec2, c: &Consts) -> u64 {
    let radius_sq = c.radius * c.radius;
    let mut inside = 0u64;
    for r in grid.stencil_rows(p, c.radius) {
        for (&x, &y) in sx[r.clone()].iter().zip(&sy[r]) {
            let (ox, oy) = (x - p.x, y - p.y);
            inside += (ox * ox + oy * oy < radius_sq) as u64;
        }
    }
    inside
}

/// Maps densities into sorted-grid order while converting them to scaled
/// pressures. Done once per particle here instead of once per pair in pass 2.
/// `particle_index[s]` is the original index of the particle in sorted slot s.
fn build_sorted_pressures(
    particle_index: &[usize],
    densities: &[Vec2],
    out_pressure: &mut [f32],
    out_pressure_near: &mut [f32],
    c: &Consts,
) {
    crate::prof_span!("single::double_density::sort_pressures");

    debug_assert_eq!(out_pressure.len(), particle_index.len());
    debug_assert_eq!(out_pressure_near.len(), particle_index.len());

    for ((p, pn), &orig) in out_pressure
        .iter_mut()
        .zip(out_pressure_near.iter_mut())
        .zip(particle_index)
    {
        let d = densities[orig];
        *p = c.k * (d.x - c.rest);
        *pn = c.k_near * d.y;
    }
}

/// Pass 2: pairwise pressure displacement. Recomputes the pair distance and
/// influence per candidate and applies the displacement in the same loop, so
/// the only memory it reads per candidate is the sorted positions and
/// pressures (contiguous runs, one per stencil row).
///
/// Candidates outside the radius have `inf == 0`, so `mag == 0` and they add
/// exactly nothing.
fn pressure_pass(
    positions: &[Vec2],
    densities: &[Vec2],
    next_positions: &mut [Vec2],
    sorted_pressure: &[f32],
    sorted_pressure_near: &[f32],
    grid: &UniformGrid,
    c: &Consts,
) {
    crate::prof_span!("single::double_density::pressure_pass");

    // Profile-only: work counters and sampled per-particle timing, as in
    // `gather_pass`. Pass 2 visits the same candidates as pass 1.
    #[cfg(feature = "profile")]
    let (mut candidates, mut rows_seen) = (0u64, 0u64);
    #[cfg(feature = "profile")]
    let mut sampler = crate::profiling_hw::PhaseSampler::new([
        "single::double_density::pressure_pass::particle",
    ]);

    let (sx, sy) = grid.sorted_xy();

    for ((p, d), next_p) in positions
        .iter()
        .zip(densities)
        .zip(next_positions.iter_mut())
    {
        #[cfg(feature = "profile")]
        let mut tick = sampler.start();
        #[cfg(feature = "profile")]
        let mut particle_candidates = 0usize;

        // This particle's own scaled pressure, once per particle.
        let pressure = c.k * (d.x - c.rest);
        let pressure_near = c.k_near * d.y;

        let (mut dx, mut dy) = (0.0f32, 0.0f32);
        for r in grid.stencil_rows(*p, c.radius) {
            #[cfg(feature = "profile")]
            {
                particle_candidates += r.len();
                rows_seen += 1;
            }
            let xs = &sx[r.clone()];
            let ys = &sy[r.clone()];
            let sp = &sorted_pressure[r.clone()];
            let spn = &sorted_pressure_near[r];
            for (((&x, &y), &q), &qn) in xs.iter().zip(ys).zip(sp).zip(spn) {
                let ox = x - p.x;
                let oy = y - p.y;
                let d = fmax((ox * ox + oy * oy).sqrt(), c.min_dist);
                let f = fmax(1.0 - d * c.inv_r, 0.0);
                let s = pressure + q;
                let s_near = pressure_near + qn;
                // dt²·(S·inf + S_near·inf²) == inf·(S + S_near·inf)
                let mag = f * (s + s_near * f);
                // displacement = unit direction · mag = offset · (mag / d)
                let k = mag / d;
                dx = dx.algebraic_add(ox * k);
                dy = dy.algebraic_add(oy * k);
            }
        }

        // Negate once instead of per pair (matches the old `-dir` sign).
        *next_p -= vec2(dx, dy);

        #[cfg(feature = "profile")]
        {
            sampler.lap(&mut tick, 0, particle_candidates as f64);
            candidates += particle_candidates as u64;
        }
    }

    #[cfg(feature = "profile")]
    sampler.finish();
    crate::prof_count!(
        "single::double_density::pressure_pass::candidates",
        candidates
    );
    crate::prof_count!("single::double_density::pressure_pass::rows", rows_seen);
}

#[allow(clippy::too_many_arguments)]
fn double_density(
    positions: &Positions,
    next_positions: &mut Positions,
    densities: &mut Densities,
    config: &SimConfig,
    grid: &UniformGrid,
    dt: f32,
    sorted_pressure: &mut [f32],
    sorted_pressure_near: &mut [f32],
) {
    crate::prof_span!("single::double_density");

    let c = Consts::new(config, dt);

    // Pass 1: density / near-density. Every densities[i] is overwritten, so
    // no reset() is needed.
    gather_pass(positions, densities, grid, &c);

    // Permute into sorted-grid order, converting density -> scaled pressure.
    build_sorted_pressures(
        grid.particle_index(),
        densities,
        sorted_pressure,
        sorted_pressure_near,
        &c,
    );

    // Pass 2: pairwise pressure displacement.
    pressure_pass(
        positions,
        densities,
        next_positions,
        sorted_pressure,
        sorted_pressure_near,
        grid,
        &c,
    );
}

fn clamp_boundary(positions: &mut Positions, bounds: Vec2) {
    crate::prof_span!("single::clamp_boundary");

    for pos in positions.iter_mut() {
        *pos = pos.clamp(vec2(-bounds.x, -bounds.y), vec2(bounds.x, bounds.y));
    }
}
