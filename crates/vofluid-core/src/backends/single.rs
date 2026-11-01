use crate::config::SimConfig;
use crate::grid::UniformGrid;
use crate::neighbor::NeighbourCache;
use crate::particle::{Densities, Particles, Positions};
use glam::*;

pub struct SingleThread {
    grid: UniformGrid,
    neighbour_cache: NeighbourCache,
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
            neighbour_cache: NeighbourCache::with_capacity(config.particle_count, 64),
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
                &mut self.neighbour_cache,
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

/// Pass 1: per particle, density / near-density plus the cached influence of
/// every stencil candidate.
///
///   B. branch-free elementwise math over every candidate (contiguous runs,
///      one per stencil row), written straight into the cache arrays. `inf`
///      is clamped to 0 for candidates outside the radius.
///   C. density sums over `inf`. Out-of-range candidates add exactly 0, and
///      they stay in the cache as zero-contribution entries.
/// The sums are kept out of loop B on purpose: a float reduction would stop
/// the compiler from vectorizing that loop.
fn gather_pass(
    positions: &[Vec2],
    densities: &mut [Vec2],
    grid: &UniformGrid,
    c: &Consts,
    cache: &mut NeighbourCache,
) {
    crate::prof_span!("single::double_density::gather_pass");

    #[cfg(feature = "profile")]
    let mut candidates = 0u64;

    let (sx, sy) = grid.sorted_xy();

    for (p, out) in positions.iter().zip(densities.iter_mut()) {
        let p = *p;

        let rows = grid.stencil_rows(p, c.radius);
        let n: usize = rows.clone().map(|r| r.len()).sum();
        #[cfg(feature = "profile")]
        {
            candidates += n as u64;
        }
        let (c_inf, c_ux, c_uy) = cache.candidate_slices(n);

        // B: elementwise, no branches, no reductions.
        let mut off = 0;
        for r in rows {
            let xs = &sx[r.clone()];
            let ys = &sy[r];
            let m = xs.len();
            let inf = &mut c_inf[off..off + m];
            let ux = &mut c_ux[off..off + m];
            let uy = &mut c_uy[off..off + m];
            for i in 0..m {
                let dx = xs[i] - p.x;
                let dy = ys[i] - p.y;
                let d = fmax((dx * dx + dy * dy).sqrt(), c.min_dist);
                let inv_d = d.recip();
                inf[i] = fmax(1.0 - d * c.inv_r, 0.0);
                ux[i] = dx * inv_d;
                uy[i] = dy * inv_d;
            }
            off += m;
        }

        // C: sum densities.
        let (mut rho, mut rho_near) = (0.0f32, 0.0f32);
        for &f in c_inf.iter() {
            let f2 = f.algebraic_mul(f);
            rho = rho.algebraic_add(f2);
            rho_near = rho_near.algebraic_add(f2.algebraic_mul(f));
        }

        *out = vec2(rho, rho_near);
        cache.finish_particle();
    }

    crate::prof_count!(
        "single::double_density::gather_pass::candidates",
        candidates
    );
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

/// Pass 2: pairwise pressure displacement.
///
/// Pass 1 stored one cache entry per stencil candidate in `stencil_rows`
/// order. Recomputing the same rows here (same positions, same grid, so the
/// same ranges) lines the cache entries up with contiguous runs of the sorted
/// pressure arrays, so no per-neighbour index lookup is needed.
fn pressure_pass(
    positions: &[Vec2],
    densities: &[Vec2],
    next_positions: &mut [Vec2],
    sorted_pressure: &[f32],
    sorted_pressure_near: &[f32],
    grid: &UniformGrid,
    cache: &NeighbourCache,
    c: &Consts,
) {
    crate::prof_span!("single::double_density::pressure_pass");

    for (i, ((pos, d), next_p)) in positions
        .iter()
        .zip(densities)
        .zip(next_positions.iter_mut())
        .enumerate()
    {
        // This particle's own scaled pressure, once per particle.
        let pressure = c.k * (d.x - c.rest);
        let pressure_near = c.k_near * d.y;
        let nb = cache.neighbours_of(i);

        let (mut dx, mut dy) = (0.0f32, 0.0f32);
        let mut off = 0;
        for r in grid.stencil_rows(*pos, c.radius) {
            let sp = &sorted_pressure[r.clone()];
            let spn = &sorted_pressure_near[r];
            let m = sp.len();
            let inf = &nb.inf[off..off + m];
            let ux = &nb.dir_x[off..off + m];
            let uy = &nb.dir_y[off..off + m];
            for j in 0..m {
                let s = pressure + sp[j];
                let s_near = pressure_near + spn[j];
                let f = inf[j];
                // dt²·(S·inf + S_near·inf²) == inf·(S + S_near·inf)
                let mag = f * (s + s_near * f);
                dx = dx.algebraic_add(ux[j] * mag);
                dy = dy.algebraic_add(uy[j] * mag);
            }
            off += m;
        }

        // Negate once instead of per pair (matches the old `-dir` sign).
        *next_p -= vec2(dx, dy);
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
    sorted_pressure: &mut [f32],
    sorted_pressure_near: &mut [f32],
) {
    crate::prof_span!("single::double_density");

    let c = Consts::new(config, dt);

    // Pass 1: gather density / near-density, cache neighbour influence.
    // Every densities[i] is overwritten, so no reset() is needed.
    cache.clear();
    gather_pass(positions, densities, grid, &c, cache);
    crate::prof_count!("single::double_density::neighbours", cache.len());

    // Permute into sorted-grid order, converting density -> scaled pressure.
    build_sorted_pressures(
        grid.particle_index(),
        densities,
        sorted_pressure,
        sorted_pressure_near,
        &c,
    );

    // Pass 2: pairwise pressure displacement from the cached entries.
    pressure_pass(
        positions,
        densities,
        next_positions,
        sorted_pressure,
        sorted_pressure_near,
        grid,
        cache,
        &c,
    );
}

fn clamp_boundary(positions: &mut Positions, bounds: Vec2) {
    crate::prof_span!("single::clamp_boundary");

    for pos in positions.iter_mut() {
        *pos = pos.clamp(vec2(-bounds.x, -bounds.y), vec2(bounds.x, bounds.y));
    }
}
