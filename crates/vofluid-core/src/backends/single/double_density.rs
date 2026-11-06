//! Double density relaxation (Clavet, Beaudoin & Poulin, 2005).
//!
//! After positions are integrated, every particle `p` is pushed away from (or
//! pulled towards) its neighbours `n` within the interaction radius `r`. With
//! `d = |n - p|`, the weight `w = max(0, 1 - d/r)` and the substep length `dt`:
//!
//! ```text
//! rho        = Σ w²                      over the neighbours of p (p included)
//! rho_near   = Σ w³
//! P          = stiffness      · dt² · (rho - rest_density)
//! P_near     = stiffness_near · dt² · rho_near
//! displacement = -Σ w · ((P_p + P_n) + (P_near_p + P_near_n) · w) · (n - p) / d
//! ```
//!
//! This is the *gather* form: each pair is evaluated from both sides and a
//! particle only writes its own displacement, so there are no write conflicts
//! and no pair symmetry to exploit. Threaded and GPU versions can keep the
//! same structure. Reads come from the grid's sorted positions and writes go
//! to `next_positions` (Jacobi style, no in-place updates).
//!
//! Two passes over the particles, in slot order, cell by cell (all particles
//! of a cell share one stencil):
//! 1. `gather_pass`: densities -> scaled pressures.
//! 2. `pressure_pass`: pressures -> displacements.
//!
//! Naming: `p` is the particle being updated, `n` a candidate neighbour.
//!
//! # Kernels
//!
//! Each stencil row is a contiguous slot range, processed 8 candidates at a
//! time with explicit `f32x8` vectors. Accumulators stay in vector registers
//! across all rows, so each particle needs one horizontal sum. A row's last
//! chunk is a full-width load whose lanes past the row's end are masked to
//! contribute exactly zero (they hold other rows' particles or stale slots).
//! Candidates outside the radius have `w == 0` and add nothing either.
//!
//! Performance notes (measured on AVX2 + FMA x86; percentages are per pass):
//! - Explicit vectors beat `[f32; 8]` loops left to the autovectorizer, which
//!   came out slower than scalar code and varied with unrelated build flags.
//! - `mul_add` / `mul_neg_add` fuse into FMA where the target has it
//!   (-7% gather, -13% pressure). Without FMA `wide` emits a separate multiply
//!   and add, so results differ in the last bits between such builds.
//! - Re-slicing the arrays to one length makes one bounds check per chunk
//!   cover every load (~3%).
//! - Tried and dropped: one flat chunk list per cell (3-7% slower); `rsqrt` +
//!   Newton instead of `sqrt` and a division (slower, and not reproducible
//!   across CPU vendors); a mask-free loop over full chunks with a peeled
//!   masked tail (cheaper in `llvm-mca`, 4-7% slower measured).

use super::Relaxation;
use super::probe::{PassLabels, PassProbe};
use super::simd::{LANES, lane_mask, load};
use crate::config::SimConfig;
use crate::grid::{SORTED_PAD, Stencil, UniformGrid};
use glam::{Vec2, vec2};
use std::ops::Range;
use wide::f32x8;

pub struct DoubleDensity {
    /// Scaled pressure `P` per particle, in slot order (see `UniformGrid`).
    /// Written by pass 1 for every slot of a sorted particle, read by pass 2.
    pub(super) pressure: Vec<f32>,
    /// Scaled near pressure `P_near`, same layout.
    pub(super) pressure_near: Vec<f32>,
    stencil: Stencil,
}

impl Relaxation for DoubleDensity {
    fn new(config: &SimConfig) -> Self {
        let slots = config.particle_count + SORTED_PAD;
        Self {
            pressure: vec![0.0; slots],
            pressure_near: vec![0.0; slots],
            stencil: Stencil::default(),
        }
    }

    fn resize_particles(&mut self, particle_count: usize) {
        self.pressure.resize(particle_count + SORTED_PAD, 0.0);
        self.pressure_near.resize(particle_count + SORTED_PAD, 0.0);
    }

    fn relax(
        &mut self,
        grid: &UniformGrid,
        next_positions: &mut [Vec2],
        config: &SimConfig,
        dt: f32,
    ) {
        crate::prof_span!("single::double_density");

        let params = Params::new(config, dt);
        self.gather_pass(grid, &params);
        self.pressure_pass(grid, next_positions, &params);
    }
}

impl DoubleDensity {
    /// Pass 1: density and near density of every particle, stored as the
    /// scaled pressures pass 2 needs.
    fn gather_pass(&mut self, grid: &UniformGrid, params: &Params) {
        crate::prof_span!("single::double_density::gather_pass");

        let (x, y) = grid.sorted_xy();
        let mut probe = PassProbe::new(&GATHER_LABELS);
        for (cx, cy, slots) in grid.occupied_cells() {
            grid.load_stencil(&mut self.stencil, cx, cy, params.radius);
            let rows = self.stencil.rows();
            probe.cell(rows);
            for slot in slots {
                let p = vec2(x[slot], y[slot]);
                let tick = probe.begin();
                let (rho, rho_near) = density(p, rows, x, y, params);
                self.pressure[slot] = params.k * (rho - params.rest_density);
                self.pressure_near[slot] = params.k_near * rho_near;
                probe.end_counting(tick, || neighbours_in_radius(p, rows, x, y, params));
            }
        }
        probe.finish();
    }

    /// Pass 2: the displacement of every particle, added to its entry in
    /// `next_positions` (the one indirect access: slot -> original index).
    fn pressure_pass(&mut self, grid: &UniformGrid, next_positions: &mut [Vec2], params: &Params) {
        crate::prof_span!("single::double_density::pressure_pass");

        let (x, y) = grid.sorted_xy();
        let particle_index = grid.particle_index();
        let sorted = Sorted {
            x,
            y,
            pressure: &self.pressure,
            pressure_near: &self.pressure_near,
        };
        let mut probe = PassProbe::new(&PRESSURE_LABELS);
        for (cx, cy, slots) in grid.occupied_cells() {
            grid.load_stencil(&mut self.stencil, cx, cy, params.radius);
            let rows = self.stencil.rows();
            probe.cell(rows);
            for slot in slots {
                let tick = probe.begin();
                next_positions[particle_index[slot]] += displacement(slot, rows, &sorted, params);
                probe.end(tick);
            }
        }
        probe.finish();
    }
}

/// Per-substep constants.
#[derive(Clone, Copy)]
pub(super) struct Params {
    pub radius: f32,
    pub inv_radius: f32,
    /// Distances are clamped up to this, so coincident particles never
    /// divide by zero.
    pub min_dist: f32,
    /// `stiffness · dt²`
    pub k: f32,
    /// `stiffness_near · dt²`
    pub k_near: f32,
    pub rest_density: f32,
}

impl Params {
    pub fn new(config: &SimConfig, dt: f32) -> Self {
        let dt_sq = dt * dt;
        let radius = config.interaction_radius;
        Self {
            radius,
            inv_radius: 1.0 / radius,
            min_dist: radius * 1e-3,
            k: config.stiffness * dt_sq,
            k_near: config.stiffness_near * dt_sq,
            rest_density: config.rest_density,
        }
    }
}

/// `Params` broadcast to all lanes.
struct LaneParams {
    inv_radius: f32x8,
    min_dist: f32x8,
}

impl LaneParams {
    #[inline(always)]
    fn new(params: &Params) -> Self {
        Self {
            inv_radius: f32x8::splat(params.inv_radius),
            min_dist: f32x8::splat(params.min_dist),
        }
    }
}

/// Sorted particle data read by pass 2, indexed by slot.
struct Sorted<'a> {
    x: &'a [f32],
    y: &'a [f32],
    pressure: &'a [f32],
    pressure_near: &'a [f32],
}

/// Distance `d` (clamped to `min_dist`) and weight `w = max(0, 1 - d/r)` for
/// one chunk of offsets `n - p`.
#[inline(always)]
fn distance_weight(ox: f32x8, oy: f32x8, c: &LaneParams) -> (f32x8, f32x8) {
    let d = ox.mul_add(ox, oy * oy).sqrt().fast_max(c.min_dist);
    let w = d
        .mul_neg_add(c.inv_radius, f32x8::ONE)
        .fast_max(f32x8::ZERO);
    (d, w)
}

/// Pass 1 kernel: `(rho, rho_near)` of the particle at `p`.
fn density(p: Vec2, rows: &[Range<usize>], x: &[f32], y: &[f32], params: &Params) -> (f32, f32) {
    let c = LaneParams::new(params);
    let (px, py) = (f32x8::splat(p.x), f32x8::splat(p.y));
    // Same length as `x`: one bounds check per chunk covers both loads.
    let y = &y[..x.len()];
    let (mut rho, mut rho_near) = (f32x8::ZERO, f32x8::ZERO);
    for row in rows {
        let mut at = row.start;
        while at < row.end {
            let mask = lane_mask((row.end - at).min(LANES));
            let (_, w) = distance_weight(load(x, at) - px, load(y, at) - py, &c);
            let w = w & mask;
            let w2 = w * w;
            rho += w2;
            rho_near = w2.mul_add(w, rho_near);
            at += LANES;
        }
    }
    (rho.reduce_add(), rho_near.reduce_add())
}

/// Pass 2 kernel: the displacement of the particle in `slot`.
fn displacement(slot: usize, rows: &[Range<usize>], sorted: &Sorted, params: &Params) -> Vec2 {
    let c = LaneParams::new(params);
    let (px, py) = (f32x8::splat(sorted.x[slot]), f32x8::splat(sorted.y[slot]));
    let pressure = f32x8::splat(sorted.pressure[slot]);
    let pressure_near = f32x8::splat(sorted.pressure_near[slot]);
    // All the same length: one bounds check per chunk covers every load.
    let len = sorted.x.len();
    let (y, n_pressure, n_pressure_near) = (
        &sorted.y[..len],
        &sorted.pressure[..len],
        &sorted.pressure_near[..len],
    );
    let (mut sum_x, mut sum_y) = (f32x8::ZERO, f32x8::ZERO);
    for row in rows {
        let mut at = row.start;
        while at < row.end {
            let mask = lane_mask((row.end - at).min(LANES));
            let (ox, oy) = (load(sorted.x, at) - px, load(y, at) - py);
            let (d, w) = distance_weight(ox, oy, &c);
            let s = pressure + load(n_pressure, at);
            let s_near = pressure_near + load(n_pressure_near, at);
            // w·(s + s_near·w), with the lanes past the row's end zeroed.
            let magnitude = (w * s_near.mul_add(w, s)) & mask;
            // Direction (n - p)/d times magnitude.
            let k = magnitude / d;
            sum_x = ox.mul_add(k, sum_x);
            sum_y = oy.mul_add(k, sum_y);
            at += LANES;
        }
    }
    // Negated once per particle instead of once per pair.
    -vec2(sum_x.reduce_add(), sum_y.reduce_add())
}

/// How many candidates lie inside the radius
fn neighbours_in_radius(
    p: Vec2,
    rows: &[Range<usize>],
    x: &[f32],
    y: &[f32],
    params: &Params,
) -> u64 {
    let radius_sq = params.radius * params.radius;
    let mut inside = 0;
    for row in rows {
        for (&nx, &ny) in x[row.clone()].iter().zip(&y[row.clone()]) {
            let (ox, oy) = (nx - p.x, ny - p.y);
            inside += (ox * ox + oy * oy < radius_sq) as u64;
        }
    }
    inside
}

// Profiler names.
static GATHER_LABELS: PassLabels = PassLabels {
    particle: "single::double_density::gather_pass::particle",
    candidates: "single::double_density::gather_pass::candidates",
    rows: "single::double_density::gather_pass::rows",
    chunks: "single::double_density::gather_pass::chunks",
    sampled: Some((
        "single::double_density::gather_pass::candidates_sampled",
        "single::double_density::gather_pass::neighbours_sampled",
    )),
};

static PRESSURE_LABELS: PassLabels = PassLabels {
    particle: "single::double_density::pressure_pass::particle",
    candidates: "single::double_density::pressure_pass::candidates",
    rows: "single::double_density::pressure_pass::rows",
    chunks: "single::double_density::pressure_pass::chunks",
    sampled: None,
};
