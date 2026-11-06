//! Internal structures (AOS) optimized for performance, not intended to be user facing API

use glam::Vec2;
use std::ops::{Deref, DerefMut};

/// Per-particle positions
#[derive(Clone, Debug, Default)]
pub struct Positions(pub Vec<Vec2>);

/// Per-particle velocities
#[derive(Clone, Debug, Default)]
pub struct Vels(pub Vec<Vec2>);

/// Per-particle previous positions. Write-once / read-once per simulation step
#[derive(Clone, Debug, Default)]
pub(crate) struct Prevs(pub(crate) Vec<Vec2>);

// deref newtype patten for type compile checks and conventional access
impl Deref for Positions {
    type Target = [Vec2];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Positions {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for Vels {
    type Target = [Vec2];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Vels {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for Prevs {
    type Target = [Vec2];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Prevs {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Transient per-particle density / near-density scratch values
/// so their memory is only allocated onece.
/// Stored as `Vec2 { x: density, y: near_density }` rather than two split
/// `Vec<f32>`s: every read/write site touches both values together
#[derive(Clone, Debug, Default)]
pub(crate) struct Densities(pub(crate) Vec<Vec2>);

impl Deref for Densities {
    type Target = [Vec2];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Densities {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Densities {
    #[inline]
    fn resize(&mut self, count: usize) {
        self.0.clear();
        self.0.resize(count, Vec2::ZERO);
    }
}

/// Struct of particle arrays
#[derive(Clone, Debug)]
pub struct Particles {
    pub positions: Positions,
    pub vels: Vels,
    pub(crate) prevs: Prevs,
    pub(crate) densities: Densities,
    /// Scratch buffer used during gather/scatter passes
    pub(crate) next_positions: Positions,
}

impl Particles {
    #[inline]
    pub fn new(count: usize) -> Self {
        Self {
            positions: Positions(vec![Vec2::ZERO; count]),
            vels: Vels(vec![Vec2::ZERO; count]),
            prevs: Prevs(vec![Vec2::ZERO; count]),
            densities: {
                let mut d = Densities::default();
                d.resize(count);
                d
            },
            next_positions: Positions(vec![Vec2::ZERO; count]),
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.positions.len()
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Adds a new particle at rest, returning its index.
    pub(crate) fn push(&mut self, position: Vec2, velocity: Vec2) -> usize {
        let idx = self.positions.len();
        self.positions.0.push(position);
        self.vels.0.push(velocity);
        self.prevs.0.push(position);
        self.densities.0.push(Vec2::ZERO);
        self.next_positions.0.push(position);
        idx
    }

    /// Removes a particle by swap-remove. The particle that was previously
    /// last now lives at `idx`, unless `idx` was already last, in which case
    /// nothing moved
    pub(crate) fn swap_remove(&mut self, idx: usize) -> usize {
        self.positions.0.swap_remove(idx);
        self.vels.0.swap_remove(idx);
        self.prevs.0.swap_remove(idx);
        self.densities.0.swap_remove(idx);
        self.next_positions.0.swap_remove(idx);
        idx
    }

    /// Snapshot of `position` saved into `prev`
    #[inline]
    pub(crate) fn save_prev_positions(&mut self) {
        crate::prof_span!("particles::save_prev_positions");

        self.prevs.copy_from_slice(&self.positions);
    }

    /// Reconstructs velocity from positional change since `save_prev_positions` was called
    #[inline]
    pub(crate) fn reconstruct_velocities(&mut self, dt: f32) {
        crate::prof_span!("particles::reconstruct_velocities");

        let inv_dt = 1.0 / dt;
        for ((pos, prev), vel) in self
            .positions
            .iter()
            .zip(self.prevs.iter())
            .zip(self.vels.iter_mut())
        {
            *vel = (*pos - *prev) * inv_dt;
        }
    }

    /// Applies a uniform acceleration to every particle's velocity
    #[inline]
    pub(crate) fn apply_uniform_acceleration(&mut self, accel: Vec2, dt: f32) {
        crate::prof_span!("particles::apply_uniform_acceleration");

        let delta = accel * dt;
        for vel in self.vels.iter_mut() {
            *vel += delta;
        }
    }

    /// Advances positions by velocity
    #[inline]
    pub(crate) fn integrate_positions(&mut self, dt: f32) {
        crate::prof_span!("particles::integrate_positions");

        for (pos, vel) in self.positions.iter_mut().zip(self.vels.iter()) {
            *pos += *vel * dt;
        }
    }

    /// Prepares `next_positions` for a scatter pass.
    /// copies current positions into it so in-place
    /// accumulation (`next_positions[i] += delta`) starts from valid positions.
    #[inline]
    pub(crate) fn begin_scatter_pass(&mut self) {
        crate::prof_span!("particles::begin_scatter_pass");

        self.next_positions.copy_from_slice(&self.positions);
    }

    /// Commits a scatter pass. swaps `positions` and `next_positions`.
    #[inline]
    pub(crate) fn end_scatter_pass(&mut self) {
        crate::prof_span!("particles::end_scatter_pass");

        std::mem::swap(&mut self.positions, &mut self.next_positions);
    }

    // ------------------------------------------------------------------
    // Public-facing conversion / access.
    // Everything about public API is experimental and in development.
    // ------------------------------------------------------------------

    /// Writes a snapshot into a caller-owned buffer, reusing its existing allocation.
    #[allow(dead_code)]
    pub(crate) fn write_api_particles(&self, out: &mut Vec<Particle>) {
        out.clear();
        out.extend(
            self.positions
                .iter()
                .zip(self.vels.iter())
                .map(|(&position, &velocity)| Particle { position, velocity }),
        );
    }

    /// Read-only per-particle view over position + velocity only
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = ParticleRef<'_>> {
        self.positions
            .iter()
            .zip(self.vels.iter())
            .map(|(position, velocity)| ParticleRef { position, velocity })
    }

    /// Mutable per-particle view over position + velocity only
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = ParticleMut<'_>> {
        self.positions
            .iter_mut()
            .zip(self.vels.iter_mut())
            .map(|(position, velocity)| ParticleMut { position, velocity })
    }

    /// Indexed read-only access to a single particle's public view.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<ParticleRef<'_>> {
        Some(ParticleRef {
            position: self.positions.get(idx)?,
            velocity: self.vels.get(idx)?,
        })
    }

    /// Indexed mutable access to a single particle's public view.
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> Option<ParticleMut<'_>> {
        if idx >= self.positions.len() {
            return None;
        }
        Some(ParticleMut {
            position: &mut self.positions.0[idx],
            velocity: &mut self.vels.0[idx],
        })
    }
}

// ============================================================================
// Public API, under development
// Only position + velocity are ever exposed.
// ============================================================================

/// User-facing snapshot of a single particle's position and velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Particle {
    pub position: Vec2,
    pub velocity: Vec2,
}

/// Borrowed read-only view of one particle's position and velocity, scoped
/// directly into the solver's internal storage (no copy).
#[derive(Debug)]
pub struct ParticleRef<'a> {
    pub position: &'a Vec2,
    pub velocity: &'a Vec2,
}

impl ParticleRef<'_> {
    #[inline]
    pub fn to_particle(&self) -> Particle {
        Particle {
            position: *self.position,
            velocity: *self.velocity,
        }
    }
}

/// Borrowed mutable view of one particle's position and velocity, scoped
/// directly into the solver's internal storage (no copy, no round-trip).
/// This is the recommended way for users to read/write particle state
/// per-frame — it mutates the solver's real backing arrays directly.
#[derive(Debug)]
pub struct ParticleMut<'a> {
    pub position: &'a mut Vec2,
    pub velocity: &'a mut Vec2,
}

impl ParticleMut<'_> {
    #[inline]
    pub fn to_particle(&self) -> Particle {
        Particle {
            position: *self.position,
            velocity: *self.velocity,
        }
    }

    #[inline]
    pub fn set(&mut self, particle: Particle) {
        *self.position = particle.position;
        *self.velocity = particle.velocity;
    }
}
