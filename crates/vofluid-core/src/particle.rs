use glam::Vec2;
use std::ops::{Deref, DerefMut};

// ============================================================================
// Internal SoA storage — pub(crate) only. Free to change layout without
// breaking the public API, as long as `Particle`/`ParticleRef`/`ParticleMut`
// keep behaving the same.
// ============================================================================

/// Per-particle positions. Touched by every algorithm in the solver
/// (integration, density relaxation, springs, collisions) — the hottest,
/// most frequently accessed field.
#[derive(Clone, Debug, Default)]
pub struct Positions(pub Vec<Vec2>);

/// Per-particle velocities. Touched by gravity, viscosity impulses, and
/// the final velocity reconstruction step (`v = (x - x_prev) / dt`).
#[derive(Clone, Debug, Default)]
pub struct Vels(pub Vec<Vec2>);

/// Per-particle previous positions. Write-once / read-once per simulation
/// step — bridges the start and end of integration so velocity can be
/// reconstructed from positional change.
#[derive(Clone, Debug, Default)]
pub(crate) struct Prevs(pub(crate) Vec<Vec2>);

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

/// Transient per-particle density / near-density scratch values, recomputed
/// every step inside double density relaxation. Not meaningful state between
/// steps — kept as a reusable buffer purely to avoid reallocating every
/// frame. Stored as `Vec2 { x: rho, y: rho_near }` rather than two split
/// `Vec<f32>`s: every read/write site touches both values together (see
/// Algorithm 2), so interleaving keeps both values in the same cache line
/// and lets `reset` zero both in a single pass instead of two.
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

    /// Zero out without shrinking capacity. Call at the start of each
    /// density-relaxation pass.
    #[inline]
    pub(crate) fn reset(&mut self) {
        self.0.iter_mut().for_each(|v| *v = Vec2::ZERO);
    }
}

/// Sparse spring graph between particle pairs (Algorithms 3 & 4).
///
/// Mirrors the GPU design: a bounded per-particle pool rather than an N×N
/// matrix or a hashmap. A spring can only exist between particles that are
/// currently neighbors (Algorithm 4 removes it once `L_ij > h`), so springs
/// per particle are bounded by `max_neighbor_count` — same capacity as your
/// neighbor query already enforces. Each particle owns its own spring slots;
/// lookup is a short linear scan over that particle's slots (capped at
/// `max_neighbor_count`, typically far under 128), which stays in L1/registers
/// and avoids both hashing and N×N memory. This intentionally matches the
/// `NeighborData` shape on the GPU side so CPU and GPU spring lifecycles stay
/// conceptually identical.
#[derive(Clone, Debug)]
pub(crate) struct Springs {
    /// Flattened pool: slot `p * capacity + k` is particle `p`'s k-th spring.
    /// `partner[slot] == u32::MAX` marks an empty slot.
    partner: Vec<u32>,
    rest_length: Vec<f32>,
    capacity: usize,
}

impl Springs {
    const EMPTY: u32 = u32::MAX;

    #[inline]
    pub(crate) fn new(particle_count: usize, capacity: usize) -> Self {
        Self {
            partner: vec![Self::EMPTY; particle_count * capacity],
            rest_length: vec![0.0; particle_count * capacity],
            capacity,
        }
    }

    #[inline]
    fn slots(&self, p: usize) -> std::ops::Range<usize> {
        let base = p * self.capacity;
        base..base + self.capacity
    }

    /// Finds particle `p`'s spring slot to `other`, if one exists. Short
    /// linear scan over `p`'s own slots only — no hashing, no global search.
    #[inline]
    pub(crate) fn find(&self, p: usize, other: u32) -> Option<usize> {
        self.slots(p).find(|&slot| self.partner[slot] == other)
    }

    /// Reads the rest length of `p`'s spring to `other`, if any.
    #[inline]
    pub(crate) fn rest_length(&self, p: usize, other: u32) -> Option<f32> {
        self.find(p, other).map(|slot| self.rest_length[slot])
    }

    /// Inserts or updates `p`'s spring to `other`. Returns `false` if no
    /// empty slot was available (pool exhausted) and the spring was dropped
    /// — mirrors the GPU overflow-counter behavior rather than panicking or
    /// growing, since the pool is sized to match the neighbor query capacity.
    pub(crate) fn set(&mut self, p: usize, other: u32, rest_length: f32) -> bool {
        if let Some(slot) = self.find(p, other) {
            self.rest_length[slot] = rest_length;
            return true;
        }
        match self
            .slots(p)
            .find(|&slot| self.partner[slot] == Self::EMPTY)
        {
            Some(slot) => {
                self.partner[slot] = other;
                self.rest_length[slot] = rest_length;
                true
            }
            None => false, // pool exhausted for this particle, dropped
        }
    }

    /// Removes `p`'s spring to `other`, if present.
    #[inline]
    pub(crate) fn remove(&mut self, p: usize, other: u32) {
        if let Some(slot) = self.find(p, other) {
            self.partner[slot] = Self::EMPTY;
        }
    }

    /// Clears all springs without shrinking the pool.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.partner.iter_mut().for_each(|p| *p = Self::EMPTY);
    }

    /// Particle `p`'s spring slots as `(partner_id, rest_length)`, skipping
    /// empty slots. Use this to iterate a particle's springs in Algorithm 3.
    #[inline]
    pub(crate) fn iter(&self, p: usize) -> impl Iterator<Item = (u32, f32)> + '_ {
        self.slots(p)
            .filter(move |&slot| self.partner[slot] != Self::EMPTY)
            .map(move |slot| (self.partner[slot], self.rest_length[slot]))
    }
}

// ============================================================================
// Internal particle store
// ============================================================================

#[derive(Clone, Debug)]
pub struct Particles {
    pub positions: Positions,
    pub vels: Vels,
    pub(crate) prevs: Prevs,
    pub(crate) densities: Densities,
    pub(crate) springs: Springs,
    /// Scratch buffer used during gather/scatter passes (density relaxation,
    /// spring displacement) that need to mutate neighbor positions without a
    /// multi-writer race. Initialized to a copy of `positions` at the start
    /// of the pass, accumulated into directly (no zero-fill needed — it
    /// already holds valid positions), then swapped with `positions` at the
    /// end — a pointer swap, not a copy-back. Cheaper than an accumulate-delta
    /// buffer, which needs both a zero-fill before and a read-modify-write
    /// apply pass after.
    pub(crate) next_positions: Positions,
}

impl Particles {
    #[inline]
    pub fn new(count: usize, max_springs_per_particle: usize) -> Self {
        Self {
            positions: Positions(vec![Vec2::ZERO; count]),
            vels: Vels(vec![Vec2::ZERO; count]),
            prevs: Prevs(vec![Vec2::ZERO; count]),
            densities: {
                let mut d = Densities::default();
                d.resize(count);
                d
            },
            springs: Springs::new(count, max_springs_per_particle),
            next_positions: Positions(vec![Vec2::ZERO; count]),
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.positions.len()
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Adds a new particle at rest, returning its index. Note: does not
    /// resize the spring pool's per-particle capacity — `Springs` is sized
    /// at construction and assumes a fixed particle count. If you need
    /// dynamic particle counts with springs, resize `Springs` separately.
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
    /// last now lives at `idx` (unless `idx` was already last, in which case
    /// nothing moved). Callers should fix up any external indices (e.g.
    /// spring pairs) that referenced `last` to now point at `idx`.
    pub(crate) fn swap_remove(&mut self, idx: usize) -> usize {
        self.positions.0.swap_remove(idx);
        self.vels.0.swap_remove(idx);
        self.prevs.0.swap_remove(idx);
        self.densities.0.swap_remove(idx);
        self.next_positions.0.swap_remove(idx);
        idx
    }

    /// Snapshot of `position`/`velocity` saved into `prev`/start-of-step
    /// state. Called at the top of each simulation step (Algorithm 1, line 8).
    #[inline]
    pub(crate) fn save_prev_positions(&mut self) {
        #[cfg(feature = "profile")]
        let _ = tracy_client::span!();

        self.prevs.copy_from_slice(&self.positions);
    }

    /// Reconstructs velocity from positional change since `save_prev_positions`
    /// was called (Algorithm 1, line 20: `v = (x - x_prev) / dt`).
    #[inline]
    pub(crate) fn reconstruct_velocities(&mut self, dt: f32) {
        #[cfg(feature = "profile")]
        let _ = tracy_client::span!();

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

    /// Applies a uniform acceleration (e.g. gravity) to every particle's
    /// velocity (Algorithm 1, line 3: `v_i += dt * g`).
    #[inline]
    pub(crate) fn apply_uniform_acceleration(&mut self, accel: Vec2, dt: f32) {
        #[cfg(feature = "profile")]
        let _ = tracy_client::span!();

        let delta = accel * dt;
        for vel in self.vels.iter_mut() {
            *vel += delta;
        }
    }

    /// Advances positions by velocity (Algorithm 1, line 10: `x_i += dt * v_i`).
    #[inline]
    pub(crate) fn integrate_positions(&mut self, dt: f32) {
        #[cfg(feature = "profile")]
        let _ = tracy_client::span!();

        for (pos, vel) in self.positions.iter_mut().zip(self.vels.iter()) {
            *pos += *vel * dt;
        }
    }

    /// Prepares `next_positions` for a scatter pass (density relaxation, spring
    /// displacement): copies current positions into it so in-place
    /// accumulation (`next_positions[i] += delta`) starts from valid positions,
    /// with no zero-fill step needed.
    #[inline]
    pub(crate) fn begin_scatter_pass(&mut self) {
        #[cfg(feature = "profile")]
        let _ = tracy_client::span!();

        self.next_positions.copy_from_slice(&self.positions);
    }

    /// Commits a scatter pass: swaps `positions` and `next_positions`. A pointer
    /// swap, not a copy — `next_positions` becomes the (now stale) previous
    /// `positions` buffer, ready to be overwritten by the next
    /// `begin_scatter_pass` call.
    #[inline]
    pub(crate) fn end_scatter_pass(&mut self) {
        #[cfg(feature = "profile")]
        let _ = tracy_client::span!();

        std::mem::swap(&mut self.positions, &mut self.next_positions);
    }

    // ------------------------------------------------------------------
    // Public-facing conversion / access
    // ------------------------------------------------------------------

    /// One-shot snapshot of all particles as the public `Particle` type.
    /// Prefer `iter()`/`iter_mut()` for repeated or mutating access — this
    /// allocates a fresh `Vec` every call.
    pub(crate) fn to_api_particles(&self) -> Vec<Particle> {
        self.positions
            .iter()
            .zip(self.vels.iter())
            .map(|(&position, &velocity)| Particle { position, velocity })
            .collect()
    }

    /// Writes a snapshot into a caller-owned buffer, reusing its existing
    /// allocation. Use this instead of `to_api_particles` in a hot/per-frame
    /// path to avoid allocating every call.
    pub(crate) fn write_api_particles(&self, out: &mut Vec<Particle>) {
        out.clear();
        out.extend(
            self.positions
                .iter()
                .zip(self.vels.iter())
                .map(|(&position, &velocity)| Particle { position, velocity }),
        );
    }

    /// Read-only per-particle view over position + velocity only — `prev`,
    /// density, and springs are intentionally inaccessible through this type.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = ParticleRef<'_>> {
        self.positions
            .iter()
            .zip(self.vels.iter())
            .map(|(position, velocity)| ParticleRef { position, velocity })
    }

    /// Mutable per-particle view over position + velocity only — `prev`,
    /// density, and springs are intentionally inaccessible through this type,
    /// so users can't desync solver invariants they shouldn't be touching.
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
// Public API — stable surface, decoupled from internal SoA layout.
// Only position + velocity are ever exposed; `prev`, density, and springs
// are solver-internal and never touched from outside this crate.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_initializes_all_buffers_to_matching_length() {
        let p = Particles::new(5, 8);
        assert_eq!(p.len(), 5);
        assert_eq!(p.positions.len(), 5);
        assert_eq!(p.vels.len(), 5);
        assert_eq!(p.prevs.len(), 5);
        assert_eq!(p.densities.len(), 5);
        assert_eq!(p.next_positions.len(), 5);
    }

    #[test]
    fn push_and_swap_remove_keep_buffers_in_sync() {
        let mut p = Particles::new(0, 8);
        p.push(Vec2::new(1.0, 0.0), Vec2::ZERO);
        p.push(Vec2::new(2.0, 0.0), Vec2::ZERO);
        p.push(Vec2::new(3.0, 0.0), Vec2::ZERO);

        p.swap_remove(0);

        assert_eq!(p.len(), 2);
        // index 0 now holds what was previously the last particle (x = 3.0)
        assert_eq!(p.positions[0], Vec2::new(3.0, 0.0));
    }

    #[test]
    fn integration_round_trip_reconstructs_velocity() {
        let mut p = Particles::new(1, 8);
        p.vels[0] = Vec2::new(1.0, 2.0);
        let dt = 0.1;

        p.save_prev_positions();
        p.integrate_positions(dt);
        p.reconstruct_velocities(dt);

        // Velocity should reconstruct to (approximately) what it was.
        assert!((p.vels[0] - Vec2::new(1.0, 2.0)).length() < 1e-5);
    }

    #[test]
    fn scatter_pass_ping_pong_commits_accumulated_positions() {
        let mut p = Particles::new(2, 8);
        p.positions[0] = Vec2::new(1.0, 1.0);
        p.positions[1] = Vec2::new(2.0, 2.0);

        p.begin_scatter_pass();
        // simulate a gather/scatter step accumulating directly into next_positions
        p.next_positions[0] += Vec2::new(1.0, 0.0);
        p.next_positions[1] += Vec2::new(0.0, 1.0);
        p.end_scatter_pass();

        assert_eq!(p.positions[0], Vec2::new(2.0, 1.0));
        assert_eq!(p.positions[1], Vec2::new(2.0, 3.0));

        // next_positions now holds the stale pre-pass positions, ready to be
        // overwritten by the next begin_scatter_pass call.
        assert_eq!(p.next_positions[0], Vec2::new(1.0, 1.0));
        assert_eq!(p.next_positions[1], Vec2::new(2.0, 2.0));
    }

    #[test]
    fn public_iter_mut_only_exposes_position_and_velocity() {
        let mut p = Particles::new(1, 8);
        for particle in p.iter_mut() {
            *particle.position = Vec2::new(5.0, 5.0);
            *particle.velocity = Vec2::new(1.0, 0.0);
        }
        assert_eq!(p.positions[0], Vec2::new(5.0, 5.0));
        assert_eq!(p.vels[0], Vec2::new(1.0, 0.0));
    }

    #[test]
    fn springs_set_find_remove() {
        let mut springs = Springs::new(4, 8);
        springs.set(0, 3, 1.5);
        springs.set(1, 2, 2.0);

        assert!(springs.find(0, 3).is_some());
        assert_eq!(springs.rest_length(0, 3), Some(1.5));
        // ownership is one-directional by design: particle 3 doesn't
        // automatically see a spring back to 0 unless set separately.
        assert!(springs.find(3, 0).is_none());

        springs.remove(0, 3);
        assert!(springs.find(0, 3).is_none());
        assert!(springs.find(1, 2).is_some());
    }

    #[test]
    fn springs_pool_exhaustion_reports_failure_without_panicking() {
        let mut springs = Springs::new(2, 2); // capacity 2 per particle
        assert!(springs.set(0, 10, 1.0));
        assert!(springs.set(0, 11, 1.0));
        // third spring for particle 0 has no free slot
        assert!(!springs.set(0, 12, 1.0));
    }

    #[test]
    fn springs_iter_skips_empty_slots() {
        let mut springs = Springs::new(2, 4);
        springs.set(0, 5, 1.0);
        springs.set(0, 6, 2.0);

        let collected: Vec<(u32, f32)> = springs.iter(0).collect();
        assert_eq!(collected.len(), 2);
        assert!(collected.contains(&(5, 1.0)));
        assert!(collected.contains(&(6, 2.0)));
    }
}
