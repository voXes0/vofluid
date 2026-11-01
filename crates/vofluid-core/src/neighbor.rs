/// Neighbour influence cache in CSR form, stored as separate arrays
/// (structure of arrays): particle `p`'s entries live at
/// `offsets[p] .. offsets[p + 1]` in each array.
///
/// A particle has one entry per stencil candidate, in the order
/// `UniformGrid::stencil_rows` yields them (row by row, slot order within a
/// row), so the neighbour slots are recoverable from the grid and are not
/// stored. Candidates outside the interaction radius keep `inf == 0`, which
/// makes them contribute exactly nothing downstream.
///
/// Valid only until the next `UniformGrid::rebuild()`.
pub struct NeighbourCache {
    /// Length is always (particles finished so far) + 1.
    offsets: Vec<u32>,
    /// Number of entries handed out so far. The arrays below may be longer;
    /// anything past `len` is scratch space.
    len: usize,
    inf: Vec<f32>,
    dir_x: Vec<f32>,
    dir_y: Vec<f32>,
}

/// Borrowed view of one particle's cached entries. All slices have the same
/// length and are in the same order.
pub struct Neighbours<'a> {
    pub inf: &'a [f32],
    /// x / y of the unit direction from the particle to the neighbour.
    pub dir_x: &'a [f32],
    pub dir_y: &'a [f32],
}

impl NeighbourCache {
    pub fn new() -> Self {
        Self {
            offsets: vec![0],
            len: 0,
            inf: Vec::new(),
            dir_x: Vec::new(),
            dir_y: Vec::new(),
        }
    }

    /// Preallocate for a known particle/candidate count to avoid growth
    /// churn on the first few steps. `avg_candidates` is a rough estimate;
    /// the arrays grow on demand if it's too low.
    pub fn with_capacity(particle_count: usize, avg_candidates: usize) -> Self {
        let cap = particle_count * avg_candidates;
        let mut offsets = Vec::with_capacity(particle_count + 1);
        offsets.push(0);
        Self {
            offsets,
            len: 0,
            inf: vec![0.0; cap],
            dir_x: vec![0.0; cap],
            dir_y: vec![0.0; cap],
        }
    }

    /// Reset for a new pass, keeping allocated capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.offsets.clear();
        self.offsets.push(0);
        self.len = 0;
    }

    /// Total entries so far. Only the profiler's counters read this, hence
    /// the gate.
    #[cfg(feature = "profile")]
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Reserves `n` entries for the particle currently being built and
    /// returns them as writable slices (`inf`, `dir_x`, `dir_y`), growing the
    /// arrays if needed. Call `finish_particle` once per particle after
    /// writing.
    #[inline]
    pub fn candidate_slices(&mut self, n: usize) -> (&mut [f32], &mut [f32], &mut [f32]) {
        let a = self.len;
        let b = a + n;
        if b > self.inf.len() {
            let new_len = b.max(self.inf.len() * 2);
            self.inf.resize(new_len, 0.0);
            self.dir_x.resize(new_len, 0.0);
            self.dir_y.resize(new_len, 0.0);
        }
        self.len = b;
        (
            &mut self.inf[a..b],
            &mut self.dir_x[a..b],
            &mut self.dir_y[a..b],
        )
    }

    /// Close out the current particle's entry range. Must be called once per
    /// particle, in particle order.
    #[inline]
    pub fn finish_particle(&mut self) {
        self.offsets.push(self.len as u32);
    }

    /// The cached entries for particle `p`, in the order they were written.
    #[inline]
    pub fn neighbours_of(&self, p: usize) -> Neighbours<'_> {
        let start = self.offsets[p] as usize;
        let end = self.offsets[p + 1] as usize;
        Neighbours {
            inf: &self.inf[start..end],
            dir_x: &self.dir_x[start..end],
            dir_y: &self.dir_y[start..end],
        }
    }
}

impl Default for NeighbourCache {
    fn default() -> Self {
        Self::new()
    }
}
