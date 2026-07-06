use glam::Vec2;

#[derive(Clone, Copy)]
pub struct CachedNeighbour {
    /// Sorted-grid slot (i.e. index into a grid's per-rebuild sorted
    /// buffers, such as `sorted_positions` / a caller's `sorted_densities`),
    /// NOT the neighbour's original particle index. Valid only until the
    /// next `UniformGrid::rebuild()` call, same lifetime as the sorted
    /// buffers it's meant to index into.
    pub n_slot: u32,
    pub inf: f32,
    pub dir_p_to_n: Vec2,
}

pub struct NeighbourCache {
    /// offsets[p] .. offsets[p + 1] is the entry range for particle p.
    /// Length is always particle_count + 1.
    offsets: Vec<u32>,
    entries: Vec<CachedNeighbour>,
}

impl NeighbourCache {
    pub fn new() -> Self {
        Self {
            offsets: vec![0],
            entries: Vec::new(),
        }
    }

    /// Preallocate for a known particle/neighbour count to avoid growth
    /// churn on the first few steps. `avg_neighbours` is a rough estimate
    pub fn with_capacity(particle_count: usize, avg_neighbours: usize) -> Self {
        let mut offsets = Vec::with_capacity(particle_count + 1);
        offsets.push(0);
        Self {
            offsets,
            entries: Vec::with_capacity(particle_count * avg_neighbours),
        }
    }

    /// Reset for a new pass, keeping allocated capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.offsets.clear();
        self.offsets.push(0);
        self.entries.clear();
    }

    /// Push a computed neighbour entry for the particle currently being
    /// built (call `finish_particle` after all of a particle's neighbours
    /// are pushed). `n_slot` is the neighbour's sorted-grid slot (see
    /// `CachedNeighbour::n_slot`), not its original particle index.
    #[inline]
    pub fn push(&mut self, n_slot: usize, inf: f32, dir_p_to_n: Vec2) {
        self.entries.push(CachedNeighbour {
            n_slot: n_slot as u32,
            inf,
            dir_p_to_n,
        });
    }

    /// Close out the current particle's neighbour range. Must be called
    /// once per particle, in particle index order, after all of that
    /// particle's `push` calls.
    #[inline]
    pub fn finish_particle(&mut self) {
        self.offsets.push(self.entries.len() as u32);
    }

    /// The cached neighbours for particle `p_indice`, in the same order
    /// they were pushed.
    #[inline]
    pub fn neighbours_of(&self, p_indice: usize) -> &[CachedNeighbour] {
        let start = self.offsets[p_indice] as usize;
        let end = self.offsets[p_indice + 1] as usize;
        &self.entries[start..end]
    }
}

impl Default for NeighbourCache {
    fn default() -> Self {
        Self::new()
    }
}
