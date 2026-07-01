use glam::Vec2;

pub struct UniformGrid {
    particle_index: Vec<usize>,
    cell_starts: Vec<u32>,
    cursors: Vec<u32>,
    pub grid_width: usize,
    pub grid_height: usize,
    pub cell_size: f32,
    cell_size_inv: f32,
    origin: Vec2,
    cell_count: usize,
    particle_cells: Vec<usize>, // Cached cells set every rebuild, for only allocating at grid build
}

const GRID_BOUNDS_PADDING: f32 = 1.5;

impl UniformGrid {
    pub fn new(bounds: Vec2, cell_size: f32, particle_count: usize) -> Self {
        let origin = Vec2::new(-bounds[0], -bounds[1]);
        let grid_bounds = bounds * GRID_BOUNDS_PADDING;
        let grid_width = (grid_bounds.x * 2.0 / cell_size).ceil() as usize + 1;
        let grid_height = (grid_bounds.y * 2.0 / cell_size).ceil() as usize + 1;
        let cell_count = grid_width * grid_height;
        Self {
            particle_cells: vec![0usize; particle_count],
            particle_index: vec![0usize; particle_count],
            cell_starts: vec![0u32; cell_count + 1],
            cursors: vec![0u32; cell_count],
            grid_width,
            grid_height,
            cell_count,
            cell_size,
            cell_size_inv: 1.0 / cell_size,
            origin,
        }
    }

    pub fn rebuild(&mut self, positions: &[Vec2]) {
        let cell_count = self.cell_count;
        // ── step 1: zero counts and count particles per cell ──────────────────
        self.cell_starts.fill(0);

        for (i, pos) in positions.iter().enumerate() {
            match self.cell_of(*pos) {
                Some(cell) => {
                    self.particle_cells[i] = cell;
                    self.cell_starts[cell] += 1;
                }
                None => {
                    self.particle_cells[i] = usize::MAX;
                }
            }
        }

        // ── step 2: prefix sum ────────────────────────────────────────────────
        let mut total = 0u32;
        for c in 0..cell_count {
            let count = self.cell_starts[c];
            self.cell_starts[c] = total;
            total += count;
        }
        self.cell_starts[cell_count] = total;

        // ── step 3: fill particle_index ───────────────────────────────────────
        // Copy start indices into cursors without allocating.
        self.cursors
            .copy_from_slice(&self.cell_starts[..cell_count]);

        for (i, _) in positions.iter().enumerate() {
            let cell = self.particle_cells[i];
            if cell == usize::MAX {
                continue;
            }
            self.particle_index[self.cursors[cell] as usize] = i;
            self.cursors[cell] += 1;
        }
    }

    pub fn resize_bounds(&mut self, bounds: Vec2) {
        let origin = Vec2::new(-bounds[0], -bounds[1]);
        let grid_bounds = bounds * GRID_BOUNDS_PADDING;
        let grid_width = (grid_bounds.x * 2.0 / self.cell_size).ceil() as usize + 1;
        let grid_height = (grid_bounds.y * 2.0 / self.cell_size).ceil() as usize + 1;
        let cell_count = grid_width * grid_height;

        self.origin = origin;
        self.grid_width = grid_width;
        self.grid_height = grid_height;
        self.cell_count = cell_count;
        self.cell_starts.resize(cell_count + 1, 0);
        self.cursors.resize(cell_count, 0);
    }

    pub fn resize_particles(&mut self, new_particle_count: usize) {
        self.particle_index.resize(new_particle_count, 0);
        self.particle_cells.resize(new_particle_count, 0);
    }

    /// Calls `f(j)` for every particle index `j` in the 3×3 cell neighbourhood
    /// of `pos`. The caller is responsible for the distance check.
    #[inline]
    pub fn query_neighbours(&self, pos: Vec2, mut f: impl FnMut(usize)) {
        let Some((ix, iy)) = self.cell_coords(pos) else {
            return;
        };
        let x_min = ix.saturating_sub(1);
        let y_min = iy.saturating_sub(1);
        let x_max = (ix + 1).min(self.grid_width - 1);
        let y_max = (iy + 1).min(self.grid_height - 1);
        for cy in y_min..=y_max {
            for cx in x_min..=x_max {
                let cell = cx + cy * self.grid_width;
                let start = self.cell_starts[cell] as usize;
                let end = self.cell_starts[cell + 1] as usize;
                for &idx in &self.particle_index[start..end] {
                    f(idx);
                }
            }
        }
    }

    #[inline]
    fn cell_coords(&self, pos: Vec2) -> Option<(usize, usize)> {
        let local = pos - self.origin;
        if local.x < 0.0 || local.y < 0.0 {
            return None;
        }
        let ix = local.x * self.cell_size_inv;
        let iy = local.y * self.cell_size_inv;
        if ix >= self.grid_width as f32 || iy >= self.grid_height as f32 {
            return None;
        }
        Some((ix as usize, iy as usize))
    }

    #[inline]
    fn cell_of(&self, pos: Vec2) -> Option<usize> {
        self.cell_coords(pos)
            .map(|(ix, iy)| ix + iy * self.grid_width)
    }
}
