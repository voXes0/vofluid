use glam::Vec2;
use std::ops::Range;

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
    sorted_x: Vec<f32>,
    sorted_y: Vec<f32>,
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
            sorted_x: vec![0.0; particle_count],
            sorted_y: vec![0.0; particle_count],
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
        crate::prof_span!("grid::rebuild");

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

        for (i, pos) in positions.iter().enumerate() {
            let cell = self.particle_cells[i];
            if cell == usize::MAX {
                continue;
            }
            let slot = self.cursors[cell] as usize;
            self.particle_index[slot] = i;
            self.sorted_x[slot] = pos.x;
            self.sorted_y[slot] = pos.y;
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
        self.sorted_x.resize(new_particle_count, 0.0);
        self.sorted_y.resize(new_particle_count, 0.0);
    }

    /// Sorted positions as separate x / y arrays, in the same slot order as
    /// `particle_index()`. Valid until the next `rebuild()`.
    #[inline]
    pub fn sorted_xy(&self) -> (&[f32], &[f32]) {
        (&self.sorted_x, &self.sorted_y)
    }

    /// One contiguous sorted-slot range per stencil row around `pos`.
    ///
    /// Cells are numbered `cx + cy * grid_width` and particles are sorted by
    /// cell, so all cells of one stencil row are consecutive in sorted order:
    /// each row is a single `start..end` range of slots. Stencil width is
    /// derived from `radius` exactly as in `query_neighbours`. These are
    /// candidates only; the caller still decides which are within `radius`.
    /// Yields nothing if `pos` is outside the grid.
    #[inline]
    pub fn stencil_rows(
        &self,
        pos: Vec2,
        radius: f32,
    ) -> impl Iterator<Item = Range<usize>> + Clone + '_ {
        let (x_min, x_max, y_min, y_max) = match self.cell_coords(pos) {
            Some((ix, iy)) => {
                let r = (radius * self.cell_size_inv).ceil() as usize;
                (
                    ix.saturating_sub(r),
                    (ix + r).min(self.grid_width - 1),
                    iy.saturating_sub(r),
                    (iy + r).min(self.grid_height - 1),
                )
            }
            // Empty y range: the iterator yields nothing.
            None => (0, 0, 1, 0),
        };
        (y_min..=y_max).map(move |cy| {
            let row = cy * self.grid_width;
            self.cell_starts[row + x_min] as usize..self.cell_starts[row + x_max + 1] as usize
        })
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

    /// The sorted-order permutation from the last `rebuild()`: slot `k`
    /// corresponds to original particle index `particle_index()[k]`. Lets
    /// callers build their own sorted-order scratch buffers (e.g. a
    /// `sorted_densities`) that stay consistent with `sorted_positions` and
    /// with slots returned by `query_neighbours_sorted`.
    #[inline]
    pub fn particle_index(&self) -> &[usize] {
        &self.particle_index
    }

    #[inline]
    fn cell_of(&self, pos: Vec2) -> Option<usize> {
        self.cell_coords(pos)
            .map(|(ix, iy)| ix + iy * self.grid_width)
    }
}
