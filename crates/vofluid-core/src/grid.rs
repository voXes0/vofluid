//! Uniform grid for neighbour queries, rebuilt every substep with a counting sort.
//!
//! Space is cut into square cells, numbered row by row: `cell = cx + cy * grid_width`.
//! [`UniformGrid::rebuild`] sorts the particles by cell into *slots*, so slot order
//! follows cell order:
//!
//! ```text
//! slots:  | cell 0 | cell 1 | cell 2 | cell 3 | ...      one packed array
//!           ^        ^        ^
//!           cell_starts[0]    cell_starts[2]
//!
//! cell `c` owns the slots cell_starts[c] .. cell_starts[c + 1]
//! ```
//!
//! Cells of one grid row are consecutive, so any run of cells in a row is one
//! contiguous slot range. The neighbourhood of a cell is therefore one range per
//! grid row, a [`Stencil`], and no per-particle neighbour lists are ever stored.
//!
//! Sorted positions are kept as separate x / y arrays so SIMD kernels can load
//! them contiguously. Every slot-indexed array must be [`SORTED_PAD`] slots
//! longer than the particle count.

use glam::Vec2;
use std::ops::Range;

/// The grid covers `GRID_BOUNDS_PADDING` times the simulation bounds, so
/// particles that overshoot the bounds before being clamped stay inside it.
const GRID_BOUNDS_PADDING: f32 = 1.2;

/// `particle_cells` value for a particle outside the grid.
const OUTSIDE: usize = usize::MAX;

/// Extra slots at the end of every slot-indexed array. A SIMD kernel loads full
/// vectors that may start in the last few slots and run past the last particle.
/// `rebuild` never writes the padding and slots after the last sorted particle
/// keep stale values, so kernels must mask lanes beyond the end of a stencil row.
pub const SORTED_PAD: usize = 8;

pub struct UniformGrid {
    cell_size: f32,
    cell_size_inv: f32,
    /// World position of cell (0, 0)'s corner.
    origin: Vec2,
    grid_width: usize,
    grid_height: usize,
    cell_count: usize,
    /// First slot of each cell, plus a final entry: the number of particles
    /// inside the grid. Cell `c` owns `cell_starts[c]..cell_starts[c + 1]`.
    cell_starts: Vec<u32>,
    /// Scratch for the scatter step of `rebuild`: next free slot per cell.
    /// A copy of `cell_starts` that advances, since `cell_starts` must stay intact.
    cursors: Vec<u32>,
    /// Scratch: cell of each particle, `OUTSIDE` if none.
    particle_cells: Vec<usize>,
    /// Slot -> original particle index.
    particle_index: Vec<usize>,
    sorted_x: Vec<f32>,
    sorted_y: Vec<f32>,
}

/// Reusable buffer for one stencil: one slot range per grid row around a cell.
/// Filled by [`UniformGrid::load_stencil`].
#[derive(Default)]
pub struct Stencil {
    rows: Vec<Range<usize>>,
}

impl Stencil {
    /// The slot ranges, one per stencil row. They are candidates only: the
    /// caller decides which lie within the interaction radius.
    #[inline]
    pub fn rows(&self) -> &[Range<usize>] {
        &self.rows
    }
}

impl UniformGrid {
    pub fn new(bounds: Vec2, cell_size: f32, particle_count: usize) -> Self {
        let (origin, grid_width, grid_height) = layout(bounds, cell_size);
        let cell_count = grid_width * grid_height;
        Self {
            cell_size,
            cell_size_inv: 1.0 / cell_size,
            origin,
            grid_width,
            grid_height,
            cell_count,
            cell_starts: vec![0; cell_count + 1],
            cursors: vec![0; cell_count],
            particle_cells: vec![0; particle_count],
            particle_index: vec![0; particle_count],
            sorted_x: vec![0.0; particle_count + SORTED_PAD],
            sorted_y: vec![0.0; particle_count + SORTED_PAD],
        }
    }

    /// Re-sorts `positions` into slots (counting sort). Particles outside the
    /// grid get no slot and are left out of every neighbour query.
    pub fn rebuild(&mut self, positions: &[Vec2]) {
        crate::prof_span!("grid::rebuild");

        let cell_count = self.cell_count;
        #[cfg(feature = "profile")]
        let mut out_of_grid = 0u64;

        // 1. Count the particles in each cell (`cell_starts` doubles as the counters).
        {
            crate::prof_span!("grid::rebuild::count_cells");
            self.cell_starts.fill(0);
            for (i, pos) in positions.iter().enumerate() {
                match self.cell_of(*pos) {
                    Some(cell) => {
                        self.particle_cells[i] = cell;
                        self.cell_starts[cell] += 1;
                    }
                    None => {
                        self.particle_cells[i] = OUTSIDE;
                        #[cfg(feature = "profile")]
                        {
                            out_of_grid += 1;
                        }
                    }
                }
            }
        }

        // 2. Prefix sum: each count becomes the sum of all counts before it,
        //    which is the first slot of that cell.
        {
            crate::prof_span!("grid::rebuild::prefix_sum");
            let mut total = 0u32;
            for start in &mut self.cell_starts[..cell_count] {
                let count = *start;
                *start = total;
                total += count;
            }
            self.cell_starts[cell_count] = total;
        }

        // 3. Scatter: put each particle in the next free slot of its cell.
        {
            crate::prof_span!("grid::rebuild::scatter");
            self.cursors
                .copy_from_slice(&self.cell_starts[..cell_count]);
            for (i, pos) in positions.iter().enumerate() {
                let cell = self.particle_cells[i];
                if cell == OUTSIDE {
                    continue;
                }
                let slot = self.cursors[cell] as usize;
                self.particle_index[slot] = i;
                self.sorted_x[slot] = pos.x;
                self.sorted_y[slot] = pos.y;
                self.cursors[cell] += 1;
            }
        }

        // The prefix sum scales with cells, the other two steps with particles.
        crate::prof_count!("grid::rebuild::cells", cell_count);
        crate::prof_count!("grid::rebuild::out_of_grid", out_of_grid);
    }

    pub fn resize_bounds(&mut self, bounds: Vec2) {
        let (origin, grid_width, grid_height) = layout(bounds, self.cell_size);
        self.origin = origin;
        self.grid_width = grid_width;
        self.grid_height = grid_height;
        self.cell_count = grid_width * grid_height;
        self.cell_starts.resize(self.cell_count + 1, 0);
        self.cursors.resize(self.cell_count, 0);
    }

    pub fn resize_particles(&mut self, particle_count: usize) {
        self.particle_cells.resize(particle_count, 0);
        self.particle_index.resize(particle_count, 0);
        self.sorted_x.resize(particle_count + SORTED_PAD, 0.0);
        self.sorted_y.resize(particle_count + SORTED_PAD, 0.0);
    }

    /// Sorted x / y positions in slot order, valid until the next `rebuild`.
    /// Both are `SORTED_PAD` slots longer than the particle count.
    #[inline]
    pub fn sorted_xy(&self) -> (&[f32], &[f32]) {
        (&self.sorted_x, &self.sorted_y)
    }

    /// Slot -> original particle index, from the last `rebuild`. Only the first
    /// `n` entries are current, where `n` is the number of particles inside the
    /// grid; the rest are stale.
    #[inline]
    pub fn particle_index(&self) -> &[usize] {
        &self.particle_index
    }

    /// Every non-empty cell as `(cx, cy, slots)`, in slot order. The slot ranges
    /// are consecutive and cover exactly the particles inside the grid. All
    /// particles of a cell share one stencil, so load it once per cell.
    #[inline]
    pub fn occupied_cells(&self) -> impl Iterator<Item = (usize, usize, Range<usize>)> + '_ {
        (0..self.grid_height).flat_map(move |cy| {
            (0..self.grid_width).filter_map(move |cx| {
                let cell = cx + cy * self.grid_width;
                let slots = self.cell_starts[cell] as usize..self.cell_starts[cell + 1] as usize;
                (!slots.is_empty()).then_some((cx, cy, slots))
            })
        })
    }

    /// Loads into `out` the stencil of cell `(cx, cy)`: the block of cells within
    /// `radius` (rounded up to whole cells, clipped to the grid), as one slot
    /// range per grid row.
    #[inline]
    pub fn load_stencil(&self, out: &mut Stencil, cx: usize, cy: usize, radius: f32) {
        out.rows.clear();
        out.rows.extend(self.stencil_rows_of(cx, cy, radius));
    }
    #[inline]
    fn stencil_rows_of(
        &self,
        cx: usize,
        cy: usize,
        radius: f32,
    ) -> impl Iterator<Item = Range<usize>> + Clone + '_ {
        // Block of cells around (cx, cy), clipped to the grid:
        // `saturating_sub` stops at 0, `min` stops at the last column / row.
        let reach = (radius * self.cell_size_inv).ceil() as usize;
        let x_min = cx.saturating_sub(reach);
        let x_max = (cx + reach).min(self.grid_width - 1);
        let y_min = cy.saturating_sub(reach);
        let y_max = (cy + reach).min(self.grid_height - 1);
        // One slot range per grid row of the block: from the first slot of its
        // leftmost cell to the first slot of the cell just after its rightmost.
        (y_min..=y_max).map(move |y| {
            let row = y * self.grid_width;
            self.cell_starts[row + x_min] as usize..self.cell_starts[row + x_max + 1] as usize
        })
    }

    /// Column and row of the cell containing `pos`, `None` outside the grid.
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

    /// Flat cell number (`cx + cy * grid_width`) containing `pos`, `None` outside the grid.
    #[inline]
    fn cell_of(&self, pos: Vec2) -> Option<usize> {
        self.cell_coords(pos)
            .map(|(ix, iy)| ix + iy * self.grid_width)
    }
}

/// Grid origin and size in cells for simulation `bounds` (`[-bounds, bounds]`).
fn layout(bounds: Vec2, cell_size: f32) -> (Vec2, usize, usize) {
    let grid_bounds = bounds * GRID_BOUNDS_PADDING;
    let width = (grid_bounds.x * 2.0 / cell_size).ceil() as usize + 1;
    let height = (grid_bounds.y * 2.0 / cell_size).ceil() as usize + 1;
    (-grid_bounds, width, height)
}
