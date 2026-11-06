//! Profile-only bookkeeping for a pass over the particles: work counters and
//! sampled per-particle timing (`profiling_hw::PhaseSampler`, one particle in
//! `SAMPLE_EVERY`).
//!
//! Without the `profile` feature `PassProbe` is a zero-sized no-op, so the
//! passes need no `cfg` of their own.

/// Names a pass reports under.
#[allow(dead_code)] // only read with the `profile` feature
pub(super) struct PassLabels {
    /// Sampled per-particle timing.
    pub particle: &'static str,
    /// Work counters: stencil candidates, stencil rows and SIMD chunks
    /// visited. `candidates / (LANES * chunks)` is the lane utilization.
    pub candidates: &'static str,
    pub rows: &'static str,
    pub chunks: &'static str,
    /// If set, real neighbours are also counted on the sampled particles
    /// (after the timed part): `(candidates_sampled, neighbours_sampled)`.
    /// Their ratio is the share of candidates inside the interaction radius.
    pub sampled: Option<(&'static str, &'static str)>,
}

#[cfg(not(feature = "profile"))]
pub(super) use disabled::PassProbe;
#[cfg(feature = "profile")]
pub(super) use enabled::PassProbe;

#[cfg(feature = "profile")]
mod enabled {
    use super::PassLabels;
    use crate::backends::single::simd::LANES;
    use crate::profiling_hw::{PhaseSampler, Tick as HwTick};
    use std::ops::Range;

    #[derive(Clone, Copy, Default)]
    struct Work {
        candidates: u64,
        rows: u64,
        chunks: u64,
    }

    pub struct Tick(HwTick);

    pub struct PassProbe {
        labels: &'static PassLabels,
        sampler: PhaseSampler<1>,
        /// Work per particle of the current cell (all share one stencil).
        cell: Work,
        total: Work,
        sampled_candidates: u64,
        sampled_neighbours: u64,
    }

    impl PassProbe {
        pub fn new(labels: &'static PassLabels) -> Self {
            Self {
                labels,
                sampler: PhaseSampler::new([labels.particle]),
                cell: Work::default(),
                total: Work::default(),
                sampled_candidates: 0,
                sampled_neighbours: 0,
            }
        }

        /// Call when a cell's stencil is loaded.
        pub fn cell(&mut self, rows: &[Range<usize>]) {
            self.cell = Work {
                candidates: rows.iter().map(|r| r.len() as u64).sum(),
                rows: rows.len() as u64,
                chunks: rows.iter().map(|r| r.len().div_ceil(LANES) as u64).sum(),
            };
        }

        /// Call before a particle's work.
        pub fn begin(&mut self) -> Tick {
            Tick(self.sampler.start())
        }

        /// Call after a particle's work.
        pub fn end(&mut self, mut tick: Tick) {
            self.sampler
                .lap(&mut tick.0, 0, self.cell.candidates as f64);
            self.total.candidates += self.cell.candidates;
            self.total.rows += self.cell.rows;
            self.total.chunks += self.cell.chunks;
        }

        /// Like `end`; on sampled particles also calls `neighbours` (untimed)
        /// for the number of candidates inside the interaction radius.
        pub fn end_counting(&mut self, tick: Tick, neighbours: impl FnOnce() -> u64) {
            let sampled = tick.0.is_sampled();
            self.end(tick);
            if sampled {
                self.sampled_candidates += self.cell.candidates;
                self.sampled_neighbours += neighbours();
            }
        }

        pub fn finish(self) {
            self.sampler.finish();
            let mut report = vec![
                (self.labels.candidates, self.total.candidates),
                (self.labels.rows, self.total.rows),
                (self.labels.chunks, self.total.chunks),
            ];
            if let Some((candidates, neighbours)) = self.labels.sampled {
                report.push((candidates, self.sampled_candidates));
                report.push((neighbours, self.sampled_neighbours));
            }
            for &(label, value) in &report {
                crate::prof_count!(label, value);
            }
        }
    }
}

#[cfg(not(feature = "profile"))]
mod disabled {
    use super::PassLabels;
    use std::ops::Range;

    pub struct Tick;

    pub struct PassProbe;

    impl PassProbe {
        #[inline(always)]
        pub fn new(_: &'static PassLabels) -> Self {
            Self
        }
        #[inline(always)]
        pub fn cell(&mut self, _: &[Range<usize>]) {}
        #[inline(always)]
        pub fn begin(&mut self) -> Tick {
            Tick
        }
        #[inline(always)]
        pub fn end(&mut self, _: Tick) {}
        #[inline(always)]
        pub fn end_counting(&mut self, _: Tick, _: impl FnOnce() -> u64) {}
        #[inline(always)]
        pub fn finish(self) {}
    }
}
