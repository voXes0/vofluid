//! Backend-agnostic, in-process profiling.
//!
//! This module owns *measurement* only: timing a labeled section of code and
//! handing the resulting `(label, duration)` (or aggregated batch) to
//! whatever `Sink` is currently installed. It knows nothing about GUIs,
//! files, or formats — see `vofluid-app`'s `profiling_sink` /
//! `profiling_gui` / `profiling_export` for the consumers that store
//! samples, draw a live graph, and write an exported report.
//!
//! Everything here compiles to nothing when the `profile` feature is off:
//! `prof_span!` expands to a unit statement, `record`/`record_batch` are
//! never called, and the sink machinery isn't linked in.
//!
//! # Usage — moderate-frequency spans
//! Most call sites (once up to a few hundred times per frame) should use
//! `prof_span!`, which times the rest of the enclosing scope and records one
//! sample on drop:
//! ```ignore
//! fn rebuild(&mut self, positions: &[Vec2]) {
//!     crate::prof_span!("grid::rebuild");
//!     // ... work ...
//! } // duration recorded here
//! ```
//!
//! # Usage — hot per-item loops
//! Per-particle bodies run tens of thousands of times per frame at high
//! particle counts. Timing each one individually would mean tens of
//! thousands of `Instant::now()` reads per frame, which costs real time and
//! inflates the very numbers being measured. Nothing is instrumented that
//! finely today (the passes time their whole loop with one `prof_span!`);
//! `SpanAggregator` is the tool if something ever needs to be:
//! accumulate count/sum/min/max locally across the loop, then hand the sink
//! one aggregated sample instead of one sample per iteration.
//! ```ignore
//! let mut agg = SpanAggregator::new();
//! for p in &particles {
//!     let t0 = std::time::Instant::now();
//!     do_work(p);
//!     agg.add(t0.elapsed());
//! }
//! agg.finish("grid::query_neighbours_sorted");
//! ```
//!
//! # Counters
//! `prof_count!("a::b::unit", n)` adds `n` to a named counter: how much work
//! a span did (candidates tested, neighbours found), as opposed to how long
//! it took. A counter named `<span label>::<unit>` belongs to that span and
//! to every span nested under it, and must be emitted once per call of the
//! span it belongs to, so a consumer can divide time by work (ns per
//! neighbour). That separates "the code got slower" from "the fluid got
//! denser", which timings alone cannot.
//!
//! # Steps and epochs
//! `Simulation::step` brackets every step with a `StepGuard`, so a sink knows
//! where one step ends and the next begins (and the wall time of the whole
//! step, including code no span covers). It also calls `begin_epoch` whenever
//! `SimConfig` differs from the previous step, so a sink can keep statistics
//! per configuration instead of blending, say, 2000 and 4500 particles into
//! one average.
//!
//! # Hardware telemetry and sampled fits (`profile` builds only)
//! A step's wall time depends on which core and clock it got as much as on
//! the code: on a hybrid CPU a step on an efficiency core, or at a throttled
//! clock, is simply slower. `StepGuard` therefore hands the sink a `HwSample`
//! per step (the core type, and every few steps an effective-clock probe) so
//! a consumer can tell machine effects from code effects. The platform code
//! lives in `profiling_hw`.
//!
//! Some loop bodies are far too short to time one by one (about 75 ns of
//! work against a timer read of tens of ns). `profiling_hw::PhaseSampler`
//! times one particle in `SAMPLE_EVERY` with a fenced timestamp-counter read
//! and accumulates `(work, ns)` sums locally (`FitSums`, the per-item
//! analogue of `SpanAggregator`), sending one batch per call through
//! `Sink::fit`. A consumer then fits `ns = overhead + per_work * work`.
//!
//! # Threading
//! The simulation itself is single-threaded today, so there's no lock
//! contention to speak of in `Sink::record` as currently called. `Sink` is
//! still specified as `Send + Sync` so a future parallel backend, or an
//! app-side export happening on another thread, can install or read from a
//! sink safely without this module needing to change.
//!
//! # Swapping sinks
//! `vofluid-app` installs an `InMemorySink` at startup (see
//! `profiling::install_sink`). Any other consumer of `vofluid-core` — a
//! benchmark harness, a headless batch runner — can install its own `Sink`
//! implementation instead (or install none at all, which discards every
//! sample for ~free via `NullSink`).

use crate::config::SimConfig;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

/// One completed span, or one batch of same-label spans aggregated together
/// (see `SpanAggregator`). `count` is `1` for an individual `ScopeGuard`
/// sample and `>= 1` for a batch.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub label: &'static str,
    pub count: u32,
    pub total: Duration,
    pub min: Duration,
    pub max: Duration,
}

/// Which kind of core a step ran on. Only hybrid Intel CPUs report it;
/// everything else is `Unknown`. `Mixed` means the thread moved between core
/// types during the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreType {
    Unknown,
    Performance,
    Efficiency,
    Mixed,
}

/// Hardware telemetry for one simulation step, taken just after the step
/// (so it is not part of the step's own wall time).
#[derive(Clone, Copy, Debug)]
pub struct HwSample {
    pub core_type: CoreType,
    /// Effective core clock from a short probe loop, in MHz. `None` on steps
    /// that weren't probed, and where the probe isn't supported.
    pub clock_mhz: Option<f32>,
}

/// Running sums of `(x, y)` pairs: enough to fit `y = a + b * x` by least
/// squares (and its r²) later, without storing the pairs.
#[derive(Clone, Copy, Debug, Default)]
pub struct FitSums {
    pub n: u64,
    pub sx: f64,
    pub sy: f64,
    pub sxx: f64,
    pub sxy: f64,
    pub syy: f64,
}

/// Least-squares line `y = intercept + slope * x` with standard errors and r²
/// (see `FitSums::line`).
#[derive(Clone, Copy, Debug)]
pub struct LineFit {
    pub intercept: f64,
    pub slope: f64,
    pub intercept_se: f64,
    pub slope_se: f64,
    pub r2: f64,
}

impl FitSums {
    pub fn mean_x(&self) -> f64 {
        self.sx / self.n.max(1) as f64
    }

    pub fn mean_y(&self) -> f64 {
        self.sy / self.n.max(1) as f64
    }

    /// Fits `y = a + b*x`. `None` with fewer than 3 points or when every `x`
    /// is equal (the slope is then undefined). When `x` spans a narrow range
    /// the intercept is an extrapolation: read its standard error.
    pub fn line(&self) -> Option<LineFit> {
        if self.n < 3 {
            return None;
        }
        let n = self.n as f64;
        // Centred sums of squares / products.
        let sxx = (self.sxx - self.sx * self.sx / n).max(0.0);
        let sxy = self.sxy - self.sx * self.sy / n;
        let syy = (self.syy - self.sy * self.sy / n).max(0.0);
        if sxx <= f64::EPSILON * self.sxx.abs().max(1.0) {
            return None;
        }
        let slope = sxy / sxx;
        let intercept = (self.sy - slope * self.sx) / n;
        let sse = (syy - slope * sxy).max(0.0);
        let variance = sse / (n - 2.0);
        let mean_x = self.sx / n;
        Some(LineFit {
            intercept,
            slope,
            intercept_se: (variance * (1.0 / n + mean_x * mean_x / sxx)).sqrt(),
            slope_se: (variance / sxx).sqrt(),
            r2: if syy > 0.0 { 1.0 - sse / syy } else { 1.0 },
        })
    }

    #[inline]
    pub fn add(&mut self, x: f64, y: f64) {
        self.n += 1;
        self.sx += x;
        self.sy += y;
        self.sxx += x * x;
        self.sxy += x * y;
        self.syy += y * y;
    }

    pub fn merge(&mut self, other: &FitSums) {
        self.n += other.n;
        self.sx += other.sx;
        self.sy += other.sy;
        self.sxx += other.sxx;
        self.sxy += other.sxy;
        self.syy += other.syy;
    }
}

/// Receives one `Sample` at a time as spans (or aggregated batches) complete.
///
/// Implementations should be cheap: `record` can be called from inside the
/// simulation loop, potentially every frame per label. See the module-level
/// "Threading" note for the current concurrency assumptions.
pub trait Sink: Send + Sync {
    fn record(&self, sample: Sample);

    /// Adds `value` to the counter `label` (see the module docs, "Counters").
    fn count(&self, _label: &'static str, _value: u64) {}

    /// The simulation configuration differs from the previous step's (or this
    /// is the first step). Everything recorded from the next step on was
    /// measured under `config`.
    fn begin_epoch(&self, _config: &SimConfig) {}

    /// A `Simulation::step` is about to run. Samples and counts recorded
    /// until `step_end` belong to that step.
    fn step_begin(&self) {}

    /// The step finished; `total` is its whole wall time.
    fn step_end(&self, _total: Duration) {}

    /// Hardware telemetry of the step that is about to end. Called by
    /// `StepGuard` right before `step_end`, so it belongs to the same step.
    fn hardware(&self, _sample: HwSample) {}

    /// A batch of `(work, ns)` sums for the sampled label `label` (see the
    /// module docs, "Hardware telemetry and sampled fits"). Batches for one
    /// label add up.
    fn fit(&self, _label: &'static str, _sums: FitSums) {}
}

/// Discards every sample. Installed by default so that `vofluid-core` never
/// depends on `vofluid-app`, and so profiling has near-zero cost when no one
/// has installed a real sink (e.g. running `vofluid-core` standalone, in
/// tests, or in a benchmark harness that doesn't care about live stats).
struct NullSink;
impl Sink for NullSink {
    #[inline(always)]
    fn record(&self, _sample: Sample) {}
}

static SINK: OnceLock<Arc<dyn Sink>> = OnceLock::new();

/// Installs the process-wide profiling sink. Call once, before spinning up
/// the simulation (e.g. at the top of `main`). Returns an `Arc` handle to
/// the *same* sink instance that was just installed, so the caller can read
/// its own sink back later (e.g. `vofluid-app`'s `InMemorySink`, read every
/// frame to draw a live graph) instead of only being able to write into it
/// via `record`/`prof_span!`.
///
/// # Panics
/// Panics if called more than once (per process). `OnceLock` has no safe
/// way to swap its contents after the fact, and silently returning a handle
/// to a *different* sink than the one just passed in would be worse than
/// failing loudly — the caller would read back a sink nothing is actually
/// writing to. Call this exactly once, at startup, before the simulation
/// runs any steps.
///
/// If never called at all, `record`/`prof_span!` fall back to `NullSink`.
pub fn install_sink<S: Sink + 'static>(sink: S) -> Arc<S> {
    let arc = Arc::new(sink);
    SINK.set(arc.clone())
        .unwrap_or_else(|_| panic!("profiling::install_sink called more than once"));
    arc
}

#[inline]
fn sink() -> &'static dyn Sink {
    SINK.get_or_init(|| Arc::new(NullSink)).as_ref()
}

/// Records a single completed span of `duration` under `label`. Normally
/// called via `prof_span!`'s drop guard rather than directly.
#[inline]
pub fn record(label: &'static str, duration: Duration) {
    sink().record(Sample {
        label,
        count: 1,
        total: duration,
        min: duration,
        max: duration,
    });
}

/// Adds `value` to the counter `label`. Normally called via `prof_count!`.
#[inline]
pub fn count(label: &'static str, value: u64) {
    sink().count(label, value);
}

/// Tells the installed sink that the configuration changed. Called by
/// `Simulation::step`; nothing else should need it.
#[inline]
pub fn begin_epoch(config: &SimConfig) {
    sink().begin_epoch(config);
}

/// Sends a batch of sampled `(work, ns)` sums for `label`. Called by
/// `profiling_hw::PhaseSampler::finish`.
#[inline]
pub fn fit(label: &'static str, sums: FitSums) {
    sink().fit(label, sums);
}

/// RAII scope guard: starts timing on creation, records on drop. Only
/// intended to be constructed via `prof_span!`.
#[must_use = "a profiling span does nothing if immediately dropped; bind it, e.g. `let _s = prof_span!(...)`"]
pub struct ScopeGuard {
    label: &'static str,
    start: std::time::Instant,
}

impl ScopeGuard {
    #[inline]
    pub fn new(label: &'static str) -> Self {
        Self {
            label,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for ScopeGuard {
    #[inline]
    fn drop(&mut self) {
        record(self.label, self.start.elapsed());
    }
}

/// RAII guard that brackets one whole simulation step for the installed
/// sink: `step_begin` on creation; on drop the elapsed wall time goes to
/// `step_end`, preceded by the step's hardware telemetry. Only
/// `Simulation::step` creates it, and only in `profile` builds.
#[cfg(feature = "profile")]
#[must_use = "a step guard does nothing if immediately dropped; bind it, e.g. `let _step = StepGuard::new()`"]
pub struct StepGuard {
    start: std::time::Instant,
    probe: crate::profiling_hw::StepProbe,
}

#[cfg(feature = "profile")]
impl StepGuard {
    #[inline]
    pub fn new() -> Self {
        sink().step_begin();
        // The first use calibrates the timestamp counter (a one-off ~2 ms
        // spin), so it must happen before the step's clock starts.
        let probe = crate::profiling_hw::StepProbe::start();
        Self {
            start: std::time::Instant::now(),
            probe,
        }
    }
}

#[cfg(feature = "profile")]
impl Drop for StepGuard {
    #[inline]
    fn drop(&mut self) {
        // Wall time first: the telemetry below is not part of the step.
        let total = self.start.elapsed();
        sink().hardware(self.probe.finish());
        sink().step_end(total);
    }
}

/// Times the remainder of the enclosing scope under `label`, recording the
/// elapsed duration to the installed `Sink` on drop.
///
/// Compiles to nothing when the `profile` feature is disabled, so call
/// sites don't need their own `#[cfg(feature = "profile")]` guard.
///
/// Do NOT use this inside a loop that runs more than roughly a few hundred
/// times per frame (e.g. once per particle) — use `SpanAggregator` there
/// instead, so the profiler doesn't itself become the bottleneck it's
/// trying to measure.
#[macro_export]
macro_rules! prof_span {
    ($label:expr) => {
        #[cfg(feature = "profile")]
        let _prof_guard = $crate::profiling::ScopeGuard::new($label);
    };
}

/// Adds `value` (anything `as u64` accepts) to the counter `label`; see the
/// module docs, "Counters", for the naming rule.
///
/// Compiles to nothing when the `profile` feature is disabled, and `value`
/// is then not even evaluated, so it may refer to variables that only exist
/// under `#[cfg(feature = "profile")]`.
#[macro_export]
macro_rules! prof_count {
    ($label:expr, $value:expr) => {
        #[cfg(feature = "profile")]
        $crate::profiling::count($label, ($value) as u64);
    };
}

/// Accumulates many short timings under one label into a single aggregated
/// `Sample`, for loops that run too often per frame to time (and record)
/// individually — e.g. once per particle.
///
/// Has no-op stand-ins when the `profile` feature is off, so call sites
/// don't need their own `#[cfg(feature = "profile")]` guard around
/// construction/`add`/`finish` — just around the `Instant::now()` timing
/// calls themselves if those aren't free to compute unconditionally.
pub struct SpanAggregator {
    #[cfg(feature = "profile")]
    count: u32,
    #[cfg(feature = "profile")]
    total: Duration,
    #[cfg(feature = "profile")]
    min: Duration,
    #[cfg(feature = "profile")]
    max: Duration,
}

impl SpanAggregator {
    #[inline]
    pub fn new() -> Self {
        #[cfg(feature = "profile")]
        {
            Self {
                count: 0,
                total: Duration::ZERO,
                min: Duration::MAX,
                max: Duration::ZERO,
            }
        }
        #[cfg(not(feature = "profile"))]
        {
            Self {}
        }
    }

    /// Records one iteration's duration into the running aggregate. Cheap:
    /// no allocation, no lock, no call into the sink — just three
    /// comparisons and an add.
    #[inline]
    #[allow(unused_variables)]
    pub fn add(&mut self, duration: Duration) {
        #[cfg(feature = "profile")]
        {
            self.count += 1;
            self.total += duration;
            self.min = self.min.min(duration);
            self.max = self.max.max(duration);
        }
    }

    /// Hands the accumulated aggregate to the installed sink as one
    /// `Sample`, under `label`. No-op if `add` was never called.
    #[inline]
    #[allow(unused_variables)]
    pub fn finish(self, label: &'static str) {
        #[cfg(feature = "profile")]
        {
            if self.count > 0 {
                sink().record(Sample {
                    label,
                    count: self.count,
                    total: self.total,
                    min: self.min,
                    max: self.max,
                });
            }
        }
    }
}

impl Default for SpanAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_recovers_an_exact_relationship() {
        let mut sums = FitSums::default();
        for x in 10..60 {
            sums.add(x as f64, 5.0 + 2.0 * x as f64);
        }
        let line = sums.line().unwrap();
        assert!((line.intercept - 5.0).abs() < 1e-6);
        assert!((line.slope - 2.0).abs() < 1e-9);
        assert!((line.r2 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn line_is_undefined_without_spread_or_samples() {
        let mut sums = FitSums::default();
        assert!(sums.line().is_none());
        for _ in 0..10 {
            sums.add(7.0, 1.0);
        }
        assert!(sums.line().is_none());
    }

    #[test]
    fn merged_sums_fit_like_one_batch() {
        let (mut a, mut b, mut all) = (FitSums::default(), FitSums::default(), FitSums::default());
        for x in 0..20 {
            let y = 3.0 + 0.5 * x as f64;
            if x % 2 == 0 { a.add(x as f64, y) } else { b.add(x as f64, y) }
            all.add(x as f64, y);
        }
        a.merge(&b);
        assert!((a.line().unwrap().slope - all.line().unwrap().slope).abs() < 1e-12);
    }
}
