//! Hardware-level measurement for the profiler (compiled only with the
//! `profile` feature): what core and clock a step really got, and timing of
//! loop bodies too short for `Instant`.
//!
//! - **Core type.** On hybrid Intel CPUs, CPUID leaf 0x1A says whether the
//!   calling thread is on a performance or an efficiency core. A step that
//!   ran on an efficiency core is simply slower, which looks like a
//!   regression if you don't know.
//! - **Clock probe.** Windows (and most OSes) expose no cheap way to read the
//!   actual core frequency, so a short loop of dependent integer adds is
//!   timed instead. Such a chain runs at about one iteration per cycle, so
//!   `iterations / time` is the clock the thread is getting right now. The
//!   absolute value assumes exactly one iteration per cycle and may be off by
//!   a constant factor; trust relative changes (throttling, turbo).
//! - **`PhaseSampler`.** Times one particle in `SAMPLE_EVERY`, phase by
//!   phase, with fenced timestamp-counter reads (a few ns, against `Instant`'s
//!   tens of ns and 100 ns tick on Windows), and accumulates the results
//!   locally; see the `profiling` module docs, "Hardware telemetry and
//!   sampled fits".
//!
//! Fences serialise the pipeline, so a sampled particle runs slower than it
//! would inside the unsampled stream (the loop bodies overlap there). Sampled
//! times are good for *ratios* between phases and for how cost scales with
//! work, not for absolute totals.
//!
//! Everything degrades gracefully off x86_64: no core type, no clock probe,
//! and the sampler falls back to `Instant`.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use crate::profiling::{CoreType, FitSums, HwSample};

/// One particle in this many is timed by a `PhaseSampler`.
pub const SAMPLE_EVERY: usize = 32;

/// The clock probe runs on every this-many'th step.
const CLOCK_PROBE_EVERY: u32 = 8;

/// Iterations of the probe loop: about 10 µs at 3 GHz, i.e. roughly 0.2% of
/// a 5 ms step on the probed steps.
#[cfg(target_arch = "x86_64")]
const CLOCK_PROBE_ITERATIONS: u64 = 32_768;

// ---------------------------------------------------------------------
// Timestamp counter
// ---------------------------------------------------------------------

/// A fenced read of the timestamp counter: neither earlier nor later
/// instructions can move across it, so a read pair brackets the code
/// between them.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn now_ticks() -> u64 {
    use core::arch::x86_64::{_mm_lfence, _rdtsc};
    // SAFETY: `rdtsc` and `lfence` exist on every x86_64 CPU (`lfence` is
    // part of SSE2, the architecture baseline) and have no memory effects.
    unsafe {
        _mm_lfence();
        let t = _rdtsc();
        _mm_lfence();
        t
    }
}

/// Fallback: nanoseconds since the first call, so one tick is one ns.
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn now_ticks() -> u64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

/// How to turn ticks into time, measured once per process.
struct Calibration {
    ns_per_tick: f64,
    /// Ticks an empty bracketed region costs (two back-to-back reads);
    /// subtracted from every sampled interval.
    overhead_ticks: u64,
}

fn calibration() -> &'static Calibration {
    static CALIBRATION: OnceLock<Calibration> = OnceLock::new();
    CALIBRATION.get_or_init(|| {
        // Tick rate against the OS clock over a short spin.
        let start = Instant::now();
        let ticks_start = now_ticks();
        while start.elapsed() < Duration::from_millis(2) {
            std::hint::spin_loop();
        }
        let ticks_end = now_ticks();
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        let ns_per_tick = elapsed_ns / ticks_end.wrapping_sub(ticks_start).max(1) as f64;

        // Best of many empty brackets: the cost of the measurement itself.
        let mut overhead_ticks = u64::MAX;
        for _ in 0..512 {
            let a = now_ticks();
            let b = now_ticks();
            overhead_ticks = overhead_ticks.min(b.wrapping_sub(a));
        }

        Calibration {
            ns_per_tick,
            overhead_ticks,
        }
    })
}

// ---------------------------------------------------------------------
// Core type and clock probe
// ---------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn core_type() -> CoreType {
    use core::arch::x86_64::{__cpuid, __cpuid_count};

    // Hybrid CPUs set CPUID.(7,0):EDX[15]; only then is leaf 0x1A meaningful.
    static HYBRID: OnceLock<bool> = OnceLock::new();
    // SAFETY: `cpuid` exists on every x86_64 CPU.
    let hybrid = *HYBRID.get_or_init(|| unsafe {
        __cpuid(0).eax >= 0x1A && (__cpuid_count(7, 0).edx >> 15) & 1 == 1
    });
    if !hybrid {
        return CoreType::Unknown;
    }
    // SAFETY: as above; the leaf's existence was checked in `HYBRID`.
    let leaf = unsafe { __cpuid_count(0x1A, 0) };
    match leaf.eax >> 24 {
        0x40 => CoreType::Performance,
        0x20 => CoreType::Efficiency,
        _ => CoreType::Unknown,
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn core_type() -> CoreType {
    CoreType::Unknown
}

/// A chain of `iterations` dependent adds: one iteration per cycle on any
/// modern core, since each add needs the previous result.
#[cfg(target_arch = "x86_64")]
fn dependent_adds(iterations: u64) -> u64 {
    let mut x = 0u64;
    // SAFETY: register-only arithmetic; no memory, no stack, flags clobbered
    // (the default for `asm!`).
    unsafe {
        core::arch::asm!(
            "2:",
            "add {x}, 1",
            "dec {n}",
            "jnz 2b",
            x = inout(reg) x,
            n = inout(reg) iterations => _,
            options(nomem, nostack),
        );
    }
    x
}

/// Effective clock in MHz, from timing `dependent_adds`.
#[cfg(target_arch = "x86_64")]
fn probe_clock_mhz() -> Option<f32> {
    let cal = calibration();
    let start = now_ticks();
    let result = dependent_adds(CLOCK_PROBE_ITERATIONS);
    let end = now_ticks();
    std::hint::black_box(result);

    let ns = end
        .wrapping_sub(start)
        .saturating_sub(cal.overhead_ticks) as f64
        * cal.ns_per_tick;
    // iterations per ns is GHz.
    (ns > 0.0).then(|| (CLOCK_PROBE_ITERATIONS as f64 / ns * 1000.0) as f32)
}

#[cfg(not(target_arch = "x86_64"))]
fn probe_clock_mhz() -> Option<f32> {
    None
}

/// Per-step hardware telemetry: created at the start of a step, `finish`ed
/// at its end.
pub struct StepProbe {
    core_at_start: CoreType,
}

impl StepProbe {
    pub fn start() -> Self {
        // Calibrates on first use; see `StepGuard::new`.
        let _ = calibration();
        Self {
            core_at_start: core_type(),
        }
    }

    pub fn finish(&self) -> HwSample {
        static STEPS: AtomicU32 = AtomicU32::new(0);

        let core_at_end = core_type();
        let core_type = if core_at_end == self.core_at_start {
            core_at_end
        } else {
            CoreType::Mixed
        };
        let nth = STEPS.fetch_add(1, Ordering::Relaxed);
        HwSample {
            core_type,
            clock_mhz: (nth % CLOCK_PROBE_EVERY == 0)
                .then(probe_clock_mhz)
                .flatten(),
        }
    }
}

// ---------------------------------------------------------------------
// Sampled per-particle timing
// ---------------------------------------------------------------------

/// Whether the current particle is being timed, and when its last phase
/// ended. Create with `PhaseSampler::start`.
#[derive(Clone, Copy)]
pub struct Tick {
    last: u64,
    on: bool,
}

impl Tick {
    /// True for the particles that are timed (one in `SAMPLE_EVERY`); lets
    /// the caller do extra measurement work on exactly those particles.
    #[inline(always)]
    pub fn is_sampled(&self) -> bool {
        self.on
    }
}

/// Times the `N` consecutive phases of a per-particle loop body on one
/// particle in `SAMPLE_EVERY`, accumulating `(work, ns)` sums per phase.
///
/// ```ignore
/// let mut sampler = PhaseSampler::new(["pass::setup", "pass::math"]);
/// for (i, particle) in particles.iter().enumerate() {
///     let mut tick = sampler.start(i);
///     /* setup */
///     sampler.lap(&mut tick, 0, work_items as f64);
///     /* math */
///     sampler.lap(&mut tick, 1, work_items as f64);
/// }
/// sampler.finish(); // one `Sink::fit` per phase
/// ```
///
/// Unsampled particles cost one predictable branch per call.
pub struct PhaseSampler<const N: usize> {
    labels: [&'static str; N],
    fits: [FitSums; N],
    cal: &'static Calibration,
}

impl<const N: usize> PhaseSampler<N> {
    pub fn new(labels: [&'static str; N]) -> Self {
        Self {
            labels,
            fits: [FitSums::default(); N],
            cal: calibration(),
        }
    }

    /// Starts the clock if particle `index` is one of the sampled ones.
    #[inline(always)]
    pub fn start(&self, index: usize) -> Tick {
        if index % SAMPLE_EVERY == 0 {
            Tick {
                last: now_ticks(),
                on: true,
            }
        } else {
            Tick { last: 0, on: false }
        }
    }

    /// Ends phase `phase` (time since `start` or the previous `lap`) and
    /// attributes it to `work` units of work (e.g. candidates tested).
    #[inline(always)]
    pub fn lap(&mut self, tick: &mut Tick, phase: usize, work: f64) {
        if !tick.on {
            return;
        }
        let now = now_ticks();
        let ticks = now
            .wrapping_sub(tick.last)
            .saturating_sub(self.cal.overhead_ticks);
        self.fits[phase].add(work, ticks as f64 * self.cal.ns_per_tick);
        // Restart after the bookkeeping above so it isn't billed to the
        // next phase.
        tick.last = now_ticks();
    }

    /// Sends the accumulated sums, one `Sink::fit` per phase that has data.
    pub fn finish(self) {
        for (label, sums) in self.labels.iter().zip(self.fits.iter()) {
            if sums.n > 0 {
                crate::profiling::fit(*label, *sums);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_is_sane() {
        let cal = calibration();
        assert!(cal.ns_per_tick > 0.01 && cal.ns_per_tick < 100.0);
    }

    #[test]
    fn only_every_nth_particle_is_sampled() {
        let sampler = PhaseSampler::new(["a"]);
        assert!(sampler.start(0).is_sampled());
        assert!(!sampler.start(1).is_sampled());
        assert!(sampler.start(SAMPLE_EVERY).is_sampled());
    }

    #[test]
    fn a_sampled_lap_is_recorded() {
        let mut sampler = PhaseSampler::new(["a"]);
        let mut tick = sampler.start(0);
        sampler.lap(&mut tick, 0, 3.0);
        assert_eq!(sampler.fits[0].n, 1);
        assert_eq!(sampler.fits[0].sx, 3.0);
    }
}
