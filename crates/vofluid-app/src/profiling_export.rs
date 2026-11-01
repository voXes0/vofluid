//! JSON export of a profiling session, meant to be handed to a human or an
//! AI for bottleneck analysis.
//!
//! Turns the sink's `ExportSnapshot` into a self-describing report: per-epoch
//! statistics (so configurations are never averaged together), derived
//! metrics that remove the arithmetic a reader would otherwise have to guess
//! (`per_step_ms`, `ns_per_unit`, ...), and the per-step flight recorder.
//! Storage lives in `profiling_sink`; this module only reshapes and writes.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::profiling_sink::{
    EpochSnapshot, ExportSnapshot, FitSnapshot, RecorderSnapshot, StatSnapshot,
};

/// Bump when a field is removed, renamed, or changes meaning. Version 1 was
/// the pre-epoch format (lifetime stats per label, per-record `recent_ms`);
/// version 2 added epochs and the flight recorder; version 3 added hardware
/// telemetry, sampled fits and the `_sampled` counters.
const SCHEMA_VERSION: u32 = 3;

/// Embedded in the file so a reader who has never seen this tool can
/// interpret it. Keep it in step with the structs below.
const EXPORT_NOTES: &str = "Schema 3. `epochs` lists configuration epochs: a new one starts whenever \
SimConfig changes or 'New epoch' is pressed (at most 16 are kept, shortest finished ones dropped \
first). Everything inside an epoch was measured under that epoch's `config`, so epochs can be compared \
with each other but must never be averaged together. `step` is the wall time of one whole \
Simulation::step and is the root: there is no root span label. Each entry of `spans` is a timed code \
section: `calls` is how often it ran, `calls_per_step` its cadence (3 = once per substep), and \
`per_step_ms` its total time per simulation step. Compare labels by `per_step_ms` (or total_ms), never \
by mean_call_ms, which is per call and so not comparable across different cadences. Spans nest by \
'::' prefix (a::b::c is inside a::b when that label exists) and otherwise sit directly under `step`; \
a span's share of the step is per_step_ms / step.mean_ms, and nested spans must not be summed. \
`counters` count work done. A counter 'owner::unit' applies to the span 'owner' and every span nested \
under it, and is emitted once per call of 'owner'; a span's `per_unit` then gives nanoseconds per unit \
of work (e.g. per neighbour), which separates slower code from a heavier workload. \
`ns_per_particle_per_call` divides a span's mean call time by the epoch's particle_count. `recorder` \
is a flight recorder of the last `len` steps, oldest first, across epochs (`epoch_ids` says which \
epoch each step belongs to): `step_ms` is the whole step, `span_ms[label]` the summed time of that span \
within the step, `counters[label]` the summed count. min/max are single-call extremes and include \
startup outliers; use the recorder to look at spikes. Spans averaging under about 2 microseconds are \
at the timer's resolution (around 100 ns on Windows) and their decimals are noise. \
`hardware` counts the epoch's steps per core type (hybrid Intel CPUs only: a step on an efficiency \
core is much slower, which looks like a regression; `mixed` means the thread moved between core types \
mid-step) and summarises the effective-clock probe (a short dependent-add loop timed every few steps; \
the absolute MHz assumes one iteration per cycle, so trust relative changes). `recorder.core_type` \
(0 unknown, 1 performance, 2 efficiency, 3 mixed) and `recorder.clock_mhz` (0 = not probed on that \
step) give the same per step, so a slow step can be checked against the machine state. `fits` come \
from timing one particle in 32 with fenced timestamp-counter reads: per sampled phase, \
`ns = intercept + slope * work` is fitted over the samples (`work` is e.g. candidates tested), with \
standard errors and r2. Sampling serialises the pipeline, so sampled ns exceed the same code's cost \
inside the unsampled stream: use fits for ratios between phases and for how cost scales with work, \
not as absolute totals; the intercept extrapolates to zero work, so read its standard error. Counters \
whose unit ends in `_sampled` are estimates from the sampled particles only and get no per_unit. Not \
recorded: render/GPU time, wall-clock duration, release vs debug optimisation level.";

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    /// Unix timestamp (seconds), so exports from one session can be ordered.
    generated_at_unix: u64,
    notes: &'static str,
    environment: Environment,
    epochs: Vec<ExportedEpoch>,
    recorder: ExportedRecorder,
}

#[derive(Serialize)]
struct Environment {
    app_version: &'static str,
    os: &'static str,
    arch: &'static str,
    debug_assertions: bool,
    logical_cpus: Option<usize>,
}

#[derive(Serialize)]
struct ExportedEpoch {
    id: u32,
    /// Global index of the first step of this epoch (step 0 = first step of
    /// the session); lines up with `recorder.first_step`.
    first_step: u64,
    steps: u64,
    particle_count: usize,
    substeps: u32,
    dt: f32,
    /// The full `SimConfig` this epoch ran under.
    config: serde_json::Value,
    step: ExportedStep,
    hardware: ExportedHardware,
    spans: Vec<ExportedSpan>,
    counters: Vec<ExportedCounter>,
    fits: Vec<ExportedFit>,
}

#[derive(Serialize)]
struct ExportedHardware {
    steps_performance: u64,
    steps_efficiency: u64,
    steps_mixed: u64,
    /// Steps on CPUs (or builds) that don't report a core type.
    steps_unknown: u64,
    clock_probes: u64,
    clock_mhz_mean: Option<f64>,
    clock_mhz_min: Option<f32>,
    clock_mhz_max: Option<f32>,
}

/// `ns = intercept + slope * work` over the sampled particles of one phase.
#[derive(Serialize)]
struct ExportedFit {
    label: &'static str,
    samples: u64,
    mean_work: f64,
    mean_ns: f64,
    /// `None` when the fit is undefined (too few samples, or constant work).
    intercept_ns: Option<f64>,
    intercept_se_ns: Option<f64>,
    slope_ns_per_work: Option<f64>,
    slope_se_ns_per_work: Option<f64>,
    r2: Option<f64>,
}

#[derive(Serialize)]
struct ExportedStep {
    total_ms: f64,
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

#[derive(Serialize)]
struct ExportedSpan {
    label: &'static str,
    calls: u64,
    calls_per_step: f64,
    total_ms: f64,
    per_step_ms: f64,
    mean_call_ms: f64,
    min_call_ms: f64,
    max_call_ms: f64,
    ns_per_particle_per_call: Option<f64>,
    per_unit: Vec<PerUnit>,
}

#[derive(Serialize)]
struct PerUnit {
    counter: &'static str,
    unit: &'static str,
    ns_per_unit: f64,
}

#[derive(Serialize)]
struct ExportedCounter {
    label: &'static str,
    calls: u64,
    total: u64,
    per_step: f64,
    /// Total divided by (calls x particle_count): e.g. neighbours per particle.
    per_particle_per_call: Option<f64>,
}

#[derive(Serialize)]
struct ExportedRecorder {
    len: usize,
    first_step: u64,
    epoch_ids: Vec<u32>,
    step_ms: Vec<f64>,
    /// 0 unknown, 1 performance, 2 efficiency, 3 mixed.
    core_type: Vec<u8>,
    /// Effective clock probe in MHz; 0 on steps that weren't probed.
    clock_mhz: Vec<f32>,
    span_ms: BTreeMap<&'static str, Vec<f64>>,
    counters: BTreeMap<&'static str, Vec<u64>>,
}

fn round(x: f64, places: i32) -> f64 {
    let factor = 10f64.powi(places);
    (x * factor).round() / factor
}

fn ns_to_ms(ns: u64) -> f64 {
    round(ns as f64 / 1e6, 6)
}

/// Counters that belong to `span`: those named `<owner>::<unit>` where
/// `owner` is the span itself or a label it is nested under.
fn per_unit(span: &StatSnapshot, counters: &[StatSnapshot]) -> Vec<PerUnit> {
    counters
        .iter()
        .filter_map(|c| {
            let (owner, unit) = c.label.rsplit_once("::")?;
            // Estimates from a subsample: dividing the whole span's time by
            // them would be meaningless.
            if unit.ends_with("_sampled") {
                return None;
            }
            let belongs = span.label == owner
                || span
                    .label
                    .strip_prefix(owner)
                    .is_some_and(|rest| rest.starts_with("::"));
            (belongs && c.sum > 0).then(|| PerUnit {
                counter: c.label,
                unit,
                ns_per_unit: round(span.sum as f64 / c.sum as f64, 3),
            })
        })
        .collect()
}

fn export_span(s: &StatSnapshot, epoch: &EpochSnapshot) -> ExportedSpan {
    let steps = epoch.steps.max(1) as f64;
    let calls = s.calls.max(1) as f64;
    let total_ns = s.sum as f64;
    ExportedSpan {
        label: s.label,
        calls: s.calls,
        calls_per_step: round(s.calls as f64 / steps, 3),
        total_ms: ns_to_ms(s.sum),
        per_step_ms: round(total_ns / 1e6 / steps, 6),
        mean_call_ms: round(total_ns / 1e6 / calls, 6),
        min_call_ms: ns_to_ms(s.min),
        max_call_ms: ns_to_ms(s.max),
        ns_per_particle_per_call: (epoch.particle_count > 0)
            .then(|| round(total_ns / calls / epoch.particle_count as f64, 3)),
        per_unit: per_unit(s, &epoch.counters),
    }
}

fn export_counter(c: &StatSnapshot, epoch: &EpochSnapshot) -> ExportedCounter {
    let steps = epoch.steps.max(1) as f64;
    ExportedCounter {
        label: c.label,
        calls: c.calls,
        total: c.sum,
        per_step: round(c.sum as f64 / steps, 3),
        per_particle_per_call: (epoch.particle_count > 0 && c.calls > 0)
            .then(|| round(c.sum as f64 / (c.calls as f64 * epoch.particle_count as f64), 3)),
    }
}

fn export_fit(f: &FitSnapshot) -> ExportedFit {
    let line = f.sums.line();
    ExportedFit {
        label: f.label,
        samples: f.sums.n,
        mean_work: round(f.sums.mean_x(), 3),
        mean_ns: round(f.sums.mean_y(), 3),
        intercept_ns: line.map(|l| round(l.intercept, 3)),
        intercept_se_ns: line.map(|l| round(l.intercept_se, 3)),
        slope_ns_per_work: line.map(|l| round(l.slope, 4)),
        slope_se_ns_per_work: line.map(|l| round(l.slope_se, 4)),
        r2: line.map(|l| round(l.r2, 4)),
    }
}

fn export_epoch(e: &EpochSnapshot) -> ExportedEpoch {
    let steps = e.steps.max(1) as f64;
    ExportedEpoch {
        id: e.id,
        first_step: e.first_step,
        steps: e.steps,
        particle_count: e.particle_count,
        substeps: e.substeps,
        dt: e.dt,
        config: e.config.clone(),
        step: ExportedStep {
            total_ms: ns_to_ms(e.step.sum),
            mean_ms: round(e.step.sum as f64 / 1e6 / steps, 6),
            min_ms: ns_to_ms(e.step.min),
            max_ms: ns_to_ms(e.step.max),
        },
        hardware: ExportedHardware {
            steps_performance: e.hardware.steps_performance,
            steps_efficiency: e.hardware.steps_efficiency,
            steps_mixed: e.hardware.steps_mixed,
            steps_unknown: e.hardware.steps_unknown,
            clock_probes: e.hardware.clock_probes,
            clock_mhz_mean: e.hardware.clock_mhz_mean.map(|m| round(m, 1)),
            clock_mhz_min: e.hardware.clock_mhz_min,
            clock_mhz_max: e.hardware.clock_mhz_max,
        },
        spans: e.spans.iter().map(|s| export_span(s, e)).collect(),
        counters: e.counters.iter().map(|c| export_counter(c, e)).collect(),
        fits: e.fits.iter().map(export_fit).collect(),
    }
}

fn export_recorder(r: &RecorderSnapshot) -> ExportedRecorder {
    ExportedRecorder {
        len: r.step_ms.len(),
        first_step: r.first_step,
        epoch_ids: r.epoch_ids.clone(),
        step_ms: r.step_ms.iter().map(|v| round(*v, 4)).collect(),
        core_type: r.core_type.clone(),
        clock_mhz: r.clock_mhz.iter().map(|m| m.round()).collect(),
        span_ms: r
            .spans
            .iter()
            .map(|(label, column)| {
                (
                    *label,
                    column.iter().map(|v| round(*v, 4)).collect::<Vec<f64>>(),
                )
            })
            .collect(),
        counters: r
            .counters
            .iter()
            .map(|(label, column)| (*label, column.iter().map(|v| *v as u64).collect::<Vec<u64>>()))
            .collect(),
    }
}

fn build_report(snapshot: &ExportSnapshot, generated_at_unix: u64) -> Report {
    Report {
        schema_version: SCHEMA_VERSION,
        generated_at_unix,
        notes: EXPORT_NOTES,
        environment: Environment {
            app_version: env!("CARGO_PKG_VERSION"),
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            debug_assertions: cfg!(debug_assertions),
            logical_cpus: std::thread::available_parallelism().ok().map(|n| n.get()),
        },
        // An epoch with no finished step has no statistics worth exporting.
        epochs: snapshot
            .epochs
            .iter()
            .filter(|e| e.steps > 0)
            .map(export_epoch)
            .collect(),
        recorder: export_recorder(&snapshot.recorder),
    }
}

/// Serializes the session to a JSON file in the current directory (named
/// `vofluid-profile-<unix seconds>.json`) and returns its path. A few tens
/// of KB per epoch plus the bounded recorder (`HISTORY_LEN` steps x labels).
pub fn export_report(snapshot: &ExportSnapshot) -> anyhow::Result<std::path::PathBuf> {
    let generated_at_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let json = serde_json::to_string_pretty(&build_report(snapshot, generated_at_unix))?;

    let dir = std::env::current_dir()?;
    let path = dir.join(format!("vofluid-profile-{generated_at_unix}.json"));
    std::fs::write(&path, json)?;

    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stat(label: &'static str, sum: u64) -> StatSnapshot {
        StatSnapshot {
            label,
            calls: 1,
            sum,
            min: sum,
            max: sum,
        }
    }

    #[test]
    fn counter_applies_to_its_owner_and_nested_spans_only() {
        let counters = [
            stat("a::b::neighbours", 100),
            stat("a::b::c::candidates", 50),
        ];

        let owner = per_unit(&stat("a::b", 1000), &counters);
        assert_eq!(owner.len(), 1);
        assert_eq!(owner[0].unit, "neighbours");
        assert_eq!(owner[0].ns_per_unit, 10.0);

        // Nested span gets both counters (its own and its ancestor's).
        assert_eq!(per_unit(&stat("a::b::c", 1000), &counters).len(), 2);

        // A sibling whose name merely shares a prefix string gets neither.
        assert!(per_unit(&stat("a::bc", 1000), &counters).is_empty());
    }

    #[test]
    fn sampled_counters_get_no_per_unit() {
        let counters = [stat("a::b::survivors_sampled", 10)];
        assert!(per_unit(&stat("a::b", 1000), &counters).is_empty());
    }
}
