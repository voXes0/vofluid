//! App-side storage for `vofluid_core::profiling`: an in-memory `Sink` that
//! keeps everything the profiler panel (`profiling_gui`) shows and the JSON
//! export (`profiling_export`) writes.
//!
//! `vofluid-core` only *measures*; this module decides how measurements are
//! kept. Three structures, all bounded so a long session can't grow them:
//!
//! - **Epochs.** Statistics are kept per configuration epoch: a new epoch
//!   starts whenever `SimConfig` changes (the core calls `begin_epoch`) or
//!   the panel's "New epoch" button is pressed. Within an epoch every number
//!   was measured under one config, so nothing blends, say, 2000 and 4500
//!   particles into a single average. At most `MAX_EPOCHS` are kept; the
//!   shortest finished ones are dropped first.
//! - **Flight recorder.** The last `HISTORY_LEN` steps, one row per step
//!   across all epochs, with each label's summed time (or count) for that
//!   step. Because every column is per *step*, labels with different
//!   cadences (once per step vs once per substep) line up row for row.
//! - **Counters.** Work counts (`prof_count!`) are stored like spans: epoch
//!   totals plus a per-step column.
//!
//! - **Hardware telemetry and sampled fits.** Per step, which core type ran
//!   it and (every few steps) the effective clock, in the recorder and as
//!   epoch totals; per epoch, the `(work, ns)` sums of the sampled phases.
//!
//! The whole sink sits behind one `Mutex`; the simulation is single-threaded
//! today, so it is uncontended. A parallel backend would want thread-local
//! buffers flushed at `step_end` instead.

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use vofluid_core::config::SimConfig;
use vofluid_core::profiling::{CoreType, FitSums, HwSample, Sample, Sink};

/// Rows kept by the flight recorder: the last 600 steps (12 s of simulated
/// time at the default dt of 0.02 s), whatever mix of cadences the labels have.
pub const HISTORY_LEN: usize = 600;

/// Most epochs kept at once.
const MAX_EPOCHS: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Span,
    Counter,
}

/// Aggregate of one label within one epoch. `sum` is nanoseconds for a span
/// and raw units for a counter; for a counter, `min`/`max` are over single
/// `count` calls.
#[derive(Clone, Copy)]
struct SlotStats {
    calls: u64,
    sum: u64,
    min: u64,
    max: u64,
}

impl SlotStats {
    const EMPTY: Self = Self {
        calls: 0,
        sum: 0,
        min: u64::MAX,
        max: 0,
    };

    fn add(&mut self, calls: u64, sum: u64, min: u64, max: u64) {
        self.calls += calls;
        self.sum += sum;
        self.min = self.min.min(min);
        self.max = self.max.max(max);
    }
}

/// One label (span or counter), interned the first time it's seen.
struct Slot {
    label: &'static str,
    kind: Kind,
    /// Flight-recorder column: one entry per recorder row, in milliseconds
    /// for a span or raw units for a counter, summed over the step.
    history: VecDeque<f64>,
    /// Accumulated so far in the step in progress (ns, or units).
    pending: u64,
}

struct Row {
    epoch: u32,
    total_ms: f64,
    /// `core_code` of the core type the step ran on.
    core: u8,
    /// Effective clock from the probe after the step; 0 when not probed.
    clock_mhz: f32,
}

/// Code for a core type in the recorder and the export: 0 unknown,
/// 1 performance, 2 efficiency, 3 mixed (moved between types mid-step).
fn core_code(core: CoreType) -> u8 {
    match core {
        CoreType::Unknown => 0,
        CoreType::Performance => 1,
        CoreType::Efficiency => 2,
        CoreType::Mixed => 3,
    }
}

/// Hardware telemetry summed over one epoch's steps.
#[derive(Clone, Copy)]
struct HwStats {
    /// Steps per `core_code`.
    steps_by_core: [u64; 4],
    probes: u64,
    clock_sum: f64,
    clock_min: f32,
    clock_max: f32,
}

impl HwStats {
    const EMPTY: Self = Self {
        steps_by_core: [0; 4],
        probes: 0,
        clock_sum: 0.0,
        clock_min: f32::MAX,
        clock_max: 0.0,
    };

    fn add(&mut self, hw: &HwSample) {
        self.steps_by_core[core_code(hw.core_type) as usize] += 1;
        if let Some(mhz) = hw.clock_mhz {
            self.probes += 1;
            self.clock_sum += mhz as f64;
            self.clock_min = self.clock_min.min(mhz);
            self.clock_max = self.clock_max.max(mhz);
        }
    }
}

struct Epoch {
    id: u32,
    /// Global index of the first step run in this epoch.
    first_step: u64,
    steps: u64,
    config: serde_json::Value,
    particle_count: usize,
    substeps: u32,
    dt: f32,
    /// Wall time of whole steps (the root; there is no root span label).
    step: SlotStats,
    /// Indexed by slot id; `calls == 0` means "not seen in this epoch".
    stats: Vec<SlotStats>,
    hw: HwStats,
    /// Sampled `(work, ns)` sums per label (`PhaseSampler`).
    fits: HashMap<&'static str, FitSums>,
}

impl Epoch {
    fn new(id: u32, first_step: u64, config: &SimConfig) -> Self {
        Self {
            id,
            first_step,
            steps: 0,
            // `SimConfig` is `Serialize` because `vofluid-app` always builds
            // `vofluid-core` with `features = ["all"]`.
            config: serde_json::to_value(config).unwrap_or(serde_json::Value::Null),
            particle_count: config.particle_count,
            substeps: config.substeps,
            dt: config.dt,
            step: SlotStats::EMPTY,
            stats: Vec::new(),
            hw: HwStats::EMPTY,
            fits: HashMap::new(),
        }
    }

    /// An epoch with no known config: records that arrive before the core
    /// ever called `begin_epoch` (e.g. a backend driven directly).
    fn unknown(id: u32, first_step: u64) -> Self {
        Self {
            id,
            first_step,
            steps: 0,
            config: serde_json::Value::Null,
            particle_count: 0,
            substeps: 0,
            dt: 0.0,
            step: SlotStats::EMPTY,
            stats: Vec::new(),
            hw: HwStats::EMPTY,
            fits: HashMap::new(),
        }
    }

    /// A fresh epoch with the same config, for "discard the warm-up". `id`
    /// and `first_step` are placeholders; the caller overwrites them.
    fn fresh_copy(&self) -> Self {
        Self {
            id: 0,
            first_step: 0,
            steps: 0,
            config: self.config.clone(),
            particle_count: self.particle_count,
            substeps: self.substeps,
            dt: self.dt,
            step: SlotStats::EMPTY,
            stats: Vec::new(),
            hw: HwStats::EMPTY,
            fits: HashMap::new(),
        }
    }

    fn snapshot(&self, slots: &[Slot]) -> EpochSnapshot {
        let mut spans = Vec::new();
        let mut counters = Vec::new();
        for (id, slot) in slots.iter().enumerate() {
            let Some(s) = self.stats.get(id).filter(|s| s.calls > 0) else {
                continue;
            };
            let snap = StatSnapshot::new(slot.label, s);
            match slot.kind {
                Kind::Span => spans.push(snap),
                Kind::Counter => counters.push(snap),
            }
        }
        spans.sort_by(|a, b| b.sum.cmp(&a.sum));
        counters.sort_by(|a, b| a.label.cmp(b.label));
        let mut fits: Vec<FitSnapshot> = self
            .fits
            .iter()
            .map(|(label, sums)| FitSnapshot {
                label: *label,
                sums: *sums,
            })
            .collect();
        fits.sort_by(|a, b| a.label.cmp(b.label));
        let h = &self.hw;
        let probed = h.probes > 0;
        EpochSnapshot {
            id: self.id,
            first_step: self.first_step,
            steps: self.steps,
            config: self.config.clone(),
            particle_count: self.particle_count,
            substeps: self.substeps,
            dt: self.dt,
            step: StatSnapshot::new("step", &self.step),
            spans,
            counters,
            hardware: HardwareSnapshot {
                steps_unknown: h.steps_by_core[0],
                steps_performance: h.steps_by_core[1],
                steps_efficiency: h.steps_by_core[2],
                steps_mixed: h.steps_by_core[3],
                clock_probes: h.probes,
                clock_mhz_mean: probed.then(|| h.clock_sum / h.probes as f64),
                clock_mhz_min: probed.then_some(h.clock_min),
                clock_mhz_max: probed.then_some(h.clock_max),
            },
            fits,
        }
    }
}

struct State {
    ids: HashMap<&'static str, usize>,
    slots: Vec<Slot>,
    /// Oldest first; the last one is the current epoch.
    epochs: VecDeque<Epoch>,
    /// Flight-recorder rows, oldest first; every slot's `history` has the
    /// same length.
    rows: VecDeque<Row>,
    next_epoch_id: u32,
    /// Global index of the next step to finish.
    next_step: u64,
    in_step: bool,
    /// Telemetry reported for the step in progress, attached at `end_step`.
    pending_hw: Option<HwSample>,
}

impl State {
    fn new() -> Self {
        Self {
            ids: HashMap::new(),
            slots: Vec::new(),
            epochs: VecDeque::new(),
            rows: VecDeque::with_capacity(HISTORY_LEN),
            next_epoch_id: 0,
            next_step: 0,
            in_step: false,
            pending_hw: None,
        }
    }

    /// Starts a new epoch built by `make(id, first_step)`. An epoch that
    /// never ran a step is replaced rather than kept, so several config
    /// changes between two steps leave one epoch, not several empty ones.
    fn start_epoch(&mut self, make: impl FnOnce(u32, u64) -> Epoch) {
        if self.epochs.back().is_some_and(|e| e.steps == 0) {
            self.epochs.pop_back();
        }
        if self.epochs.len() >= MAX_EPOCHS {
            // Drop the least informative finished epoch (never the current
            // one, which is the last).
            let shortest = (0..self.epochs.len() - 1).min_by_key(|&i| self.epochs[i].steps);
            if let Some(i) = shortest {
                self.epochs.remove(i);
            }
        }
        let epoch = make(self.next_epoch_id, self.next_step);
        self.next_epoch_id += 1;
        self.epochs.push_back(epoch);
    }

    fn ensure_epoch(&mut self) {
        if self.epochs.is_empty() {
            self.start_epoch(Epoch::unknown);
        }
    }

    fn id_of(&mut self, label: &'static str, kind: Kind) -> usize {
        if let Some(&id) = self.ids.get(label) {
            return id;
        }
        let id = self.slots.len();
        self.ids.insert(label, id);
        self.slots.push(Slot {
            label,
            kind,
            // Backfill so every column keeps one entry per recorder row.
            history: VecDeque::from(vec![0.0; self.rows.len()]),
            pending: 0,
        });
        id
    }

    fn add(&mut self, label: &'static str, kind: Kind, calls: u64, sum: u64, min: u64, max: u64) {
        let id = self.id_of(label, kind);
        self.ensure_epoch();
        if let Some(epoch) = self.epochs.back_mut() {
            if epoch.stats.len() <= id {
                epoch.stats.resize(id + 1, SlotStats::EMPTY);
            }
            epoch.stats[id].add(calls, sum, min, max);
        }
        if self.in_step {
            self.slots[id].pending += sum;
        }
    }

    fn end_step(&mut self, total: Duration) {
        self.in_step = false;
        self.ensure_epoch();
        let total_ns = total.as_nanos() as u64;
        let hw = self.pending_hw.take();
        let mut epoch_id = 0;
        if let Some(epoch) = self.epochs.back_mut() {
            epoch.steps += 1;
            epoch.step.add(1, total_ns, total_ns, total_ns);
            if let Some(hw) = &hw {
                epoch.hw.add(hw);
            }
            epoch_id = epoch.id;
        }
        if self.rows.len() == HISTORY_LEN {
            self.rows.pop_front();
            for slot in &mut self.slots {
                slot.history.pop_front();
            }
        }
        self.rows.push_back(Row {
            epoch: epoch_id,
            total_ms: total_ns as f64 / 1e6,
            core: hw.map_or(0, |h| core_code(h.core_type)),
            clock_mhz: hw.and_then(|h| h.clock_mhz).unwrap_or(0.0),
        });
        for slot in &mut self.slots {
            let value = match slot.kind {
                Kind::Span => slot.pending as f64 / 1e6,
                Kind::Counter => slot.pending as f64,
            };
            slot.history.push_back(value);
            slot.pending = 0;
        }
        self.next_step += 1;
    }
}

/// `vofluid_core::profiling::Sink` keeping bounded per-epoch statistics and
/// a per-step flight recorder in memory.
pub struct InMemorySink {
    state: Mutex<State>,
}

impl InMemorySink {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State::new()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap()
    }

    /// Starts a fresh epoch with the current config (statistics restart from
    /// zero; the flight recorder is untouched). Used to discard a warm-up
    /// period, e.g. while freshly added particles are still settling.
    pub fn start_new_epoch(&self) {
        let mut state = self.lock();
        // An epoch that ran no step is already fresh.
        let template = match state.epochs.back() {
            Some(current) if current.steps > 0 => current.fresh_copy(),
            _ => return,
        };
        state.start_epoch(move |id, first_step| Epoch {
            id,
            first_step,
            ..template
        });
    }

    /// Cheap view for the live panel: the current epoch only, with the
    /// recorder's history as `f32` sparklines. `None` until a step finished.
    pub fn live_view(&self) -> Option<LiveView> {
        let state = self.lock();
        let epoch = state.epochs.back()?;
        if epoch.steps == 0 {
            return None;
        }
        let steps = epoch.steps as f64;
        let ms = |ns: u64| ns as f64 / 1e6;

        let step = LiveRow {
            label: "step",
            calls: epoch.step.calls,
            per_step_ms: (ms(epoch.step.sum) / steps) as f32,
            mean_call_ms: (ms(epoch.step.sum) / epoch.step.calls as f64) as f32,
            max_call_ms: ms(epoch.step.max) as f32,
            history: state.rows.iter().map(|r| r.total_ms as f32).collect(),
        };

        let mut spans: Vec<(u64, LiveRow)> = Vec::new();
        let mut counters = Vec::new();
        for (id, slot) in state.slots.iter().enumerate() {
            let Some(s) = epoch.stats.get(id).filter(|s| s.calls > 0) else {
                continue;
            };
            match slot.kind {
                Kind::Span => spans.push((
                    s.sum,
                    LiveRow {
                        label: slot.label,
                        calls: s.calls,
                        per_step_ms: (ms(s.sum) / steps) as f32,
                        mean_call_ms: (ms(s.sum) / s.calls as f64) as f32,
                        max_call_ms: ms(s.max) as f32,
                        history: slot.history.iter().map(|v| *v as f32).collect(),
                    },
                )),
                Kind::Counter => counters.push(LiveCounter {
                    label: slot.label,
                    per_step: s.sum as f64 / steps,
                }),
            }
        }
        spans.sort_by(|a, b| b.0.cmp(&a.0));
        counters.sort_by(|a, b| a.label.cmp(b.label));

        let hw = &epoch.hw;
        let hw_steps = hw.steps_by_core.iter().sum::<u64>().max(1) as f32;
        let percent = |code: usize| hw.steps_by_core[code] as f32 / hw_steps * 100.0;
        let hardware = LiveHardware {
            performance_pct: percent(1),
            efficiency_pct: percent(2),
            mixed_pct: percent(3),
            last_clock_mhz: state
                .rows
                .iter()
                .rev()
                .map(|r| r.clock_mhz)
                .find(|&mhz| mhz > 0.0),
            mean_clock_mhz: (hw.probes > 0).then(|| (hw.clock_sum / hw.probes as f64) as f32),
        };
        let mut fits: Vec<LiveFit> = epoch
            .fits
            .iter()
            .filter_map(|(label, sums)| {
                sums.line().map(|line| LiveFit {
                    label: *label,
                    samples: sums.n,
                    intercept_ns: line.intercept,
                    slope_ns: line.slope,
                })
            })
            .collect();
        fits.sort_by(|a, b| a.label.cmp(b.label));

        Some(LiveView {
            epoch_id: epoch.id,
            epoch_count: state.epochs.len(),
            particle_count: epoch.particle_count,
            substeps: epoch.substeps,
            dt: epoch.dt,
            steps: epoch.steps,
            step,
            spans: spans.into_iter().map(|(_, row)| row).collect(),
            counters,
            hardware,
            fits,
        })
    }

    /// Everything, for the JSON export. Clones all epochs and the whole
    /// recorder, so call it on demand (a button press), not every frame.
    pub fn export_snapshot(&self) -> ExportSnapshot {
        let state = self.lock();
        let mut spans = Vec::new();
        let mut counters = Vec::new();
        for slot in &state.slots {
            let column: Vec<f64> = slot.history.iter().copied().collect();
            match slot.kind {
                Kind::Span => spans.push((slot.label, column)),
                Kind::Counter => counters.push((slot.label, column)),
            }
        }
        ExportSnapshot {
            epochs: state
                .epochs
                .iter()
                .map(|e| e.snapshot(&state.slots))
                .collect(),
            recorder: RecorderSnapshot {
                first_step: state.next_step - state.rows.len() as u64,
                epoch_ids: state.rows.iter().map(|r| r.epoch).collect(),
                step_ms: state.rows.iter().map(|r| r.total_ms).collect(),
                core_type: state.rows.iter().map(|r| r.core).collect(),
                clock_mhz: state.rows.iter().map(|r| r.clock_mhz).collect(),
                spans,
                counters,
            },
        }
    }
}

impl Default for InMemorySink {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for InMemorySink {
    fn record(&self, sample: Sample) {
        self.lock().add(
            sample.label,
            Kind::Span,
            sample.count as u64,
            sample.total.as_nanos() as u64,
            sample.min.as_nanos() as u64,
            sample.max.as_nanos() as u64,
        );
    }

    fn count(&self, label: &'static str, value: u64) {
        self.lock().add(label, Kind::Counter, 1, value, value, value);
    }

    fn begin_epoch(&self, config: &SimConfig) {
        self.lock()
            .start_epoch(|id, first_step| Epoch::new(id, first_step, config));
    }

    fn step_begin(&self) {
        self.lock().in_step = true;
    }

    fn step_end(&self, total: Duration) {
        self.lock().end_step(total);
    }

    fn hardware(&self, sample: HwSample) {
        self.lock().pending_hw = Some(sample);
    }

    fn fit(&self, label: &'static str, sums: FitSums) {
        let mut state = self.lock();
        state.ensure_epoch();
        if let Some(epoch) = state.epochs.back_mut() {
            epoch.fits.entry(label).or_default().merge(&sums);
        }
    }
}

// ---------------------------------------------------------------------
// Views handed to consumers (panel, export)
// ---------------------------------------------------------------------

/// One timed label (or the whole step) in the current epoch, for the panel.
pub struct LiveRow {
    pub label: &'static str,
    pub calls: u64,
    /// Total time per simulation step: the number to compare across labels,
    /// since labels run at different cadences.
    pub per_step_ms: f32,
    pub mean_call_ms: f32,
    pub max_call_ms: f32,
    /// Per-step times from the flight recorder, oldest first.
    pub history: Vec<f32>,
}

pub struct LiveCounter {
    pub label: &'static str,
    pub per_step: f64,
}

pub struct LiveView {
    pub epoch_id: u32,
    pub epoch_count: usize,
    pub particle_count: usize,
    pub substeps: u32,
    pub dt: f32,
    pub steps: u64,
    /// Whole `Simulation::step`.
    pub step: LiveRow,
    /// Sorted by total time in the epoch, descending.
    pub spans: Vec<LiveRow>,
    /// Sorted by label.
    pub counters: Vec<LiveCounter>,
    pub hardware: LiveHardware,
    /// Sampled least-squares fits, sorted by label.
    pub fits: Vec<LiveFit>,
}

/// Where the current epoch's steps ran (percent of steps; all zero on CPUs
/// that don't report a core type) and the effective clock.
pub struct LiveHardware {
    pub performance_pct: f32,
    pub efficiency_pct: f32,
    pub mixed_pct: f32,
    /// Most recent probe in the recorder.
    pub last_clock_mhz: Option<f32>,
    pub mean_clock_mhz: Option<f32>,
}

/// `ns = intercept + slope * work` for a sampled label.
pub struct LiveFit {
    pub label: &'static str,
    pub samples: u64,
    pub intercept_ns: f64,
    pub slope_ns: f64,
}

/// Aggregate of one label in one epoch. `sum`/`min`/`max` are nanoseconds
/// for a span and raw units for a counter.
pub struct StatSnapshot {
    pub label: &'static str,
    pub calls: u64,
    pub sum: u64,
    pub min: u64,
    pub max: u64,
}

impl StatSnapshot {
    fn new(label: &'static str, s: &SlotStats) -> Self {
        Self {
            label,
            calls: s.calls,
            sum: s.sum,
            min: s.min,
            max: s.max,
        }
    }
}

pub struct EpochSnapshot {
    pub id: u32,
    pub first_step: u64,
    pub steps: u64,
    pub config: serde_json::Value,
    pub particle_count: usize,
    pub substeps: u32,
    pub dt: f32,
    /// Whole-step wall time, labelled `"step"`.
    pub step: StatSnapshot,
    /// Sorted by total time, descending.
    pub spans: Vec<StatSnapshot>,
    /// Sorted by label.
    pub counters: Vec<StatSnapshot>,
    pub hardware: HardwareSnapshot,
    /// Sampled sums per label, sorted by label.
    pub fits: Vec<FitSnapshot>,
}

/// Steps per core type in an epoch, and the clock probe statistics (MHz).
pub struct HardwareSnapshot {
    pub steps_unknown: u64,
    pub steps_performance: u64,
    pub steps_efficiency: u64,
    pub steps_mixed: u64,
    pub clock_probes: u64,
    pub clock_mhz_mean: Option<f64>,
    pub clock_mhz_min: Option<f32>,
    pub clock_mhz_max: Option<f32>,
}

pub struct FitSnapshot {
    pub label: &'static str,
    pub sums: FitSums,
}

/// The flight recorder: the last rows, oldest first, across epochs. Every
/// vector has the same length.
pub struct RecorderSnapshot {
    /// Global index of the first row's step.
    pub first_step: u64,
    pub epoch_ids: Vec<u32>,
    pub step_ms: Vec<f64>,
    /// `core_code` per step: 0 unknown, 1 performance, 2 efficiency, 3 mixed.
    pub core_type: Vec<u8>,
    /// Effective clock per step; 0 for steps that weren't probed.
    pub clock_mhz: Vec<f32>,
    /// Summed time per step, in milliseconds, per span label.
    pub spans: Vec<(&'static str, Vec<f64>)>,
    /// Summed count per step, per counter label.
    pub counters: Vec<(&'static str, Vec<f64>)>,
}

pub struct ExportSnapshot {
    /// Oldest first.
    pub epochs: Vec<EpochSnapshot>,
    pub recorder: RecorderSnapshot,
}
