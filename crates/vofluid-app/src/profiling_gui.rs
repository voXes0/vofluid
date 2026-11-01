//! Live egui panel for the profiler: the current epoch's configuration, the
//! whole step and every timed label with its cost per simulation step and a
//! rolling per-step graph, plus the work counters, the core type and clock the
//! steps got, the sampled per-phase cost fits, and buttons to export the
//! session and to start a new epoch.
//!
//! Storage is `profiling_sink::InMemorySink`; the JSON written by the
//! "Export profile" button is `profiling_export`. This file only draws.
//! `vofluid-core` knows none of this (it only measures), so the measurement
//! layer stays reusable without egui or `serde_json`.

use crate::profiling_export::export_report;
use crate::profiling_sink::{InMemorySink, LiveCounter, LiveFit, LiveRow};

/// Draws a "Profiler" window: epoch header, the whole step, then every label
/// sorted by total time in the epoch (worst offender first), then counters.
/// "Export profile" writes a JSON report (see `profiling_export`); "New
/// epoch" restarts the statistics under the current config, e.g. to discard
/// a warm-up while newly added particles settle.
///
/// Takes `&mut egui::Ui` and gets the `Context` via `.ctx()`, matching
/// `gui::draw_config_panel`'s pattern (the panel is its own floating
/// `egui::Window`, not laid out in the root `Ui` — the `Ui` is only used to
/// reach `ctx()`).
pub fn draw_profiler_panel(root_ui: &mut egui::Ui, sink: &InMemorySink, export_status: &mut Option<String>) {
    let ctx = root_ui.ctx().clone();
    egui::Window::new("Profiler")
        .default_width(340.0)
        .resizable(true)
        .show(&ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Export profile").clicked() {
                    *export_status = Some(match export_report(&sink.export_snapshot()) {
                        Ok(path) => format!("Wrote {}", path.display()),
                        Err(e) => format!("Export failed: {e}"),
                    });
                }
                if ui.button("New epoch").clicked() {
                    sink.start_new_epoch();
                }
                if let Some(status) = export_status {
                    ui.label(egui::RichText::new(status.as_str()).weak());
                }
            });

            ui.separator();

            let Some(view) = sink.live_view() else {
                ui.label("No steps recorded yet.");
                return;
            };

            ui.label(
                egui::RichText::new(format!(
                    "epoch {} ({} kept)  {} particles  {} substeps  dt {}  {} steps",
                    view.epoch_id,
                    view.epoch_count,
                    view.particle_count,
                    view.substeps,
                    view.dt,
                    view.steps,
                ))
                .weak()
                .monospace(),
            );

            // Machine state: a slow epoch on an efficiency core or at a low
            // clock says nothing about the code.
            let hw = &view.hardware;
            if hw.performance_pct + hw.efficiency_pct + hw.mixed_pct > 0.0 {
                ui.label(
                    egui::RichText::new(format!(
                        "cores  P {:.0}%  E {:.0}%  mixed {:.0}%",
                        hw.performance_pct, hw.efficiency_pct, hw.mixed_pct,
                    ))
                    .weak()
                    .monospace(),
                );
            }
            if let Some(last) = hw.last_clock_mhz {
                ui.label(
                    egui::RichText::new(format!(
                        "clock probe  last {:.0} MHz  mean {:.0} MHz",
                        last,
                        hw.mean_clock_mhz.unwrap_or(last),
                    ))
                    .weak()
                    .monospace(),
                );
            }

            ui.separator();

            egui::ScrollArea::vertical()
                .max_height(480.0)
                .show(ui, |ui| {
                    draw_row(ui, &view.step, view.step.per_step_ms);
                    ui.separator();
                    for row in &view.spans {
                        draw_row(ui, row, view.step.per_step_ms);
                        ui.separator();
                    }
                    for counter in &view.counters {
                        draw_counter(ui, counter);
                    }
                    if !view.fits.is_empty() {
                        ui.separator();
                        ui.label(egui::RichText::new("sampled fits: ns = a + b * work").strong());
                        for fit in &view.fits {
                            draw_fit(ui, fit);
                        }
                    }
                });
        });
}

fn draw_row(ui: &mut egui::Ui, row: &LiveRow, step_ms: f32) {
    // Per-step cost is the number to compare across labels: labels run at
    // different cadences (once per step vs once per substep), so their
    // per-call averages are not comparable.
    let share = if step_ms > 0.0 {
        row.per_step_ms / step_ms * 100.0
    } else {
        0.0
    };
    ui.label(egui::RichText::new(row.label).monospace().strong());
    ui.label(
        egui::RichText::new(format!(
            "{:.3} ms/step ({:.1}% of step)  call avg {:.4}  max {:.3}  n={}",
            row.per_step_ms, share, row.mean_call_ms, row.max_call_ms, row.calls,
        ))
        .weak()
        .monospace(),
    );
    draw_history_graph(ui, &row.history);
}

fn draw_counter(ui: &mut egui::Ui, counter: &LiveCounter) {
    ui.label(
        egui::RichText::new(format!("{}: {:.0}/step", counter.label, counter.per_step))
            .weak()
            .monospace(),
    );
}

fn draw_fit(ui: &mut egui::Ui, fit: &LiveFit) {
    ui.label(
        egui::RichText::new(format!(
            "{}: {:.1} + {:.3} * work  (n={})",
            fit.label, fit.intercept_ns, fit.slope_ns, fit.samples,
        ))
        .weak()
        .monospace(),
    );
}

/// Same rolling-graph approach as `gui::draw_sim_history_graph`, but reused
/// generically per label here rather than hardcoded to the total sim-step
/// timer. Raw painter, no plotting crate; the Y axis is auto-scaled to each
/// label's own recent range so a cheap label's tiny fluctuations are still
/// visible instead of flatlining next to an expensive one.
fn draw_history_graph(ui: &mut egui::Ui, history: &[f32]) {
    let (response, painter) =
        ui.allocate_painter(egui::Vec2::new(ui.available_width(), 32.0), egui::Sense::hover());
    let rect = response.rect;

    painter.rect_filled(rect, 2.0, egui::Color32::from_black_alpha(60));

    if history.len() < 2 {
        return;
    }

    let min = history.iter().copied().fold(f32::MAX, f32::min);
    let max = history.iter().copied().fold(f32::MIN, f32::max);
    let range = (max - min).max(0.001);
    let padded_min = min - range * 0.1;
    let padded_max = max + range * 0.1;
    let padded_range = (padded_max - padded_min).max(0.001);

    let to_y = |v: f32| -> f32 {
        let t = (v - padded_min) / padded_range;
        rect.bottom() - t.clamp(0.0, 1.0) * rect.height()
    };

    let n = history.len();
    let points: Vec<egui::Pos2> = history
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let t = i as f32 / (n - 1).max(1) as f32;
            egui::Pos2::new(rect.left() + t * rect.width(), to_y(v))
        })
        .collect();

    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(1.2, egui::Color32::from_rgb(140, 200, 255)),
    ));
}
