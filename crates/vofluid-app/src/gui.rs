use std::ops::MulAssign;

use egui::Vec2;
use egui_wgpu::{RendererOptions, ScreenDescriptor};
use vofluid_core::sim::Simulation;
use winit::{event::WindowEvent, window::Window};

/// Thin wrapper around the egui <-> winit <-> wgpu plumbing.
///
/// Owns everything needed to turn winit events into egui input, run the UI
/// closure, and paint the result on top of the particle render pass. None of
/// this touches `Simulation` directly except through `Gui::ui`, which is
/// where the actual `SimConfig` panel lives.
pub struct Gui {
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl Gui {
    pub fn new(device: &wgpu::Device, window: &Window, output_format: wgpu::TextureFormat) -> Self {
        let ctx = egui::Context::default();

        let state = egui_winit::State::new(
            ctx,
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            Some(device.limits().max_texture_dimension_2d as usize),
        );

        let renderer = egui_wgpu::Renderer::new(
            device,
            output_format,
            RendererOptions {
                msaa_samples: 1,
                ..Default::default()
            },
        );

        Self { state, renderer }
    }

    /// Feed a winit window event to egui. Returns `true` if egui consumed
    /// the event (i.e. the app shouldn't act on it further, e.g. a click
    /// that landed on a slider rather than the simulation viewport).
    pub fn handle_window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.state.on_window_event(window, event).consumed
    }

    /// Runs the UI closure, tessellates the result, uploads buffers/textures,
    /// and paints onto `pass`. `pass` is expected to already be mid-render
    /// (cleared, with the particle draw already issued) — egui paints on top.
    ///
    /// `run_ui` receives a `&mut Ui` covering the whole content area (current
    /// egui API — `Context::run` was removed in favor of `Context::run_ui`).
    /// Build your own `egui::Window`/panels off of `ui.ctx()` inside it.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        window: &Window,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        run_ui: impl FnMut(&mut egui::Ui),
    ) {
        let raw_input = self.state.take_egui_input(window);
        let ctx = self.state.egui_ctx().clone();

        let full_output = ctx.run_ui(raw_input, run_ui);

        self.state
            .handle_platform_output(window, full_output.platform_output);

        let pixels_per_point = full_output.pixels_per_point;
        let paint_jobs = ctx.tessellate(full_output.shapes, pixels_per_point);

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }

        self.renderer
            .update_buffers(device, queue, encoder, &paint_jobs, &screen_descriptor);

        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();

            self.renderer
                .render(&mut pass, &paint_jobs, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}

/// Tracks gui_state/resume state for the substeps field without losing whatever
/// value the user had configured before pausing.
pub struct GuiState {
    /// `substeps` value to restore when resuming. `None` while running.
    paused_substeps: Option<u32>,
}

impl GuiState {
    pub fn new() -> Self {
        Self {
            paused_substeps: None,
        }
    }

    pub fn is_paused(&self) -> bool {
        self.paused_substeps.is_some()
    }

    pub fn gui_state(&mut self, sim: &mut Simulation) {
        if self.paused_substeps.is_none() {
            self.paused_substeps = Some(sim.config.substeps);
            sim.config.substeps = 0;
        }
    }

    pub fn start(&mut self, sim: &mut Simulation) {
        if let Some(prev) = self.paused_substeps.take() {
            sim.config.substeps = prev;
        }
    }

    pub fn toggle(&mut self, sim: &mut Simulation) {
        if self.is_paused() {
            self.start(sim);
        } else {
            self.gui_state(sim);
        }
    }
}

/// Draws a small always-on-top FPS/frame-time readout in the corner of the
/// screen. Kept separate from `draw_config_panel` so it stays visible even
/// if the config window is collapsed, moved, or closed.
///
/// The sim step is shown as a rolling min/max/avg graph rather than a single
/// smoothed number: an EMA hides exactly the spikes/stutter you want to see
/// when hunting a perf regression at high particle counts. `render_ms` (CPU
/// encode/submit time) stays a single smoothed figure since it's normally
/// steady and isn't what you're currently chasing.
pub fn draw_fps_overlay(
    ctx: &egui::Context,
    total_fps: f32,
    render_ms: f32,
    sim_history_ms: &[f32],
    sim_stats: Option<(f32, f32, f32)>,
) {
    egui::Area::new(egui::Id::new("fps_overlay"))
        .anchor(egui::Align2::RIGHT_TOP, Vec2::new(-8.0, 8.0))
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style())
                .fill(egui::Color32::from_black_alpha(180))
                .show(ui, |ui| {
                    ui.set_width(220.0);
                    ui.vertical(|ui| {
                        ui.label(
                            egui::RichText::new(format!("{total_fps:.0} FPS"))
                                .color(egui::Color32::WHITE)
                                .monospace()
                                .strong(),
                        );
                        ui.label(
                            egui::RichText::new(format!("render: {render_ms:.2} ms"))
                                .color(egui::Color32::from_rgb(140, 200, 255))
                                .monospace(),
                        );

                        ui.add_space(4.0);

                        if let Some((min, max, avg)) = sim_stats {
                            ui.label(
                                egui::RichText::new(format!(
                                    "sim  min {min:.2}  avg {avg:.2}  max {max:.2} ms"
                                ))
                                .color(egui::Color32::from_rgb(255, 200, 120))
                                .monospace(),
                            );
                        } else {
                            ui.label(
                                egui::RichText::new("sim: warming up…")
                                    .color(egui::Color32::from_rgb(255, 200, 120))
                                    .monospace(),
                            );
                        }

                        draw_sim_history_graph(ui, sim_history_ms, sim_stats);
                    });
                });
        });
}

/// Paints a rolling sim-step-time graph via the raw painter (no plotting
/// crate dependency). Y axis auto-scales to the current window's min/max
/// each frame so spikes are always visible regardless of overall scale.
fn draw_sim_history_graph(ui: &mut egui::Ui, history_ms: &[f32], stats: Option<(f32, f32, f32)>) {
    let (response, painter) =
        ui.allocate_painter(Vec2::new(ui.available_width(), 60.0), egui::Sense::hover());
    let rect = response.rect;

    painter.rect_filled(rect, 2.0, egui::Color32::from_black_alpha(60));

    if history_ms.len() < 2 {
        return;
    }

    let Some((min, max, avg)) = stats else {
        return;
    };

    // Pad the range a bit so the line doesn't hug the top/bottom edge, and
    // guard against a degenerate (flat) range.
    let range = (max - min).max(0.001);
    let padded_min = min - range * 0.1;
    let padded_max = max + range * 0.1;
    let padded_range = (padded_max - padded_min).max(0.001);

    let to_y = |ms: f32| -> f32 {
        let t = (ms - padded_min) / padded_range;
        rect.bottom() - t.clamp(0.0, 1.0) * rect.height()
    };

    // Average line, drawn first so the samples/spikes sit on top of it.
    let avg_y = to_y(avg);
    painter.hline(
        rect.x_range(),
        avg_y,
        egui::Stroke::new(
            1.0,
            egui::Color32::from_rgba_unmultiplied(255, 200, 120, 120),
        ),
    );

    let n = history_ms.len();
    let points: Vec<egui::Pos2> = history_ms
        .iter()
        .enumerate()
        .map(|(i, &ms)| {
            let t = i as f32 / (n - 1).max(1) as f32;
            let x = rect.left() + t * rect.width();
            egui::Pos2::new(x, to_y(ms))
        })
        .collect();

    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 200, 120)),
    ));
}

/// Draws the `SimConfig` tuning panel plus Reset/Pause/Start controls.
///
/// Takes the `&mut Ui` handed to us by `Context::run_ui` (current egui API;
/// `Context::run` was removed). We only use it to get at `ui.ctx()` since the
/// panel is its own floating `egui::Window`, not laid out in the root `Ui`.
///
/// Mirrors how the file-watcher reload path (`app.rs`) reacts to a changed
/// `particle_count`: just call `sim.initialize_particles()` again. There's
/// no separate "apply" step — every field is live-edited in place via egui's
/// widgets, which write directly into `sim.config`.
pub fn draw_config_panel(root_ui: &mut egui::Ui, sim: &mut Simulation, gui_state: &mut GuiState) {
    let ctx = root_ui.ctx().clone();
    egui::Window::new("Simulation Config")
        .default_width(280.0)
        .resizable(true)
        .show(&ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Reset").clicked() {
                    sim.initialize_particles();
                }
                if gui_state.is_paused() {
                    if ui.button("Start").clicked() {
                        gui_state.start(sim);
                    }
                } else if ui.button("Pause").clicked() {
                    gui_state.gui_state(sim);
                }
            });

            ui.separator();

            ui.add_enabled_ui(!gui_state.is_paused(), |ui| {
                ui.add(egui::Slider::new(&mut sim.config.substeps, 0..=16).text("substeps"));
            });

            ui.add(
                egui::DragValue::new(&mut sim.config.dt)
                    .speed(0.0001)
                    .range(0.0001..=1.0)
                    .prefix("dt: "),
            );

            ui.separator();

            let mut particle_count = sim.config.particle_count;
            let particle_count_changed = ui
                .add(
                    egui::DragValue::new(&mut particle_count)
                        .speed(1.0)
                        .range(1..=200_000)
                        .prefix("particle_count: "),
                )
                .changed();
            if particle_count_changed {
                sim.config.particle_count = particle_count;
                sim.ensure_particle_count();
            }

            ui.separator();

            ui.add(
                egui::DragValue::new(&mut sim.config.rest_density)
                    .speed(0.1)
                    .range(0.0..=1000.0)
                    .prefix("rest_density: "),
            );
            ui.add(
                egui::DragValue::new(&mut sim.config.stiffness)
                    .speed(0.1)
                    .range(0.0..=1000.0)
                    .prefix("stiffness: "),
            );
            ui.add(
                egui::DragValue::new(&mut sim.config.stiffness_near)
                    .speed(0.1)
                    .range(0.0..=1000.0)
                    .prefix("stiffness_near: "),
            );
            ui.add(
                egui::DragValue::new(&mut sim.config.interaction_radius)
                    .speed(0.5)
                    .range(0.0..=2000.0)
                    .prefix("interaction_radius: "),
            );

            ui.separator();

            ui.add(
                egui::DragValue::new(&mut sim.config.yield_ratio_stretch)
                    .speed(0.01)
                    .range(0.0..=10.0)
                    .prefix("yield_ratio_stretch: "),
            );
            ui.add(
                egui::DragValue::new(&mut sim.config.yield_ratio_compress)
                    .speed(0.01)
                    .range(0.0..=10.0)
                    .prefix("yield_ratio_compress: "),
            );
            ui.add(
                egui::DragValue::new(&mut sim.config.plasticity)
                    .speed(0.1)
                    .range(0.0..=1000.0)
                    .prefix("plasticity: "),
            );
            ui.add(
                egui::DragValue::new(&mut sim.config.spring_stiffness)
                    .speed(1.0)
                    .range(0.0..=10_000.0)
                    .prefix("spring_stiffness: "),
            );

            ui.separator();

            ui.label("gravity");
            ui.horizontal(|ui| {
                ui.add(egui::Slider::new(&mut sim.config.gravity.x, -29.43..=2994.3).prefix("x: "));
                ui.add(egui::Slider::new(&mut sim.config.gravity.y, -29.43..=2994.3).prefix("y: "));
            });

            ui.label("bounds");
            ui.horizontal(|ui| {
                let mut bounds = sim.config.bounds;
                let bounds_changed = ui
                    .add(
                        egui::Slider::new(&mut bounds.x, sim.config.interaction_radius..=1000.0)
                            .prefix("x: "),
                    )
                    .changed()
                    || ui
                        .add(
                            egui::Slider::new(
                                &mut bounds.y,
                                sim.config.interaction_radius..=1000.0,
                            )
                            .prefix("y: "),
                        )
                        .changed();

                if bounds_changed {
                    sim.config.bounds = bounds;
                    sim.resize_bounds(bounds);
                }
            });
        });
}
