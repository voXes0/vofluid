use std::{sync::Arc, vec};

use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::mpsc;
use vofluid_core::{config::SimConfig, sim::Simulation};
use vofluid_render::{ParticleRenderer, RendererConfig};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::gui::{Gui, GuiState, draw_config_panel, draw_fps_overlay};

#[cfg(feature = "profile")]
use crate::profiling_gui::draw_profiler_panel;
#[cfg(feature = "profile")]
use crate::profiling_sink::InMemorySink;

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: Arc<Window>,
}

impl GpuState {
    async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::defaults(),
                ..Default::default()
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            window,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
        }
    }
}

// ── App ───────────────────────────────────────────────────────────────────────

pub struct App {
    gpu: Option<GpuState>,
    renderer: Option<ParticleRenderer>,
    gui: Option<Gui>,
    gui_state: GuiState,
    frame_stats: FrameStats,
    #[cfg(feature = "profile")]
    profiler_sink: std::sync::Arc<InMemorySink>,
    #[cfg(feature = "profile")]
    profiler_export_status: Option<String>,
    /// Wall-clock timestamp of the last `RedrawRequested`.
    last_tick: std::time::Instant,
    sim: Simulation,
    config_rx: mpsc::Receiver<notify::Result<Event>>,
    _watcher: RecommendedWatcher,
    target_frame_time: std::time::Duration,
}

/// Exponential moving average of a duration, in seconds. Reusable for
/// timing any per-frame phase (sim step, render, total wall time, ...)
/// without jittering wildly from frame to frame.
#[derive(Default)]
struct SmoothedTimer {
    smoothed_secs: f32,
}

impl SmoothedTimer {
    const SMOOTHING: f32 = 0.1;

    fn record(&mut self, dt: std::time::Duration) {
        let dt = dt.as_secs_f32();
        if self.smoothed_secs == 0.0 {
            self.smoothed_secs = dt;
        } else {
            self.smoothed_secs += (dt - self.smoothed_secs) * Self::SMOOTHING;
        }
    }

    fn ms(&self) -> f32 {
        self.smoothed_secs * 1000.0
    }

    fn hz(&self) -> f32 {
        if self.smoothed_secs > 0.0 {
            1.0 / self.smoothed_secs
        } else {
            0.0
        }
    }
}

/// Per-frame timing breakdown: how long the sim step took vs the render
struct FrameStats {
    last_frame_start: std::time::Instant,
    total: SmoothedTimer,
    sim_step: SmoothedTimer,
    render: SmoothedTimer,
    /// Ring buffer of the last `SIM_HISTORY_LEN` raw sim-step timings, in milliseconds.
    sim_history: std::collections::VecDeque<f32>,
}

const SIM_HISTORY_LEN: usize = 240;

impl FrameStats {
    fn new() -> Self {
        Self {
            last_frame_start: std::time::Instant::now(),
            total: SmoothedTimer::default(),
            sim_step: SmoothedTimer::default(),
            render: SmoothedTimer::default(),
            sim_history: std::collections::VecDeque::with_capacity(SIM_HISTORY_LEN),
        }
    }

    /// Measures wall-clock time since the previous frame started
    fn begin_frame(&mut self) {
        let now = std::time::Instant::now();
        self.total.record(now - self.last_frame_start);
        self.last_frame_start = now;
    }

    fn record_sim_step(&mut self, dt: std::time::Duration) {
        self.sim_step.record(dt);

        let ms = dt.as_secs_f32() * 1000.0;
        if self.sim_history.len() == SIM_HISTORY_LEN {
            self.sim_history.pop_front();
        }
        self.sim_history.push_back(ms);
    }

    fn record_render(&mut self, dt: std::time::Duration) {
        self.render.record(dt);
    }

    /// (min, max, avg) over the current history window, in ms. Returns
    /// `None` if no samples have been recorded yet.
    fn sim_history_stats(&self) -> Option<(f32, f32, f32)> {
        if self.sim_history.is_empty() {
            return None;
        }
        let min = self.sim_history.iter().copied().fold(f32::MAX, f32::min);
        let max = self.sim_history.iter().copied().fold(f32::MIN, f32::max);
        let avg = self.sim_history.iter().sum::<f32>() / self.sim_history.len() as f32;
        Some((min, max, avg))
    }
}

// Note to myself: `record_render`'s input only covers CPU-side work,
// not the actual GPU time, for example performed for shaders.
impl App {
    /// Sets up the app state with blank fields
    pub fn new(
        sim: Simulation,
        #[cfg(feature = "profile")] profiler_sink: Arc<InMemorySink>,
    ) -> Self {
        // Config asset hotreload, Notify crate watches the file and sends it down stream, here it is set up to watch
        let (tx, config_rx) = mpsc::channel::<notify::Result<Event>>();
        let mut watcher = RecommendedWatcher::new(tx, notify::Config::default())
            .expect("failed to create watcher");
        watcher
            .watch(Path::new("assets/default.ron"), RecursiveMode::NonRecursive)
            .expect("failed to watch config file");

        Self {
            gpu: None,
            renderer: None,
            gui: None,
            gui_state: GuiState::new(),
            frame_stats: FrameStats::new(),
            #[cfg(feature = "profile")]
            profiler_sink,
            #[cfg(feature = "profile")]
            profiler_export_status: None,
            last_tick: std::time::Instant::now(),
            sim,
            config_rx,
            _watcher: watcher,
            target_frame_time: std::time::Duration::from_secs_f64(1.0 / 120.0),
        }
    }

    /// Renders a frame of the app, returns if `gpu` or `render` are none or `is_surface_configured` is false
    fn render_frame(&mut self, event_loop: &ActiveEventLoop) {
        let gpu = match &mut self.gpu {
            Some(g) => g,
            None => return,
        };
        let renderer = match &mut self.renderer {
            Some(r) => r,
            None => return,
        };

        if !gpu.is_surface_configured {
            return;
        }

        let output = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                t
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
            wgpu::CurrentSurfaceTexture::Lost => {
                log::error!("Lost device");
                event_loop.exit();
                return;
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.025,
                            g: 0.025,
                            b: 0.025,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            renderer.render(&mut pass);
        }

        if let Some(gui) = &mut self.gui {
            let sim = &mut self.sim;
            let gui_state = &mut self.gui_state;
            let render_ms = self.frame_stats.render.ms();
            let total_fps = self.frame_stats.total.hz();
            // Contiguous copy of the ring buffer for the graph. Cheap:
            // capped at SIM_HISTORY_LEN (240) f32s, once per frame.
            let sim_history: Vec<f32> = self.frame_stats.sim_history.iter().copied().collect();
            let sim_stats = self.frame_stats.sim_history_stats();
            #[cfg(feature = "profile")]
            let profiler_sink = &self.profiler_sink;
            #[cfg(feature = "profile")]
            let export_status = &mut self.profiler_export_status;
            gui.render(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                &gpu.window,
                &view,
                gpu.config.width,
                gpu.config.height,
                |ctx| {
                    draw_fps_overlay(ctx, total_fps, render_ms, &sim_history, sim_stats);
                    draw_config_panel(ctx, sim, gui_state);
                    #[cfg(feature = "profile")]
                    draw_profiler_panel(ctx, profiler_sink, export_status);
                },
            );
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        gpu.queue.present(output);
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        let window = match &self.gpu {
            Some(g) => &g.window,
            None => return,
        };
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            (KeyCode::KeyF, true) => {
                if window.fullscreen().is_some() {
                    window.set_fullscreen(None);
                } else {
                    window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                }
            }
            (KeyCode::Space, true) => {
                if self.gui_state.is_paused() {
                    self.gui_state.start(&mut self.sim);
                } else {
                    self.gui_state.pause(&mut self.sim);
                }
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Vofluid")
            .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let gpu = pollster::block_on(GpuState::new(window)).expect("failed to create GPU state");

        let renderer = ParticleRenderer::new(
            &gpu.device,
            &self.sim.particles.positions,
            &self.sim.particles.vels,
            RendererConfig {
                shader: None,
                target_format: gpu.config.format,
                sample_count: 1,
                depth_format: None,
                particle_radius: 5.0,
                viewport_size: [gpu.config.width as f32, gpu.config.height as f32],
                blend_state: wgpu::BlendState::ALPHA_BLENDING,
            },
        );

        let gui = Gui::new(&gpu.device, &gpu.window, gpu.config.format);

        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.gui = Some(gui);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if self.gpu.is_none() {
            return;
        }

        let egui_consumed = match (&mut self.gui, &self.gpu) {
            (Some(gui), Some(gpu)) => gui.handle_window_event(&gpu.window, &event),
            _ => false,
        };

        if let Ok(Ok(notify_event)) = self.config_rx.try_recv() {
            use notify::EventKind::*;
            if matches!(notify_event.kind, Modify(_) | Create(_),) {
                let old_particle_count = self.sim.config.particle_count;
                match load_sim_config("assets/default.ron") {
                    Ok(cfg) => {
                        self.sim.config = cfg;
                        if self.sim.config.particle_count != old_particle_count {
                            self.sim.initialize_particles();
                        }
                    }
                    Err(e) => eprintln!("Failed to reload config: {e}"),
                }
            }
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                }
                if let (Some(gpu), Some(renderer)) = (&self.gpu, &mut self.renderer) {
                    renderer.set_viewport_size(
                        &gpu.queue,
                        gpu.config.width as f32,
                        gpu.config.height as f32,
                    );
                }
            }
            WindowEvent::RedrawRequested => {
                // Frame cap bail out early until at least target_frame_time has passed since the
                {
                    let since_last = std::time::Instant::now() - self.frame_stats.last_frame_start;
                    if since_last < self.target_frame_time {
                        if let Some(gpu) = &self.gpu {
                            gpu.window.request_redraw();
                        }
                        return;
                    }
                }

                self.frame_stats.begin_frame();

                // Advance simulated time by however much wall-clock time
                // actually passed, rather than calling `step()` once per render
                {
                    let now = std::time::Instant::now();
                    let elapsed = (now - self.last_tick).as_secs_f32();
                    self.last_tick = now;

                    let report = self.sim.advance(elapsed);
                    if let Some(avg_secs) = report.avg_step_time_secs() {
                        self.frame_stats
                            .record_sim_step(std::time::Duration::from_secs_f32(avg_secs));
                    }
                }

                let render_t0 = std::time::Instant::now();

                // Upload latest particle positions then draw.
                if let (Some(gpu), Some(renderer)) = (&self.gpu, &mut self.renderer) {
                    renderer.upload_positions(
                        &gpu.device,
                        &gpu.queue,
                        &self.sim.particles.positions,
                    );
                }
                self.render_frame(event_loop);

                self.frame_stats.record_render(render_t0.elapsed());

                if let Some(gpu) = &self.gpu {
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } if !egui_consumed => self.handle_key(event_loop, code, key_state.is_pressed()),
            _ => {}
        }
    }
}

// ── entry points ──────────────────────────────────────────────────────────────

fn load_sim_config(path: &str) -> anyhow::Result<SimConfig> {
    let text = std::fs::read_to_string(path)?;
    Ok(vofluid_core::ron::from_str(&text)?)
}

pub fn run() -> anyhow::Result<()> {
    env_logger::init();

    #[cfg(feature = "profile")]
    let profiler_sink = vofluid_core::profiling::install_sink(InMemorySink::new());

    let config = load_sim_config("assets/default.ron").expect("could not load config");
    let sim = Simulation::new(
        vofluid_core::backend::Backend::Single(vofluid_core::backends::single::SingleThread::new(
            &config,
        )),
        config,
    );

    let event_loop = EventLoop::new()?;
    #[cfg(feature = "profile")]
    let mut app = App::new(sim, profiler_sink);
    #[cfg(not(feature = "profile"))]
    let mut app = App::new(sim);
    event_loop.run_app(&mut app)?;

    Ok(())
}
