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

#[cfg(not(target_arch = "wasm32"))]
use crate::gui::{Gui, GuiState, draw_config_panel};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use winit::{event_loop, platform::web::EventLoopExtWebSys};

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
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
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
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::defaults()
                },
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
    #[cfg(target_arch = "wasm32")]
    proxy: Option<event_loop::EventLoopProxy<(GpuState, ParticleRenderer)>>,
    gpu: Option<GpuState>,
    renderer: Option<ParticleRenderer>,
    #[cfg(not(target_arch = "wasm32"))]
    gui: Option<Gui>,
    #[cfg(not(target_arch = "wasm32"))]
    gui_state: GuiState,
    sim: Simulation,
    config_rx: mpsc::Receiver<notify::Result<Event>>,
    _watcher: RecommendedWatcher,
}

impl App {
    pub fn new(
        sim: Simulation,
        #[cfg(target_arch = "wasm32")] event_loop: &EventLoop<(GpuState, ParticleRenderer)>,
    ) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());

        let (tx, config_rx) = mpsc::channel::<notify::Result<Event>>();
        let mut watcher = RecommendedWatcher::new(tx, notify::Config::default())
            .expect("failed to create watcher");
        watcher
            .watch(Path::new("assets/default.ron"), RecursiveMode::NonRecursive)
            .expect("failed to watch config file");

        Self {
            #[cfg(target_arch = "wasm32")]
            proxy,
            gpu: None,
            renderer: None,
            #[cfg(not(target_arch = "wasm32"))]
            gui: None,
            #[cfg(not(target_arch = "wasm32"))]
            gui_state: GuiState::new(),
            sim,
            config_rx,
            _watcher: watcher,
        }
    }

    fn render_frame(&mut self, event_loop: &ActiveEventLoop) {
        let gpu = match &mut self.gpu {
            Some(g) => g,
            None => return,
        };
        let renderer = match &mut self.renderer {
            Some(r) => r,
            None => return,
        };

        gpu.window.request_redraw();

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

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(gui) = &mut self.gui {
            let sim = &mut self.sim;
            let gui_state = &mut self.gui_state;
            gui.render(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                &gpu.window,
                &view,
                gpu.config.width,
                gpu.config.height,
                |ctx| draw_config_panel(ctx, sim, gui_state),
            );
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        let window = match &self.gpu {
            Some(g) => &g.window,
            None => return,
        };
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            (KeyCode::KeyW, true) => {
                if window.fullscreen().is_some() {
                    window.set_fullscreen(None);
                } else {
                    window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                }
            }
            _ => {}
        }
    }
}

impl ApplicationHandler<(GpuState, ParticleRenderer)> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut attrs = Window::default_attributes()
            .with_title("Vofluid")
            .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";
            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            attrs = attrs.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            let gpu =
                pollster::block_on(GpuState::new(window)).expect("failed to create GPU state");

            let renderer = ParticleRenderer::new(
                &gpu.device,
                &self.sim.particles.positions.0,
                &self.sim.particles.vels.0,
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

        #[cfg(target_arch = "wasm32")]
        {
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    let mut gpu = GpuState::new(window)
                        .await
                        .expect("failed to create GPU state");

                    let renderer = ParticleRenderer::new(
                        &gpu.device,
                        &particle_instances,
                        RendererConfig {
                            shader: None,
                            target_format: gpu.config.format,
                            sample_count: 1,
                            depth_format: None,
                            particle_radius: 10.0,
                            viewport_size: [gpu.config.width as f32, gpu.config.height as f32],
                            blend_state: wgpu::BlendState::ALPHA_BLENDING,
                        },
                    );

                    proxy
                        .send_event((gpu, renderer))
                        .expect("Event loop closed");
                });
            }
        }
    }

    fn user_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        (gpu, renderer): (GpuState, ParticleRenderer),
    ) {
        #[cfg(target_arch = "wasm32")]
        {
            let size = gpu.window.inner_size();
            gpu.window.request_redraw();
            gpu.resize(size.width, size.height);
        }
        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
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

        #[cfg(not(target_arch = "wasm32"))]
        let egui_consumed = match (&mut self.gui, &self.gpu) {
            (Some(gui), Some(gpu)) => gui.handle_window_event(&gpu.window, &event),
            _ => false,
        };
        #[cfg(target_arch = "wasm32")]
        let egui_consumed = false;

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
                self.sim.step();

                // Upload latest particle positions then draw.
                if let (Some(gpu), Some(renderer)) = (&self.gpu, &mut self.renderer) {
                    renderer.upload_positions(
                        &gpu.device,
                        &gpu.queue,
                        &self.sim.particles.positions.0,
                    );
                }
                self.render_frame(event_loop);
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
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();

    #[cfg(target_arch = "wasm32")]
    console_log::init_with_level(log::Level::Info).unwrap();
    let config = load_sim_config("assets/default.ron").expect("could not load config");
    let sim = Simulation::new(
        vofluid_core::backend::Backend::Single(vofluid_core::backends::single::SingleThread::new(
            &config,
        )),
        config,
    );

    #[cfg(not(target_arch = "wasm32"))]
    {
        let event_loop = EventLoop::with_user_event().build()?;
        let mut app = App::new(sim);
        event_loop.run_app(&mut app)?;
    }

    #[cfg(target_arch = "wasm32")]
    {
        let event_loop = EventLoop::with_user_event().build()?;
        let app = App::new(sim, &event_loop);
        event_loop.spawn_app(app);
    }

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();
    Ok(())
}
