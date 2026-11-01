// vofluid-web: browser entry point for vofluid.
//
// Rust owns the entire render loop here (via requestAnimationFrame,
// scheduled from within Rust) rather than JS driving per-frame calls into
// wasm. See the discussion that led here: this keeps the JS/wasm boundary
// crossed once per frame instead of twice (step + draw), and avoids
// duplicating the event-loop logic vofluid-app already has natively.
//
// NOTE: this calls `sim.step()` once per animation frame, not
// `sim.advance(dt)`. `Simulation::advance` uses `std::time::Instant::now()`
// internally for step timing, which panics on wasm32-unknown-unknown (the
// std time syscalls aren't implemented for this target). vofluid-core
// itself hasn't been changed to work around this — see conversation notes.
// Practical effect: sim speed is tied to the browser's actual frame rate
// rather than wall-clock time. Fine for this early test; revisit if a
// dropped-frames slowdown becomes a problem.
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;

use vofluid_core::backend::Backend;
use vofluid_core::backends::single::SingleThread;
use vofluid_core::config::SimConfig;
use vofluid_core::sim::Simulation;
use vofluid_render::{ParticleRenderer, RendererConfig};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

const CANVAS_ID: &str = "canvas";

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
}

impl GpuState {
    async fn new(canvas: web_sys::HtmlCanvasElement) -> Result<Self, JsValue> {
        let width = canvas.width();
        let height = canvas.height();

        // WebGPU first; browsers without it (older Safari, some Firefox
        // versions) fall through to WebGL2 via wgpu's GL backend. Listing
        // both lets wgpu's instance creation pick whichever the browser
        // actually supports rather than hard-failing on WebGPU absence.
        // InstanceDescriptor has no Default impl in this wgpu version (see
        // vofluid-app::app::GpuState::new for the same pattern natively),
        // so every field is set explicitly.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| JsValue::from_str(&format!("create_surface failed: {e}")))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("request_adapter failed: {e}")))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                // WebGL2 has much lower limits than native; downlevel
                // defaults are required there and harmless if the WebGPU
                // backend ends up being used instead.
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                ..Default::default()
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("request_device failed: {e}")))?;

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
            width,
            height,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Ok(Self {
            surface,
            device,
            queue,
            config,
        })
    }
}

struct AppState {
    gpu: GpuState,
    renderer: ParticleRenderer,
    sim: Simulation,
}

fn get_canvas() -> Result<web_sys::HtmlCanvasElement, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let document = window
        .document()
        .ok_or_else(|| JsValue::from_str("no document"))?;
    let canvas = document
        .get_element_by_id(CANVAS_ID)
        .ok_or_else(|| JsValue::from_str(&format!("no element with id '{CANVAS_ID}'")))?;
    canvas
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| JsValue::from_str("element is not a canvas"))
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .expect("no window")
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("request_animation_frame failed");
}

#[wasm_bindgen(start)]
pub fn run() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).unwrap_or(());

    let canvas = get_canvas()?;

    wasm_bindgen_futures::spawn_local(async move {
        let gpu = match GpuState::new(canvas).await {
            Ok(g) => g,
            Err(e) => {
                log::error!("GPU init failed: {e:?}");
                return;
            }
        };

        // TODO: load from assets/default.ron once config loading is wired
        // up for wasm (fetch instead of std::fs::read_to_string). Defaults
        // for now to get something on screen.
        let config = SimConfig::default();
        let sim = Simulation::new(Backend::Single(SingleThread::new(&config)), config);

        let renderer = ParticleRenderer::new(
            &gpu.device,
            &sim.particles.positions.0,
            &sim.particles.vels.0,
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

        let state = Rc::new(RefCell::new(AppState { gpu, renderer, sim }));

        // Self-rescheduling requestAnimationFrame loop, entirely on the
        // Rust side. `f` is wrapped in Rc<RefCell<Option<Closure>>> so the
        // closure can reference and reschedule itself.
        let f = Rc::new(RefCell::new(None::<Closure<dyn FnMut()>>));
        let g = f.clone();

        *g.borrow_mut() = Some(Closure::new(move || {
            {
                let mut state = state.borrow_mut();
                let AppState { gpu, renderer, sim } = &mut *state;

                sim.step();

                renderer.upload_positions(&gpu.device, &gpu.queue, &sim.particles.positions.0);

                match gpu.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(output) => {
                        let view = output
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
                        let mut encoder =
                            gpu.device
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
                        gpu.queue.submit(std::iter::once(encoder.finish()));
                        output.present();
                    }
                    wgpu::CurrentSurfaceTexture::Suboptimal(output) => {
                        gpu.surface.configure(&gpu.device, &gpu.config);
                        output.present();
                    }
                    wgpu::CurrentSurfaceTexture::Outdated => {
                        gpu.surface.configure(&gpu.device, &gpu.config);
                    }
                    wgpu::CurrentSurfaceTexture::Timeout
                    | wgpu::CurrentSurfaceTexture::Occluded
                    | wgpu::CurrentSurfaceTexture::Validation => {}
                    wgpu::CurrentSurfaceTexture::Lost => {
                        log::error!("Surface lost");
                    }
                }
            }

            request_animation_frame(f.borrow().as_ref().unwrap());
        }));

        request_animation_frame(g.borrow().as_ref().unwrap());
    });

    Ok(())
}
