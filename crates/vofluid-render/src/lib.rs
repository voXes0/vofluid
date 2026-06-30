use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ── uniform layout ────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Globals {
    window_size: [f32; 2],
    particle_radius: f32,
    _pad: f32,
}

// ── config the caller provides at construction ────────────────────────────────

pub struct RendererConfig {
    pub shader: Option<wgpu::ShaderModule>,
    pub target_format: wgpu::TextureFormat,
    pub sample_count: u32,
    pub depth_format: Option<wgpu::TextureFormat>,
    pub particle_radius: f32,
    pub viewport_size: [f32; 2],
    pub blend_state: wgpu::BlendState,
}

// ── renderer ──────────────────────────────────────────────────────────────────

pub struct ParticleRenderer {
    render_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    globals_buffer: wgpu::Buffer,
    position_buffer: wgpu::Buffer,
    velocity_buffer: wgpu::Buffer,
    particle_count: u32,
    buffer_capacity: u32,
    bind_group_layout: wgpu::BindGroupLayout,
    config: RendererConfig,
}

impl ParticleRenderer {
    pub fn new(
        device: &wgpu::Device,
        positions: &[glam::Vec2],
        velocities: &[glam::Vec2],
        mut config: RendererConfig,
    ) -> Self {
        assert_eq!(
            positions.len(),
            velocities.len(),
            "positions and velocities must have matching length"
        );

        let globals = Globals {
            window_size: config.viewport_size,
            particle_radius: config.particle_radius,
            _pad: 0.0,
        };

        let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Position Buffer"),
            contents: bytemuck::cast_slice(positions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let velocity_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Velocity Buffer"),
            contents: bytemuck::cast_slice(velocities),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Globals Buffer"),
            contents: bytemuck::bytes_of(&globals),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &position_buffer,
            &velocity_buffer,
            &globals_buffer,
        );

        // extract shader before config is moved into self
        let shader = config.shader.take().unwrap_or_else(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Particle Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/circle.wgsl").into()),
            })
        });

        let render_pipeline = Self::create_pipeline(device, &shader, &bind_group_layout, &config);

        Self {
            render_pipeline,
            bind_group,
            globals_buffer,
            position_buffer,
            velocity_buffer,
            particle_count: positions.len() as u32,
            buffer_capacity: positions.len() as u32,
            bind_group_layout,
            config,
        }
    }

    pub fn upload_positions(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[glam::Vec2],
    ) {
        self.ensure_capacity(device, positions.len() as u32);
        queue.write_buffer(&self.position_buffer, 0, bytemuck::cast_slice(positions));
        self.particle_count = positions.len() as u32;
    }

    pub fn upload_velocities(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        velocities: &[glam::Vec2],
    ) {
        self.ensure_capacity(device, velocities.len() as u32);
        queue.write_buffer(&self.velocity_buffer, 0, bytemuck::cast_slice(velocities));
    }

    pub fn set_viewport_size(&mut self, queue: &wgpu::Queue, width: f32, height: f32) {
        self.config.viewport_size = [width, height];
        self.update_globals(queue);
    }

    pub fn render<'rp>(&'rp self, pass: &mut wgpu::RenderPass<'rp>) {
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..4, 0..self.particle_count);
    }

    pub fn reload_shader(&mut self, device: &wgpu::Device, source: &str) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader (reloaded)"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        self.render_pipeline =
            Self::create_pipeline(device, &shader, &self.bind_group_layout, &self.config);
    }

    // ── private ───────────────────────────────────────────────────────────────

    /// Grows position_buffer/velocity_buffer (and recreates the bind group)
    /// if `required` exceeds the currently allocated capacity. No-op
    /// otherwise, so repeated uploads at or under capacity never reallocate
    /// or touch the bind group — this keeps the common per-frame case (same
    /// particle count every frame) free of any allocation churn.
    fn ensure_capacity(&mut self, device: &wgpu::Device, required: u32) {
        if required <= self.buffer_capacity {
            return;
        }

        let byte_size = (required as u64) * (std::mem::size_of::<[f32; 2]>() as u64);

        self.position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Position Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.velocity_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Velocity Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.position_buffer,
            &self.velocity_buffer,
            &self.globals_buffer,
        );

        self.buffer_capacity = required;
    }

    fn update_globals(&self, queue: &wgpu::Queue) {
        let globals = Globals {
            window_size: self.config.viewport_size,
            particle_radius: self.config.particle_radius,
            _pad: 0.0,
        };
        queue.write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        position_buffer: &wgpu::Buffer,
        velocity_buffer: &wgpu::Buffer,
        globals_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: velocity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: globals_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn create_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        bind_group_layout: &wgpu::BindGroupLayout,
        config: &RendererConfig,
    ) -> wgpu::RenderPipeline {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Particle Pipeline Layout"),
            bind_group_layouts: &[Some(bind_group_layout)],
            immediate_size: 0,
        });

        let depth_stencil = config.depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil,
            multisample: wgpu::MultisampleState {
                count: config.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }
}

#[cfg(test)]
mod tests {}
