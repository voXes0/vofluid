struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local: vec2<f32>,
};

struct Particle {
    pos: vec2<f32>,
}

struct Globals {
    window_size: vec2<f32>,
    particle_radius: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> velocities: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> globals: Globals;

@vertex
fn vs_main(
    @builtin(vertex_index)   vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let x = f32(vertex_index & 1u);
    let y = f32((vertex_index >> 1u) & 1u);
    let corner = vec2<f32>(x, y) * 2.0 - 1.0;

    let center = particles[instance_index].pos;
    let scale = vec2<f32>(
        globals.particle_radius * 2.0 / globals.window_size.x,
        globals.particle_radius * 2.0 / globals.window_size.y,
    );
    let center_ndc = vec2<f32>(
        center.x / (globals.window_size.x * 0.5),
        -center.y / (globals.window_size.y * 0.5),
    );

    var out: VertexOutput;
    out.clip_position = vec4<f32>(center_ndc + corner * scale, 0.0, 1.0);
    out.local = corner;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = dot(in.local, in.local);
    let edge_width = fwidth(dist) * 2.0;
    let alpha = 1.0 - smoothstep(1.0 - edge_width, 1.0, dist);
    return vec4<f32>(0.1, 0.65, 0.4, alpha);
}
