use glam::Vec2;

/// Simulation configuration parameters.
/// Serialisation behind the `serde` feature so the core stays lean
/// when used without config-file support.
#[cfg_attr(feature = "serialzation", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct SimConfig {
    pub debug: bool,
    pub substeps: u32,
    pub dt: f32,
    pub particle_count: usize,
    pub rest_density: f32,
    pub stiffness: f32,
    pub stiffness_near: f32,
    pub interaction_radius: f32,
    pub gravity: Vec2,
    pub yield_ratio_stretch: f32,
    pub yield_ratio_compress: f32,
    pub plasticity: f32,
    pub spring_stiffness: f32,
    pub bounds: Vec2,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            debug: false,
            substeps: 1,
            dt: 1.0 / 60.0,
            particle_count: 1,
            rest_density: 10.,
            stiffness: 9.,
            stiffness_near: 15.,
            interaction_radius: 100.,
            yield_ratio_compress: 0.1,
            yield_ratio_stretch: 0.1,
            plasticity: 18.,
            spring_stiffness: 900.,
            gravity: Vec2 { x: 0., y: 9.81 },
            bounds: Vec2 { x: 500., y: 500. },
        }
    }
}
