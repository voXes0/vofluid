use glam::Vec2;

/// Simulation configuration parameters.
/// `PartialEq` lets `Simulation` notice a changed config
#[cfg_attr(feature = "serialzation", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
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
            substeps: 3,
            dt: 0.02,
            particle_count: 2000,
            rest_density: 10.,
            stiffness: 500.,
            stiffness_near: 1250.,
            interaction_radius: 50.,
            yield_ratio_compress: 0.1,
            yield_ratio_stretch: 0.1,
            plasticity: 18.,
            spring_stiffness: 0.,
            gravity: Vec2 { x: 0.0, y: 981.0 },
            bounds: Vec2 { x: 300., y: 300. },
        }
    }
}
