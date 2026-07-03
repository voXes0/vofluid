use crate::grid::UniformGrid;
use crate::particle::Positions;

pub struct NeighborCache {
    starts: Vec<u32>,
    neighbor: Vec<u32>,
    distance: Vec<f32>,
}

impl NeighborCache {
    pub fn build(positions: &Positions, grid: &UniformGrid, i_radius_sq: f32) -> Self {
        let n = positions.len();
        let mut starts = vec![0u32; n + 1];

        let mut accepted: Vec<(u32, u32, f32)> = Vec::new();

        for (p_indice, pos) in positions.iter().enumerate() {
            grid.query_neighbours(*pos, |n_indice| {
                if n_indice == p_indice {
                    return;
                }
                let n_pos = positions[n_indice];
                let offset = n_pos - pos;
                let distance_sq = offset.length_squared();
                if distance_sq > i_radius_sq {
                    return;
                }
                let distance = distance_sq.sqrt();
                accepted.push((p_indice as u32, n_indice as u32, distance));
            });
        }

        // Pass 2: prefix sum from counts derived from `accepted`.
        for &(owner, _, _) in &accepted {
            starts[owner as usize + 1] += 1;
        }
        for i in 0..n {
            starts[i + 1] += starts[i];
        }

        // Pass 3: scatter into flat arrays via cursors.
        let total = starts[n] as usize;
        let mut neighbor = vec![0u32; total];
        let mut distance = vec![0.0f32; total];
        let mut cursors = starts.clone();

        for &(owner, nbr, d) in &accepted {
            let cursor = &mut cursors[owner as usize];
            neighbor[*cursor as usize] = nbr;
            distance[*cursor as usize] = d;
            *cursor += 1;
        }

        Self {
            starts,
            neighbor,
            distance,
        }
    }

    #[inline]
    pub fn neighbours_of(&self, p: usize) -> impl Iterator<Item = (u32, f32)> + '_ {
        let s = self.starts[p] as usize;
        let e = self.starts[p + 1] as usize;
        self.neighbor[s..e]
            .iter()
            .copied()
            .zip(self.distance[s..e].iter().copied())
    }
}
