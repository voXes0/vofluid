//! Safe 8-lane SIMD helpers shared by the kernels.
//!
//! Built on `wide::f32x8`: AVX on x86 (per `target-cpu`), simd128 on wasm,
//! NEON on arm, scalar code elsewhere. No `unsafe` is needed.

use crate::grid::SORTED_PAD;
use wide::f32x8;

pub(super) const LANES: usize = 8;

const _: () = assert!(
    SORTED_PAD >= LANES,
    "a full-width load at any slot must stay in bounds"
);

/// Loads `LANES` consecutive values starting at slot `at`.
#[inline(always)]
pub(super) fn load(a: &[f32], at: usize) -> f32x8 {
    // The slice is exactly `LANES` long, so the conversion compiles away.
    f32x8::from(<[f32; LANES]>::try_from(&a[at..at + LANES]).unwrap())
}

/// All-ones bits in the first `valid` lanes, zero elsewhere. AND it with a
/// vector to zero the lanes past the end of a stencil row.
#[inline(always)]
pub(super) fn lane_mask(valid: usize) -> f32x8 {
    const IOTA: f32x8 = f32x8::new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    IOTA.simd_lt(f32x8::splat(valid as f32))
}
