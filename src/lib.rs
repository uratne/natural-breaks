#![doc = include_str!("../README.md")]

pub mod error;
pub mod k_n2;
pub mod k_nlogn;
pub mod kn;

use error::Error;
/// Re-exported so downstream users can use the `ToPrimitive` bound in their own
/// generic code without adding `num-traits` to their `Cargo.toml`.
pub use num_traits::ToPrimitive;

/// Clustered values returned by [`classify`] and [`classify_with_sort`].
pub type ClassifiedResult<T> = Vec<Vec<T>>;

/// Half-open index ranges `[start, end)` returned by [`classify_indices`] and
/// [`classify_indices_with_sort`].
pub type IndexRanges = Vec<(usize, usize)>;

// Default top-level convenience functions using the O(kn²) implementation.
// For specific algorithm choice, use the module directly:
//   k_n2::KNSquared, kn::KN, k_nlogn::KNLogN

/// Classifies pre-sorted data into `k` clusters using the natural breaks algorithm.
///
/// Uses the O(kn²) algorithm from Wang & Song (2011) with an added pruning optimisation.
/// See [`k_n2::KNSquared`] for details.
///
/// # Warning
/// **`data` MUST be sorted in ascending order.** Passing unsorted data will produce
/// meaningless results without any error. Use [`classify_with_sort`] if your data is
/// not pre-sorted.
///
/// Returns an error if the data contains NaN values.
pub fn classify<T>(data: Vec<T>, k: usize) -> Result<ClassifiedResult<T>, Error>
where
    T: PartialOrd + Clone + ToPrimitive,
{
    k_n2::KNSquared::classify(data, k)
}

/// Classifies pre-sorted data into `k` clusters, returning [`IndexRanges`].
///
/// Each returned tuple `(start, end)` represents a half-open range `[start, end)`
/// into the input slice.
///
/// Uses the O(kn²) algorithm from Wang & Song (2011) with an added pruning optimisation.
/// See [`k_n2::KNSquared`] for details.
///
/// # Warning
/// **`data` MUST be sorted in ascending order.** Passing unsorted data will produce
/// meaningless results without any error. Use [`classify_indices_with_sort`] if your
/// data is not pre-sorted.
pub fn classify_indices<T>(data: &[T], k: usize) -> Result<IndexRanges, Error>
where
    T: PartialOrd + Clone + ToPrimitive,
{
    k_n2::KNSquared::classify_indices(data, k)
}

/// Sorts the data and classifies it into `k` clusters using the natural breaks algorithm.
///
/// Uses the O(kn² + n log n) algorithm from Wang & Song (2011) with an added pruning
/// optimisation. See [`k_n2::KNSquared`] for details.
///
/// Returns an error if the data contains NaN values.
pub fn classify_with_sort<T>(data: Vec<T>, k: usize) -> Result<ClassifiedResult<T>, Error>
where
    T: PartialOrd + Clone + ToPrimitive,
{
    k_n2::KNSquared::classify_with_sort(data, k)
}

/// Sorts the data and classifies into `k` clusters, returning [`IndexRanges`].
///
/// Each returned tuple `(start, end)` represents a half-open range `[start, end)`
/// into the **sorted** data.
///
/// Uses the O(kn² + n log n) algorithm from Wang & Song (2011) with an added pruning
/// optimisation. See [`k_n2::KNSquared`] for details.
///
/// Returns an error if the data contains NaN values.
pub fn classify_indices_with_sort<T>(data: Vec<T>, k: usize) -> Result<IndexRanges, Error>
where
    T: PartialOrd + Clone + ToPrimitive,
{
    k_n2::KNSquared::classify_indices_with_sort(data, k)
}
