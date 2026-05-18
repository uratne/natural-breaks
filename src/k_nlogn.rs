use num_traits::ToPrimitive;

use crate::{error::Error, util::validate_and_convert, ClassifiedResult, IndexRanges};

/// O(k·n·log n) natural breaks classifier.
///
/// Same problem as [`KNSquared`](crate::k_n2::KNSquared), but uses the
/// divide-and-conquer DP optimization to replace the inner O(n) scan with
/// O(log n) amortized work per row, exploiting the monotonicity of the
/// optimal last-class break (the "no-crossing-paths" property).
///
/// Based on:
/// - Hilferink, "Fisher's Natural Breaks Classification complexity proof",
///   Object Vision BV. <https://geodms.nl/docs/fisher's-natural-breaks-classification-complexity-proof.html>
/// - Wang & Song, "Optimal Classification of Quantitative Data",
///   *The R Journal*, Vol. 3/2, December 2011.
///   <https://journal.r-project.org/articles/RJ-2011-015/>
///
/// # Memory variants
///
/// By default this stores the full *k* × *n* backtrack matrix for O(k·n)
/// memory and a single DP pass. Enable the `low-memory` feature to drop to
/// O(n) memory at the cost of re-running the algorithm *k* times for
/// backtracking — O(k²·n·log n) total time.
pub struct KNLogN {}

// ---------------------------------------------------------------------------
// Prefix sums: O(1) WCSS for any contiguous range
// ---------------------------------------------------------------------------

/// Precomputed prefix sums of x and x², enabling O(1) WCSS queries.
struct PrefixSums {
    /// prefix_sum[i] = sum of data[0..i]
    prefix_sum: Vec<f64>,
    /// prefix_sum_sq[i] = sum of data[0..i].map(|x| x * x)
    prefix_sum_sq: Vec<f64>,
}

impl PrefixSums {
    fn new(data: &[f64]) -> Self {
        let n = data.len();
        let mut prefix_sum = Vec::with_capacity(n + 1);
        let mut prefix_sum_sq = Vec::with_capacity(n + 1);
        prefix_sum.push(0.0);
        prefix_sum_sq.push(0.0);
        let mut s = 0.0;
        let mut s_sq = 0.0;
        for &x in data {
            s += x;
            s_sq += x * x;
            prefix_sum.push(s);
            prefix_sum_sq.push(s_sq);
        }
        Self { prefix_sum, prefix_sum_sq }
    }

    /// WCSS of `data[start..=end]` (inclusive on both ends).
    ///
    /// Returns `sum_sq - sum² / count`. This is mathematically equivalent
    /// to `Σ (xᵢ - μ)²` but computable in O(1) from prefix sums.
    ///
    /// TODO: For workloads with large values and small variance, consider
    /// a Welford-style accumulator to avoid catastrophic cancellation.
    #[inline]
    fn wcss(&self, start: usize, end: usize) -> f64 {
        debug_assert!(start <= end);
        let count = (end - start + 1) as f64;
        let sum = self.prefix_sum[end + 1] - self.prefix_sum[start];
        let sum_sq = self.prefix_sum_sq[end + 1] - self.prefix_sum_sq[start];
        // Clamp to 0 to guard against tiny negatives from floating-point error
        // when all values in the range are equal.
        (sum_sq - sum * sum / count).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// One row of the DP via divide-and-conquer
// ---------------------------------------------------------------------------

/// Fills `best_cost_curr[row_lo..=row_hi]` and `last_split_curr[row_lo..=row_hi]`
/// for the current number-of-classes `j`, knowing that:
///   - `best_cost_prev[i]` = optimal WCSS for `data[0..=i]` split into `j-1` clusters
///   - the optimal split point for every row in `[row_lo..=row_hi]` lies in
///     `[search_lo..=search_hi]` (the no-crossing-paths invariant)
///
/// This is the divide step. We find the optimum for `mid_row` by scanning its
/// allowed split range, then recurse on the two halves with shrunken ranges.
#[allow(clippy::too_many_arguments)]
fn fill_row_dc(
    prefix: &PrefixSums,
    best_cost_prev: &[f64],
    best_cost_curr: &mut [f64],
    last_split_curr: &mut [usize],
    row_lo: usize,
    row_hi: usize,
    search_lo: usize,
    search_hi: usize,
) {
    if row_lo > row_hi {
        return;
    }
    let mid_row = row_lo + (row_hi - row_lo) / 2;

    // For row `mid_row`, find the candidate_split in [search_lo..=search_hi]
    // that minimizes:
    //   best_cost_prev[candidate_split - 1] + wcss(candidate_split, mid_row)
    //
    // candidate_split is the start index of the *last* cluster.
    // Valid range: search_lo ≤ candidate_split ≤ min(search_hi, mid_row).
    let upper = search_hi.min(mid_row);

    let mut best_cost = f64::INFINITY;
    // Initialize to search_lo so a sane value is recorded even in degenerate cases.
    let mut best_split = search_lo;

    for candidate_split in search_lo..=upper {
        // Cost of the first j-1 clusters covering data[0..candidate_split].
        // When candidate_split == 0, the "previous" partition is empty (j=1 path),
        // which the caller arranges by setting best_cost_prev[anything] correctly
        // for the j=1 base case. For j ≥ 2 we always have candidate_split ≥ 1.
        let left_cost = if candidate_split == 0 {
            0.0
        } else {
            best_cost_prev[candidate_split - 1]
        };
        let right_cost = prefix.wcss(candidate_split, mid_row);
        let total = left_cost + right_cost;

        if total < best_cost {
            best_cost = total;
            best_split = candidate_split;
        }
    }

    best_cost_curr[mid_row] = best_cost;
    last_split_curr[mid_row] = best_split;

    // Monotonicity: rows above mid have argmin ≤ best_split,
    //               rows below mid have argmin ≥ best_split.
    // Both halves keep best_split in range so ties are handled correctly.
    if mid_row > row_lo {
        fill_row_dc(
            prefix,
            best_cost_prev,
            best_cost_curr,
            last_split_curr,
            row_lo,
            mid_row - 1,
            search_lo,
            best_split,
        );
    }
    fill_row_dc(
        prefix,
        best_cost_prev,
        best_cost_curr,
        last_split_curr,
        mid_row + 1,
        row_hi,
        best_split,
        search_hi,
    );
}

// ---------------------------------------------------------------------------
// Full-memory variant: stores the entire k × n backtrack matrix
// ---------------------------------------------------------------------------

#[cfg(not(feature = "low-memory"))]
fn compute_dp(data: &[f64], k: usize) -> Vec<Vec<usize>> {
    let n = data.len();
    let prefix = PrefixSums::new(data);

    // matrix_b[i][m] = start index of the last cluster when partitioning
    //                 data[0..=i] into m+1 clusters.
    let mut matrix_b: Vec<Vec<usize>> = vec![vec![0; k]; n];

    // best_cost_prev holds row j-1; best_cost_curr is filled for row j.
    let mut best_cost_prev: Vec<f64> = vec![f64::INFINITY; n];
    let mut best_cost_curr: Vec<f64> = vec![f64::INFINITY; n];

    // Base case m = 0 (one cluster): cluster spans data[0..=i] entirely.
    for i in 0..n {
        best_cost_prev[i] = prefix.wcss(0, i);
        matrix_b[i][0] = 0;
    }

    // Fill m = 1..k-1 (i.e. 2..k clusters)
    for m in 1..k {
        // Reset current row.
        for v in best_cost_curr.iter_mut() {
            *v = f64::INFINITY;
        }
        let mut last_split_curr: Vec<usize> = vec![0; n];

        // For m clusters, the smallest valid i is m (need at least m+1 values
        // for m+1 clusters → wait, we 0-index so m clusters here = m+1 in math).
        // We want at least m+1 elements to form m+1 clusters, so i ≥ m.
        fill_row_dc(
            &prefix,
            &best_cost_prev,
            &mut best_cost_curr,
            &mut last_split_curr,
            m,       // row_lo: data[0..=m] is the smallest range that fits m+1 clusters
            n - 1,   // row_hi
            m,       // search_lo: last cluster must start at index ≥ m so prior clusters fit
            n - 1,   // search_hi
        );

        for i in m..n {
            matrix_b[i][m] = last_split_curr[i];
        }

        // Swap row buffers for next iteration.
        std::mem::swap(&mut best_cost_prev, &mut best_cost_curr);
    }

    matrix_b
}

// ---------------------------------------------------------------------------
// Low-memory variant: only O(n) memory; backtracks by re-running on prefixes
// ---------------------------------------------------------------------------

#[cfg(feature = "low-memory")]
/// Runs the DP for `k` clusters over `data[0..=upto]` and returns the
/// last-cluster start indices for that exact `(upto, k-1)` cell.
///
/// Only computes what's needed to recover `last_split_curr[upto]` for the
/// final row — enough for one step of backtracking.
fn solve_last_split(prefix: &PrefixSums, n: usize, upto: usize, k: usize) -> usize {
    if k == 1 {
        return 0;
    }

    let effective_n = upto + 1; // working over data[0..=upto]

    let mut best_cost_prev: Vec<f64> = vec![f64::INFINITY; effective_n];
    let mut best_cost_curr: Vec<f64> = vec![f64::INFINITY; effective_n];

    // Base m = 0: one cluster.
    for (i, cost) in best_cost_prev.iter_mut().enumerate().take(effective_n) {
        *cost = prefix.wcss(0, i);
    }

    let mut last_split_curr: Vec<usize> = vec![0; effective_n];

    for m in 1..k {
        for v in best_cost_curr.iter_mut() {
            *v = f64::INFINITY;
        }
        for v in last_split_curr.iter_mut() {
            *v = 0;
        }

        fill_row_dc(
            prefix,
            &best_cost_prev,
            &mut best_cost_curr,
            &mut last_split_curr,
            m,
            effective_n - 1,
            m,
            effective_n - 1,
        );

        std::mem::swap(&mut best_cost_prev, &mut best_cost_curr);
    }

    let _ = n; // silence unused warning when n isn't needed
    last_split_curr[upto]
}

#[cfg(feature = "low-memory")]
/// Reconstructs cluster boundaries by re-running the DP on shrinking
/// prefixes. Returns the half-open index ranges directly — no k×n
/// backtrack matrix is allocated, keeping memory at O(n).
fn compute_ranges(data: &[f64], k: usize) -> IndexRanges {
    let n = data.len();
    let prefix = PrefixSums::new(data);

    // Backtrack from (n-1, k-1) down to (·, 0).
    // At each step, re-run the DP on data[0..=cluster_end-1] with (m+1) clusters
    // to find where the m-th cluster starts.
    let mut ranges: IndexRanges = Vec::with_capacity(k);
    let mut cluster_end = n;
    let mut m = k - 1;
    loop {
        let start = solve_last_split(&prefix, n, cluster_end - 1, m + 1);
        ranges.push((start, cluster_end));
        if m == 0 {
            break;
        }
        cluster_end = start;
        m -= 1;
    }
    ranges.reverse();
    ranges
}

// ---------------------------------------------------------------------------
// Backtracking helpers (only needed for full-memory variant)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "low-memory"))]
fn backtrack_values<T: Clone>(data: &[T], matrix_b: &[Vec<usize>], k: usize) -> ClassifiedResult<T> {
    let n = data.len();
    let mut result: ClassifiedResult<T> = Vec::with_capacity(k);
    let mut cluster_end = n;
    let mut m = k - 1;
    loop {
        let cluster_start = matrix_b[cluster_end - 1][m];
        result.push(data[cluster_start..cluster_end].to_vec());
        if m == 0 {
            break;
        }
        cluster_end = cluster_start;
        m -= 1;
    }
    result.reverse();
    result
}

#[cfg(not(feature = "low-memory"))]
fn backtrack_indices(matrix_b: &[Vec<usize>], k: usize, n: usize) -> IndexRanges {
    let mut result: IndexRanges = Vec::with_capacity(k);
    let mut cluster_end = n;
    let mut m = k - 1;
    loop {
        let cluster_start = matrix_b[cluster_end - 1][m];
        result.push((cluster_start, cluster_end));
        if m == 0 {
            break;
        }
        cluster_end = cluster_start;
        m -= 1;
    }
    result.reverse();
    result
}

// ---------------------------------------------------------------------------
// Public API (mirrors KNSquared)
// ---------------------------------------------------------------------------

impl KNLogN {
    /// Classifies pre-sorted data into `k` clusters.
    ///
    /// Time complexity: O(k·n·log n) by default, or O(k²·n·log n) with the
    /// `low-memory` feature enabled.
    ///
    /// # Warning
    /// **`data` MUST be sorted in ascending order.** Passing unsorted data
    /// will produce meaningless results without any error.
    #[cfg(not(feature = "low-memory"))]
    pub fn classify<T>(data: Vec<T>, k: usize) -> Result<ClassifiedResult<T>, Error>
    where
        T: PartialOrd + Clone + ToPrimitive,
    {
        let converted_data = validate_and_convert(&data, k)?;
        let matrix_b = compute_dp(&converted_data, k);
        Ok(backtrack_values(&data, &matrix_b, k))
    }

    /// Classifies pre-sorted data into `k` clusters.
    ///
    /// Time complexity: O(k²·n·log n) with the `low-memory` feature.
    /// Memory: O(n).
    ///
    /// # Warning
    /// **`data` MUST be sorted in ascending order.** Passing unsorted data
    /// will produce meaningless results without any error.
    #[cfg(feature = "low-memory")]
    pub fn classify<T>(data: Vec<T>, k: usize) -> Result<ClassifiedResult<T>, Error>
    where
        T: PartialOrd + Clone + ToPrimitive,
    {
        let converted_data = validate_and_convert(&data, k)?;
        let ranges = compute_ranges(&converted_data, k);
        let result = ranges
            .into_iter()
            .map(|(start, end)| data[start..end].to_vec())
            .collect();
        Ok(result)
    }

    /// Classifies pre-sorted data into `k` clusters, returning [`IndexRanges`].
    #[cfg(not(feature = "low-memory"))]
    pub fn classify_indices<T>(data: &[T], k: usize) -> Result<IndexRanges, Error>
    where
        T: PartialOrd + Clone + ToPrimitive,
    {
        let converted_data = validate_and_convert(data, k)?;
        let matrix_b = compute_dp(&converted_data, k);
        Ok(backtrack_indices(&matrix_b, k, data.len()))
    }

    /// Classifies pre-sorted data into `k` clusters, returning [`IndexRanges`].
    #[cfg(feature = "low-memory")]
    pub fn classify_indices<T>(data: &[T], k: usize) -> Result<IndexRanges, Error>
    where
        T: PartialOrd + Clone + ToPrimitive,
    {
        let converted_data = validate_and_convert(data, k)?;
        Ok(compute_ranges(&converted_data, k))
    }

    /// Sorts the data, then classifies into `k` clusters, returning [`IndexRanges`].
    pub fn classify_indices_with_sort<T>(mut data: Vec<T>, k: usize) -> Result<IndexRanges, Error>
    where
        T: PartialOrd + Clone + ToPrimitive,
    {
        for window in data.windows(2) {
            if window[0].partial_cmp(&window[1]).is_none() {
                return Err(Error::NaNError);
            }
        }
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self::classify_indices(&data, k)
    }

    /// Sorts the data, then classifies into `k` clusters.
    pub fn classify_with_sort<T>(mut data: Vec<T>, k: usize) -> Result<ClassifiedResult<T>, Error>
    where
        T: PartialOrd + Clone + ToPrimitive,
    {
        for window in data.windows(2) {
            if window[0].partial_cmp(&window[1]).is_none() {
                return Err(Error::NaNError);
            }
        }
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self::classify(data, k)
    }
}

// ---------------------------------------------------------------------------
// Tests — mirror KNSquared's test suite so we can verify equivalence
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_clustering() {
        let data = vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
        let result = KNLogN::classify(data, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(result[1], vec![10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_three_clusters() {
        let data = vec![1, 2, 3, 10, 11, 12, 50, 51, 52];
        let result = KNLogN::classify(data, 3).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec![1, 2, 3]);
        assert_eq!(result[1], vec![10, 11, 12]);
        assert_eq!(result[2], vec![50, 51, 52]);
    }

    #[test]
    fn test_k_equals_n() {
        let data = vec![5.0, 10.0, 15.0];
        let result = KNLogN::classify(data, 3).unwrap();
        assert_eq!(result, vec![vec![5.0], vec![10.0], vec![15.0]]);
    }

    #[test]
    fn test_single_cluster() {
        let data = vec![1.0, 2.0, 3.0];
        let result = KNLogN::classify(data, 1).unwrap();
        assert_eq!(result, vec![vec![1.0, 2.0, 3.0]]);
    }

    #[test]
    fn test_duplicates() {
        let data = vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        let result = KNLogN::classify(data, 2).unwrap();
        assert_eq!(result[0], vec![1.0, 1.0, 1.0]);
        assert_eq!(result[1], vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_four_clusters() {
        let data = vec![1, 2, 10, 11, 20, 21, 30, 31];
        let result = KNLogN::classify(data, 4).unwrap();
        assert_eq!(result, vec![vec![1, 2], vec![10, 11], vec![20, 21], vec![30, 31]]);
    }

    #[test]
    fn test_wcss_matches_optimal() {
        fn wcss(cluster: &[f64]) -> f64 {
            let mean = cluster.iter().sum::<f64>() / cluster.len() as f64;
            cluster.iter().map(|x| (x - mean).powi(2)).sum()
        }

        let cases: Vec<(Vec<f64>, usize, f64)> = vec![
            (vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0], 2, 4.0),
            (vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 50.0, 51.0, 52.0], 3, 6.0),
            (vec![1.0, 3.0, 5.0, 7.0, 9.0, 50.0, 52.0, 54.0], 2, 48.0),
        ];

        for (data, k, expected_wcss) in cases {
            let result = KNLogN::classify(data, k).unwrap();
            let total_wcss: f64 = result.iter().map(|c| wcss(c)).sum();
            assert!(
                (total_wcss - expected_wcss).abs() < 1e-9,
                "k={k}: expected WCSS={expected_wcss}, got {total_wcss}"
            );
        }
    }

    /// Cross-check against KNSquared: both must achieve the same total WCSS.
    /// Exact index ranges may differ when multiple optimal splits exist
    /// (tie-breaking), so we compare cost rather than indices.
    #[test]
    fn test_agrees_with_kn_squared() {
        use crate::k_n2::KNSquared;

        fn total_wcss(clusters: &[Vec<f64>]) -> f64 {
            clusters.iter().map(|c| {
                let mean = c.iter().sum::<f64>() / c.len() as f64;
                c.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            }).sum()
        }

        let cases: Vec<(Vec<f64>, usize)> = vec![
            (vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0], 2),
            (vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 50.0, 51.0, 52.0], 3),
            (vec![1.0, 3.0, 5.0, 7.0, 9.0, 50.0, 52.0, 54.0], 2),
            (vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0], 3),
            (vec![0.1, 0.2, 0.3, 0.4, 0.5, 10.0, 10.1, 10.2, 100.0, 100.1], 4),
        ];

        for (data, k) in cases {
            let log_result = KNLogN::classify(data.clone(), k).unwrap();
            let sq_result = KNSquared::classify(data.clone(), k).unwrap();
            let log_wcss = total_wcss(&log_result);
            let sq_wcss = total_wcss(&sq_result);
            assert!(
                (log_wcss - sq_wcss).abs() < 1e-9,
                "WCSS mismatch on data={data:?}, k={k}: KNLogN={log_wcss}, KNSquared={sq_wcss}"
            );
        }
    }
}