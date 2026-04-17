use num_traits::ToPrimitive;

use std::any::type_name;

use crate::error::Error;

type ClassifiedResult<T> = Vec<Vec<T>>;

/// A list of half-open index ranges `[start, end)` representing cluster boundaries.
pub type IndexRanges = Vec<(usize, usize)>;

/// O(kn²) natural breaks classifier.
///
/// Based on the algorithm described in:
/// Wang & Song, "Optimal Classification of Quantitative Data",
/// *The R Journal*, Vol. 3/2, December 2011.
/// <https://journal.r-project.org/articles/RJ-2011-015/>
///
/// This implementation adds an early-exit pruning step: the inner loop breaks
/// when the running within-cluster sum of squares already exceeds the current
/// best, exploiting the monotonicity of WCSS on sorted data.
pub struct KNSquared {}

/// Validates inputs and converts the data slice to `f64`.
///
/// Returns an error if `k` is zero, if `n < k`, or if any element
/// cannot be losslessly cast to `f64`.
fn validate_and_convert<T: ToPrimitive>(data: &[T], k: usize) -> Result<Vec<f64>, Error> {
    let n = data.len();
    let t_type = type_name::<T>();

    if k == 0 {
        return Err(Error::ZeroClusters);
    }

    if n < k {
        return Err(Error::NLessThanKError);
    }

    data.iter()
        .map(|t| t.to_f64().ok_or(Error::CastError(t_type.to_string())))
        .collect()
}

/// Fills the dynamic-programming tables and returns the backtrack matrix.
///
/// - `matrix_d[i][m]` = minimum WCSS for partitioning `data[0..=i]` into `m + 1` clusters.
/// - `matrix_b[i][m]` = the start index of the last cluster in that optimal partition.
///
/// The first column (`m = 0`, i.e. one cluster) is filled with the running
/// WCSS over the whole prefix via the incremental `d_next` / `mu_next`
/// recurrence.  Subsequent columns iterate backwards over candidate split
/// points and apply an early-exit prune: if the partial WCSS of the
/// right-hand cluster already exceeds the best solution found so far we
/// can stop, because extending the cluster further can only increase it.
fn compute_dp(converted_data: &[f64], k: usize) -> Vec<Vec<usize>> {
    let n = converted_data.len();
    // matrix_d[i][m]: min WCSS for data[0..=i] split into m+1 clusters
    let mut matrix_d: Vec<Vec<f64>> = vec![vec![f64::INFINITY; k]; n];
    // matrix_b[i][m]: start index of the last cluster in the optimal split
    let mut matrix_b: Vec<Vec<usize>> = vec![vec![0; k]; n];

    // Base case: one cluster (m = 0) covering data[0..=i]
    let mut mu = converted_data[0];
    let mut d_running = 0f64;
    matrix_d[0][0] = 0.0;
    matrix_b[0][0] = 0;
    for i in 1..n {
        let xi = converted_data[i];
        let count = i + 1;
        d_running = d_next(d_running, count, xi, mu);
        mu = mu_next(xi, count, mu);
        matrix_d[i][0] = d_running;
        matrix_b[i][0] = 0;
    }

    // Fill columns m = 1..k-1
    for m in 1..k {
        for i in m..n {
            // Try putting data[j..=i] into the m-th cluster.
            // Walk j backwards from i so we can incrementally compute the
            // WCSS of the right-hand cluster.
            let mut d_xi_2_xj = 0f64;
            let mut mu_prev = converted_data[i];
            let mut lowest_d = f64::INFINITY;
            let mut b = i;

            // Singleton cluster: data[i..=i], cost comes entirely from the left
            let cost = matrix_d[i - 1][m - 1];
            if cost < lowest_d {
                lowest_d = cost;
                b = i;
            }

            for j in (m..i).rev() {
                let count = i - j + 1;
                d_xi_2_xj = d_next(d_xi_2_xj, count, converted_data[j], mu_prev);
                mu_prev = mu_next(converted_data[j], count, mu_prev);

                // Early-exit prune: right-hand cluster alone already costs
                // more than the best total seen so far.
                if d_xi_2_xj >= lowest_d {
                    break;
                }

                if matrix_d[j - 1][m - 1] + d_xi_2_xj < lowest_d {
                    lowest_d = matrix_d[j - 1][m - 1] + d_xi_2_xj;
                    b = j;
                }
            }

            matrix_d[i][m] = lowest_d;
            matrix_b[i][m] = b;
        }
    }

    matrix_b
}

/// Walks the backtrack matrix from the last cluster back to the first,
/// collecting the data slices for each cluster.
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

/// Same as [`backtrack_values`] but returns half-open index ranges instead of
/// cloned data slices.
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

impl KNSquared {
    /// Classifies pre-sorted data into `k` clusters using the natural breaks algorithm.
    ///
    /// Time complexity: O(kn²) where `k` is the number of clusters and `n` is the data length.
    ///
    /// # Warning
    /// **`data` MUST be sorted in ascending order.** Passing unsorted data will produce
    /// meaningless results without any error. Use [`classify_with_sort`](Self::classify_with_sort)
    /// if your data is not pre-sorted.
    pub fn classify<T>(data: Vec<T>, k: usize) -> Result<ClassifiedResult<T>, Error>
    where
        T: PartialOrd + Clone + ToPrimitive,
    {
        let converted_data = validate_and_convert(&data, k)?;
        let matrix_b = compute_dp(&converted_data, k);
        Ok(backtrack_values(&data, &matrix_b, k))
    }

    /// Classifies pre-sorted data into `k` clusters, returning [`IndexRanges`].
    ///
    /// Each returned tuple `(start, end)` represents a half-open range `[start, end)`
    /// into the input slice.
    ///
    /// Time complexity: O(kn²) where `k` is the number of clusters and `n` is the data length.
    ///
    /// # Warning
    /// **`data` MUST be sorted in ascending order.** Passing unsorted data will produce
    /// meaningless results without any error.
    pub fn classify_indices<T>(data: &[T], k: usize) -> Result<IndexRanges, Error>
    where
        T: PartialOrd + Clone + ToPrimitive,
    {
        let converted_data = validate_and_convert(data, k)?;
        let matrix_b = compute_dp(&converted_data, k);
        Ok(backtrack_indices(&matrix_b, k, data.len()))
    }

    /// Sorts the data, then classifies into `k` clusters, returning [`IndexRanges`].
    ///
    /// Each returned tuple `(start, end)` represents a half-open range `[start, end)`
    /// into the **sorted** data. Returns an error if the data contains NaN values.
    ///
    /// Time complexity: O(kn² + n log n) where `k` is the number of clusters and `n` is
    /// the data length.
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

    /// Sorts the data, then classifies into `k` clusters using the natural breaks algorithm.
    ///
    /// Returns an error if the data contains NaN values.
    ///
    /// Time complexity: O(kn² + n log n) where `k` is the number of clusters and `n` is
    /// the data length.
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

/// Incrementally updates the running mean after adding `xi` to a cluster of
/// `count` elements (including `xi` itself).
fn mu_next(xi: f64, count: usize, mu_prev: f64) -> f64 {
    (xi + (count - 1) as f64 * mu_prev) / count as f64
}

/// Incrementally updates the running WCSS (within-cluster sum of squares)
/// after adding `xi` to a cluster of `count` elements whose previous mean
/// was `mu_prev`.
fn d_next(d_prev: f64, count: usize, xi: f64, mu_prev: f64) -> f64 {
    d_prev + ((count - 1) as f64 / count as f64) * (xi - mu_prev).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_clustering() {
        let data = vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
        let result = KNSquared::classify(data, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(result[1], vec![10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_three_clusters() {
        let data = vec![1, 2, 3, 10, 11, 12, 50, 51, 52];
        let result = KNSquared::classify(data, 3).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec![1, 2, 3]);
        assert_eq!(result[1], vec![10, 11, 12]);
        assert_eq!(result[2], vec![50, 51, 52]);
    }

    #[test]
    fn test_k_equals_n() {
        let data = vec![5.0, 10.0, 15.0];
        let result = KNSquared::classify(data, 3).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec![5.0]);
        assert_eq!(result[1], vec![10.0]);
        assert_eq!(result[2], vec![15.0]);
    }

    #[test]
    fn test_single_cluster() {
        let data = vec![1.0, 2.0, 3.0];
        let result = KNSquared::classify(data, 1).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_zero_clusters_error() {
        let data = vec![1.0, 2.0];
        let result = KNSquared::classify(data, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_n_less_than_k_error() {
        let data = vec![1.0, 2.0];
        let result = KNSquared::classify(data, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_classify_with_sort() {
        let data = vec![12.0, 1.0, 11.0, 2.0, 10.0, 3.0];
        let result = KNSquared::classify_with_sort(data, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(result[1], vec![10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_duplicates() {
        let data = vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        let result = KNSquared::classify(data, 2).unwrap();
        assert_eq!(result[0], vec![1.0, 1.0, 1.0]);
        assert_eq!(result[1], vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_four_clusters() {
        let data = vec![1, 2, 10, 11, 20, 21, 30, 31];
        let result = KNSquared::classify(data, 4).unwrap();
        assert_eq!(result, vec![vec![1, 2], vec![10, 11], vec![20, 21], vec![30, 31]]);
    }

    #[test]
    fn test_classify_indices() {
        let data = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
        let result = KNSquared::classify_indices(&data, 2).unwrap();
        assert_eq!(result, vec![(0, 3), (3, 6)]);

        let data = [1, 2, 10, 11, 20, 21, 30, 31];
        let result = KNSquared::classify_indices(&data, 4).unwrap();
        assert_eq!(result, vec![(0, 2), (2, 4), (4, 6), (6, 8)]);
    }

    /// Cross-validate WCSS against brute-force verified values
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
            let result = KNSquared::classify(data, k).unwrap();
            let total_wcss: f64 = result.iter().map(|c| wcss(c)).sum();
            assert!(
                (total_wcss - expected_wcss).abs() < 1e-9,
                "k={k}: expected WCSS={expected_wcss}, got {total_wcss}"
            );
        }
    }
}