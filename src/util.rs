use std::any::type_name;

use num_traits::ToPrimitive;

use crate::error::Error;

/// Validates inputs and converts the data slice to `f64`.
///
/// Returns an error if `k` is zero, if `n < k`, or if any element
/// cannot be losslessly cast to `f64`.
pub(crate) fn validate_and_convert<T: ToPrimitive>(data: &[T], k: usize) -> Result<Vec<f64>, Error> {
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