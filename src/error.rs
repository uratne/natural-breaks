#[derive(Debug)]
pub enum Error {
    NaNError,
    CastError(String),
    ZeroClusters,
    NLessThanKError,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NaNError => write!(f, "NaN values are not supported for sorting"),
            Error::CastError(t) => write!(f, "failed to cast {t} to f64"),
            Error::ZeroClusters => write!(f, "cluster count needs to be greater than 0"),
            Error::NLessThanKError => write!(f, "cluster count greater than array size"),
        }
    }
}

impl std::error::Error for Error {}