pub trait Validate<E> {
    fn validate(&self) -> Result<(), E>;
}

pub trait Function<T, S, E> {
    fn function(&self, x: T) -> Result<S, E>;
}

pub trait Range {
    fn range(&self) -> (Option<f64>, Option<f64>);
    fn default_range(&self) -> (f64, f64);

    fn range_or_default(&self) -> (f64, f64) {
        let (ll, ul) = self.range();
        let (default_ll, default_ul) = self.default_range();

        (ll.unwrap_or(default_ll), ul.unwrap_or(default_ul))
    }
    fn has_default_range(&self) -> bool {
        let (ll, ul) = self.range_or_default();
        let (default_ll, default_ul) = self.default_range();
        ll == default_ll && ul == default_ul
    }
}

pub trait Integrate<E, O>: Function<f64, f64, O> + Range {
    fn integral(&self) -> Result<f64, E>;
    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, E>;
}

/// Compute the normalization constant for a truncated distribution.
///
/// This helper function calculates the probability mass within truncation bounds
/// for any distribution with a CDF. It's used by NormalPrior, CauchyPrior, StudentTPrior,
/// and any other truncated distributions.
///
/// # Arguments
/// * `lower` - Lower truncation bound
/// * `upper` - Upper truncation bound
/// * `cdf` - Cumulative distribution function that takes (x, lower_tail) -> Result<f64, E>
///
/// # Returns
/// The normalization constant (probability mass within bounds)
pub fn truncated_normalization<E>(
    lower: f64,
    upper: f64,
    mut cdf: impl FnMut(f64, bool) -> Result<f64, E>,
) -> Result<f64, E> {
    match (lower.is_infinite(), upper.is_infinite()) {
        (true, true) => Ok(1.0),
        (false, true) => cdf(lower, false), // P(X > lower) = 1 - F(lower)
        (true, false) => cdf(upper, true),  // P(X < upper) = F(upper)
        (false, false) => {
            let p_lower = cdf(lower, true)?;
            let p_upper = cdf(upper, true)?;
            Ok(p_upper - p_lower)
        }
    }
}
