//! The error type returned by rmath's distribution functions.

use std::borrow::Cow;
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A distribution family, as named in error messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Distribution {
    Normal,
    Cauchy,
    StudentT,
    Beta,
    Binomial,
    Uniform,
    Exponential,
    ChiSquared,
}

impl fmt::Display for Distribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Distribution::Normal => "normal",
            Distribution::Cauchy => "Cauchy",
            Distribution::StudentT => "Student t",
            Distribution::Beta => "beta",
            Distribution::Binomial => "binomial",
            Distribution::Uniform => "uniform",
            Distribution::Exponential => "exponential",
            Distribution::ChiSquared => "chi-squared",
        })
    }
}

/// An error from one of rmath's distribution functions.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Error {
    /// A parameter is outside the distribution's parameter space, e.g. a
    /// non-positive standard deviation.
    #[error("invalid {distribution} parameter: {parameter} = {value} ({requirement})")]
    InvalidParameter {
        distribution: Distribution,
        parameter: Cow<'static, str>,
        value: f64,
        requirement: Cow<'static, str>,
    },
}

/// The result type returned by rmath's distribution functions.
pub type Result<T> = std::result::Result<T, Error>;

/// Returns an [`Error::InvalidParameter`] unless `valid`.
pub(crate) fn require(
    valid: bool,
    distribution: Distribution,
    parameter: &'static str,
    value: f64,
    requirement: &'static str,
) -> Result<()> {
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            distribution,
            parameter: Cow::Borrowed(parameter),
            value,
            requirement: Cow::Borrowed(requirement),
        })
    }
}

/// `value` is a finite real number.
pub(crate) fn require_finite(
    distribution: Distribution,
    parameter: &'static str,
    value: f64,
) -> Result<()> {
    require(
        value.is_finite(),
        distribution,
        parameter,
        value,
        "must be finite",
    )
}

/// `value` is positive (and not NaN).
pub(crate) fn require_positive(
    distribution: Distribution,
    parameter: &'static str,
    value: f64,
) -> Result<()> {
    require(
        value > 0.0,
        distribution,
        parameter,
        value,
        "must be positive",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dbeta, dbinom, dnorm, dt, dunif, lbeta};

    #[test]
    fn errors_name_the_distribution_and_parameter() {
        let error = dnorm(0.0, 0.0, -1.0, false).unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid normal parameter: sd = -1 (must be positive)"
        );
        let cases: [(Result<f64>, Distribution, &str); 6] = [
            (dbeta(0.5, 2.0, 0.0, false), Distribution::Beta, "shape2"),
            (dbinom(1.0, 2.5, 0.3, false), Distribution::Binomial, "size"),
            (dbinom(1.0, 3.0, 1.5, false), Distribution::Binomial, "prob"),
            (dunif(0.5, 1.0, 0.0, false), Distribution::Uniform, "max"),
            (dt(0.5, f64::NAN, false), Distribution::StudentT, "df"),
            (lbeta(-1.0, 2.0), Distribution::Beta, "a"),
        ];
        for (result, want_distribution, want_parameter) in cases {
            match result {
                Err(Error::InvalidParameter {
                    distribution,
                    parameter,
                    ..
                }) => {
                    assert_eq!(distribution, want_distribution);
                    assert_eq!(parameter, want_parameter);
                }
                other => panic!("expected an invalid {want_parameter}, got {other:?}"),
            }
        }
    }
}
