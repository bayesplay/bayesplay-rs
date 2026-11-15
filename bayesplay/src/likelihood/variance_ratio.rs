use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

use super::{LikelihoodError, Observation};
use crate::common::{Function, Validate};

/// Likelihood for a population variance ratio.
///
/// `u` is the observed variance ratio and `df` is its degrees of freedom. The
/// parameter evaluated by [`Function::function`] is the population variance
/// ratio `r = sigma^2 / sigma0^2`.
#[derive(Default, Clone, Copy, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub struct VarianceRatioLikelihood {
    pub u: f64,
    pub df: f64,
}

impl VarianceRatioLikelihood {
    pub fn new(u: f64, df: f64) -> Self {
        Self { u, df }
    }
}

impl Function<f64, f64, LikelihoodError> for VarianceRatioLikelihood {
    fn function(&self, r: f64) -> Result<f64, LikelihoodError> {
        self.validate()?;

        if !r.is_finite() || r <= 0.0 {
            return Ok(0.0);
        }

        let log_r = r.ln();
        let log_scaled_observation = self.df.ln() + self.u.ln() - log_r;
        let half_df = self.df / 2.0;
        let exponential_term = if log_scaled_observation > f64::MAX.ln() {
            f64::INFINITY
        } else {
            log_scaled_observation.exp() / 2.0
        };
        let log_density = self.df.ln() - log_r + (half_df - 1.0) * log_scaled_observation
            - exponential_term
            - half_df * std::f64::consts::LN_2
            - ln_gamma(half_df);
        Ok(log_density.exp())
    }
}

impl Validate<LikelihoodError> for VarianceRatioLikelihood {
    fn validate(&self) -> Result<(), LikelihoodError> {
        let mut errors = Vec::new();

        if !self.u.is_finite() || self.u <= 0.0 {
            errors.push(LikelihoodError::InvalidU(self.u));
        }
        if !self.df.is_finite() || self.df <= 0.0 {
            errors.push(LikelihoodError::InvalidDF(self.df));
        }

        LikelihoodError::from_errors(errors)
    }
}

impl Observation for VarianceRatioLikelihood {
    fn update_observation(&mut self, observation: f64) {
        self.u = observation;
    }

    fn get_observation(&self) -> Option<f64> {
        Some(self.u)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::likelihood::Likelihood;

    #[test]
    fn matches_r_reference_values() {
        let likelihood = VarianceRatioLikelihood::new(1.6, 9.0);
        let r = [0.25, 0.64, 1.0, 1.69, 4.0, 9.61];
        let expected = [
            6.163_407_474_191_212e-8,
            3.754_993_539_943_915e-2,
            2.892_755_505_575_561e-1,
            5.158_140_262_124_65e-1,
            1.250_927_010_633_626e-1,
            6.927_791_874_628_43e-3,
        ];

        for (r, expected) in r.into_iter().zip(expected) {
            assert_relative_eq!(
                likelihood.function(r).unwrap(),
                expected,
                max_relative = 1e-13
            );
        }
    }

    #[test]
    fn is_zero_outside_positive_finite_support() {
        let likelihood = VarianceRatioLikelihood::new(1.6, 9.0);
        for x in [-1.0, 0.0, f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
            assert_eq!(likelihood.function(x).unwrap(), 0.0);
        }
    }

    #[test]
    fn combines_tail_terms_before_exponentiating() {
        let likelihood = VarianceRatioLikelihood::new(1.0, 1.0);
        assert_relative_eq!(
            likelihood.function(1e200).unwrap(),
            3.989_422_804_014_327e-101,
            max_relative = 1e-13
        );
    }

    #[test]
    fn peaks_at_observed_variance_ratio() {
        let likelihood = VarianceRatioLikelihood::new(1.6, 9.0);
        let mode = 1.6;
        let at_mode = likelihood.function(mode).unwrap();
        assert!(at_mode > likelihood.function(mode * 0.99).unwrap());
        assert!(at_mode > likelihood.function(mode * 1.01).unwrap());
        assert_relative_eq!(likelihood.get_observation().unwrap(), mode);
    }

    #[test]
    fn validates_parameters_and_round_trips_observation() {
        assert!(matches!(
            VarianceRatioLikelihood::new(0.0, 9.0).validate(),
            Err(LikelihoodError::InvalidU(0.0))
        ));
        assert!(matches!(
            VarianceRatioLikelihood::new(1.0, 0.0).validate(),
            Err(LikelihoodError::InvalidDF(0.0))
        ));
        assert!(matches!(
            VarianceRatioLikelihood::new(-1.0, -1.0).validate(),
            Err(LikelihoodError::MultipleErrors(_))
        ));

        let mut likelihood = VarianceRatioLikelihood::new(1.0, 9.0);
        likelihood.update_observation(1.4);
        assert_relative_eq!(likelihood.u, 1.4);
        assert_relative_eq!(likelihood.get_observation().unwrap(), 1.4);

        let dispatched: Likelihood = likelihood.into();
        assert_relative_eq!(
            dispatched.function(1.2).unwrap(),
            likelihood.function(1.2).unwrap()
        );
    }
}
