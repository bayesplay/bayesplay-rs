use itertools::izip;
use rmath::integrate;

use std::ops::Mul;
use thiserror::Error;

use serde::{Deserialize, Serialize};

use crate::common::Function;
use crate::common::Integrate;
use crate::common::Range;
use crate::common::Validate;

use crate::likelihood::*;
use crate::prior::*;

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
/// Model definition  
///
/// A model is constructed from a [Prior] and a [Likelihood]
///
///
/// Ordinarily, a [Model] will be constructed by multiplying together a
/// [Prior] and a [Likelihood]
///
/// ```
/// use bayesplay::prelude::*;
///
/// # fn main() {
/// //let likelihood = likelihood!(family = "normal", mean = 0.2, se = 4.0);
/// //let prior = prior!(family = "normal", mean = 0.0, sd = 1.0);
///
/// //let model: Model = likelihood * prior;
/// # }
/// ```
pub struct Model {
    pub prior: Prior,
    pub likelihood: Likelihood,
    pub range: (f64, f64),
}

#[derive(Error, Debug)]
pub enum IntegralError {
    #[error("Error with prior: {0}")]
    Prior(PriorError),
    #[error("Error with likelihood: {0}")]
    Likelihood(LikelihoodError),
    #[error("Error with integration: {0}")]
    Intergaration(rmath::integration::IntegrationError),
}

impl Model {
    pub fn get_observation(&self) -> Option<f64> {
        self.likelihood.get_observation()
    }

    pub fn posterior(&self) -> Result<Posterior, IntegralError> {
        let constant = self.integral()?;

        let model = *self;

        Ok(Posterior { model, constant })
    }
    pub fn predictive(&self) -> Predictive {
        Predictive(*self)
    }
}

impl Range for Model {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        let prior = self.prior;
        prior.range()
    }
    fn default_range(&self) -> (f64, f64) {
        let prior = self.prior;
        prior.default_range()
    }
}

impl Function<f64, f64, anyhow::Error> for Model {
    fn function(&self, x: f64) -> Result<f64, anyhow::Error> {
        let prior = self.prior;
        let likelihood = self.likelihood;
        let res = prior.function(x)? * likelihood.function(x)?;
        Ok(res)
    }
}

impl Integrate<IntegralError, anyhow::Error> for Model {
    /// Integrate a model object
    /// ```
    /// use bayesplay::prelude::*;
    ///
    /// # fn main() {
    /// let likelihood = NormalLikelihood::new(0.2, 4.0);
    /// let prior = NormalPrior::new(0.0, 1.0, (Some(-f64::INFINITY), Some(f64::INFINITY)));
    ///
    /// let model = Model {prior, likelihood, range: (-f64::INFINITY, f64::INFINITY)};
    /// let auc = model.integral().unwrap();
    ///
    /// assert_eq!(auc as f32, 0.09664394965841581 as f32);
    /// # }
    /// ```
    fn integral(&self) -> Result<f64, IntegralError> {
        let prior = self.prior;
        let likelihood = self.likelihood;

        let (lower, upper) = self.range_or_default();
        prior.validate().map_err(IntegralError::Prior)?;
        likelihood.validate().map_err(IntegralError::Likelihood)?;

        match prior {
            Prior::Point(point) => likelihood
                .function(point.point)
                .map_err(IntegralError::Likelihood),
            _ => {
                let model = *self;
                let f = move |x| model.function(x).unwrap();
                let h = integrate!(f = f, lower = lower, upper = upper);

                match h {
                    Ok(v) => Ok(v.value),
                    Err(e) => Err(IntegralError::Intergaration(e)),
                }
            }
        }
    }
    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, IntegralError> {
        let (_, _) = (lb, ub);
        self.integral()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Posterior {
    model: Model,
    constant: f64,
}

#[derive(Debug, Copy, Clone)]
pub struct Predictive(Model);

impl Function<f64, f64, anyhow::Error> for Posterior {
    fn function(&self, x: f64) -> Result<f64, anyhow::Error> {
        let prior = self.model.prior;
        let likelihood = self.model.likelihood;
        let k = self.constant;
        let res = (likelihood.function(x)? * prior.function(x)?) / k;
        Ok(res)
    }
}

impl Function<&[f64], Vec<Option<f64>>, anyhow::Error> for Posterior {
    fn function(&self, x: &[f64]) -> Result<Vec<Option<f64>>, anyhow::Error> {
        let prior = self.model.prior;
        let prior = prior.function(x).unwrap();
        let likelihood = self.model.likelihood;
        let likelihood = likelihood.function(x).unwrap();
        let norm = (self.model.likelihood * self.model.prior).integral()?;
        let res = izip!(prior, likelihood)
            .map(|(p, l)| match (p, l, norm) {
                (Some(p), Some(l), norm) => Some((p * l) / norm),
                _ => None,
            })
            .collect();
        Ok(res)
    }
}


impl Function<f64, f64, anyhow::Error> for Predictive {
    fn function(&self, x: f64) -> Result<f64, anyhow::Error> {
        let mut likelihood = self.0.likelihood;
        likelihood.update_observation(x);
        let model = self.0.prior * likelihood;
        model
            .integral()
            .map_err(|_| anyhow::Error::msg("Error with integration"))
    }
}

impl Function<&[f64], Vec<Option<f64>>, anyhow::Error> for Predictive {
    fn function(&self, x: &[f64]) -> Result<Vec<Option<f64>>, anyhow::Error> {
        let mut likelihood = self.0.likelihood;
        let res: Vec<Option<f64>> = x
            .iter()
            .map(|x| {
                likelihood.update_observation(*x);
                let model = self.0.prior * likelihood;
                model.integral().ok()
            })
            .collect();
        Ok(res)
    }
}

impl Mul<Prior> for Likelihood {
    type Output = Model;
    /// Create a model from a prior and a likelihood
    /// ```
    /// # use bayesplay::prelude::*;
    /// # fn main() {
    /// //let likelihood = likelihood!(family = "normal", mean = 0.2, se = 4.0);
    /// //let prior = prior!(family = "normal", mean = 0.0, sd = 1.0);
    /// //let model = likelihood * prior;
    /// # }
    /// ```
    fn mul(self, rhs: Prior) -> Self::Output {
        let prior = rhs;
        let likelihood = self;
        let range = prior.range_or_default();
        Model {
            prior,
            likelihood,
            range,
        }
    }
}

impl Mul<Likelihood> for Prior {
    type Output = Model;
    /// Create a model from a prior and a likelihood
    /// ```
    /// # use bayesplay::prelude::*;
    /// # fn main() {
    /// //let likelihood = likelihood!(family = "normal", mean = 0.2, se = 4.0);
    /// //let prior = prior!(family = "normal", mean = 0.0, sd = 1.0);
    /// //let model = prior * likelihood;
    /// # }
    /// ```
    fn mul(self, rhs: Likelihood) -> Self::Output {
        let likelihood = rhs;
        let prior = self;
        let range = prior.range_or_default();
        Model {
            prior,
            likelihood,
            range,
        }
    }
}

impl Range for Posterior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        let prior = self.model.prior;
        prior.range()
    }
    fn default_range(&self) -> (f64, f64) {
        let prior = self.model.prior;
        prior.default_range()
    }
}

impl Integrate<IntegralError, anyhow::Error> for Posterior {
    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, IntegralError> {
        let posterior = *self;
        let lb = lb.unwrap_or(posterior.range_or_default().0);
        let ub = ub.unwrap_or(posterior.range_or_default().1);

        let f = move |x| posterior.function(x).unwrap();
        let h = integrate!(f = f, lower = lb, upper = ub);
        match h {
            Ok(v) => Ok(v.value),
            Err(e) => Err(IntegralError::Intergaration(e)),
        }
    }
    fn integral(&self) -> Result<f64, IntegralError> {
        let posterior = *self;
        let (lb, ub) = posterior.model.prior.range_or_default();
        let f = move |x| posterior.function(x).unwrap();
        let h = integrate!(f = f, lower = lb, upper = ub);
        match h {
            Ok(v) => Ok(v.value),
            Err(e) => Err(IntegralError::Intergaration(e)),
        }
    }
}
