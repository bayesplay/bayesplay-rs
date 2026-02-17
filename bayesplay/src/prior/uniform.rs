use rmath::dunif;
use serde::{Deserialize, Serialize};

use super::Normalize;
use crate::common::Function;
use crate::common::Range;
use crate::common::Validate;

use super::PriorError;

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub struct UniformPrior {
    pub min: f64,
    pub max: f64,
}

impl UniformPrior {
    pub fn new(min: f64, max: f64) -> Self {
        UniformPrior { min, max }
    }
}

impl Range for UniformPrior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        (Some(self.min), Some(self.max))
    }

    fn default_range(&self) -> (f64, f64) {
        (-f64::INFINITY, f64::INFINITY)
    }
}

impl Function<f64, f64, PriorError> for UniformPrior {
    fn function(&self, x: f64) -> Result<f64, PriorError> {
        self.validate()?;
        let min = self.min;
        let max = self.max;
        Ok(dunif!(x = x, min = min, max = max))
    }
}

impl Validate<PriorError> for UniformPrior {
    fn validate(&self) -> Result<(), PriorError> {
        if self.min == self.max {
            Err(PriorError::InvalidRange)?;
        }

        if self.min > self.max {
            Err(PriorError::InvalidRange)?;
        }

        Ok(())
    }
}

impl Normalize for UniformPrior {
    fn normalize(&self) -> Result<f64, PriorError> {
        Ok(1.0)
    }
}
