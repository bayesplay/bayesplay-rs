use rmath::dcauchy;
use rmath::pcauchy;

use serde::{Deserialize, Serialize};

use super::Normalize;
use super::PriorError;
use crate::common::truncated_normalization;
use crate::common::Function;
use crate::common::Range;
use crate::common::Validate;

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub struct CauchyPrior {
    pub location: f64,
    pub scale: f64,
    pub range: (Option<f64>, Option<f64>),
}

impl CauchyPrior {
    pub fn new(location: f64, scale: f64, range: (Option<f64>, Option<f64>)) -> Self {
        CauchyPrior {
            location,
            scale,
            range,
        }
    }
}

// FIXME: This should return multiple errors if necessary
impl Validate<PriorError> for CauchyPrior {
    fn validate(&self) -> Result<(), PriorError> {
        if self.scale <= 0.0 {
            Err(PriorError::InvalidScale(self.scale))?;
        }

        if !self.has_default_range() & (self.range.0 == self.range.1) {
            Err(PriorError::InvalidRange)?;
        }
        Ok(())
    }
}

impl Range for CauchyPrior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        self.range
    }

    fn default_range(&self) -> (f64, f64) {
        (-f64::INFINITY, f64::INFINITY)
    }
}

impl Normalize for CauchyPrior {
    fn normalize(&self) -> Result<f64, PriorError> {
        let (lower, upper) = self.range_or_default();
        let res = truncated_normalization(lower, upper, |x, lower_tail| {
            pcauchy!(
                q = x,
                location = self.location,
                scale = self.scale,
                lower_tail = lower_tail
            )
            .map_err(PriorError::DistributionError)
        })?;

        if res == 0.0 {
            Err(PriorError::NormalizingError)?;
        }
        Ok(res)
    }
}

impl Function<f64, f64, PriorError> for CauchyPrior {
    fn function(&self, x: f64) -> Result<f64, PriorError> {
        self.validate()?;
        let location = self.location;
        let scale = self.scale;

        if self.has_default_range() {
            return dcauchy!(x = x, location = location, scale = scale)
                .map_err(PriorError::DistributionError);
        }

        let (lower, upper) = self.range_or_default();

        if !(lower..=upper).contains(&x) {
            return Ok(0.0);
        }
        let k = self.normalize()?;
        Ok(dcauchy!(x = x, location = location, scale = scale)
            .map_err(PriorError::DistributionError)?
            / k)
    }
}
