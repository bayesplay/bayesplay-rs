use rmath::dcauchy;
use rmath::pcauchy;

use serde::{Deserialize, Serialize};

use super::Normalize;
use super::{Prior, PriorError};
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
    #[allow(clippy::new_ret_no_self)]
    pub fn new(location: f64, scale: f64, range: (Option<f64>, Option<f64>)) -> Prior {
        Prior::Cauchy(CauchyPrior {
            location,
            scale,
            range,
        })
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
        let res = match (lower.is_infinite(), upper.is_infinite()) {
            (true, true) => 1.0,
            (false, true) => pcauchy!(
                q = lower,
                location = self.location,
                scale = self.scale,
                lower_tail = false
            )
            .map_err(PriorError::DistributionError)?,
            (true, false) => pcauchy!(
                q = upper,
                location = self.location,
                scale = self.scale,
                lower_tail = true
            )
            .map_err(PriorError::DistributionError)?,
            (false, false) => {
                1.0 - pcauchy!(
                    q = lower,
                    location = self.location,
                    scale = self.scale,
                    lower_tail = true
                )
                .map_err(PriorError::DistributionError)?
                    - (1.0
                        - pcauchy!(
                            q = upper,
                            location = self.location,
                            scale = self.scale,
                            lower_tail = true
                        )
                        .map_err(PriorError::DistributionError)?)
            }
        };

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
