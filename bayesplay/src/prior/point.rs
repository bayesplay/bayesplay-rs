use super::Normalize;
use super::{Prior, PriorError};
use crate::common::Range;
use crate::common::Validate;
use crate::common::Function;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub struct PointPrior {
    pub point: f64,
}

impl PointPrior {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(point: f64) -> Prior {
        Prior::Point(PointPrior { point })
    }
}

impl Validate<PriorError> for PointPrior {
    fn validate(&self) -> Result<(), PriorError> {
        if self.point.is_nan() {
            return Err(PriorError::InvalidValue(
                self.point,
                f64::NEG_INFINITY,
                f64::INFINITY,
            ));
        }
        Ok(())
    }
}

impl Function<f64, f64, PriorError> for PointPrior {
    fn function(&self, x: f64) -> Result<f64, PriorError> {
        self.validate()?;
        if x == self.point { Ok(1.0) } else { Ok(0.0) }
    }
}

impl Normalize for PointPrior {
    fn normalize(&self) -> Result<f64, PriorError> {
        Ok(1.0)
    }
}

impl Range for PointPrior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        (Some(self.point), Some(self.point))
    }
    fn default_range(&self) -> (f64, f64) {
        (self.point, self.point)
    }
}
