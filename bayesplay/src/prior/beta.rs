use rmath::dbeta;

use serde::Deserialize;
use serde::Serialize;

use crate::common::Function;
use crate::common::Range;
use crate::common::Validate;

use super::Normalize;
use super::Prior;
use super::PriorError;

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub struct BetaPrior {
    pub alpha: f64,
    pub beta: f64,
    pub range: (Option<f64>, Option<f64>),
}

impl BetaPrior {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(alpha: f64, beta: f64, range: (Option<f64>, Option<f64>)) -> Prior {
        Prior::Beta(BetaPrior { alpha, beta, range })
    }
}

impl Range for BetaPrior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        self.range
    }

    fn default_range(&self) -> (f64, f64) {
        (0.0, 1.0)
    }
}

impl Validate<PriorError> for BetaPrior {
    fn validate(&self) -> Result<(), PriorError> {
        let mut errors: Vec<PriorError> = Vec::with_capacity(4);
        let (lower, upper) = self.range_or_default();

        if lower < 0.0 || upper > 1.0 {
            errors.push(PriorError::InvalidRangeBounds)
        }

        if self.alpha.is_sign_negative() {
            errors.push(PriorError::InvalidShapeParameter(self.alpha))
        };

        if self.beta.is_sign_negative() {
            errors.push(PriorError::InvalidShapeParameter(self.beta))
        }

        if lower == upper {
            errors.push(PriorError::InvalidRange)
        }

        if errors.len() == 1 {
            return Err(errors[0].clone());
        }
        if errors.len() == 2 {
            return Err(PriorError::MultiError2(
                Box::new(errors[0].clone()),
                Box::new(errors[1].clone()),
            ));
        }

        if errors.len() == 3 {
            return Err(PriorError::MultiError3(
                Box::new(errors[0].clone()),
                Box::new(errors[1].clone()),
                Box::new(errors[2].clone()),
            ));
        }

        if errors.len() == 3 {
            return Err(PriorError::MultiError4(
                Box::new(errors[0].clone()),
                Box::new(errors[1].clone()),
                Box::new(errors[2].clone()),
                Box::new(errors[3].clone()),
            ));
        }

        Ok(())
    }
}

impl Function<f64, f64, PriorError> for BetaPrior {
    fn function(&self, x: f64) -> Result<f64, PriorError> {
        self.validate()?;

        // FIXME: Currently it'll error if you use a truncated beta distraction
        // But truncated betas are not currently permitted on the frontend
        if !self.has_default_range() {
            return Err(PriorError::InvalidRange);
        };

        let shape1 = self.alpha;
        let shape2 = self.beta;
        let k = 1.0;

        if !(0.0..=1.0).contains(&x) {
            Err(PriorError::InvalidValue(x, 0.0, 1.0))?;
        }

        Ok(dbeta!(x = x, shape1 = shape1, shape2 = shape2)
            .map_err(PriorError::DistributionError)?
            * k)
    }
}

impl Normalize for BetaPrior {
    fn normalize(&self) -> Result<f64, PriorError> {
        if !self.has_default_range() {
            todo!("Truncated beta priors are not allowed!")
        }
        // FIXME:: Currently it'll error if you use a truncated beta distraction
        Ok(1.0)
    }
}
