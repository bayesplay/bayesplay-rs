use rmath::dt_scaled;
use rmath::pt_scaled;

use serde::{Deserialize, Serialize};

use super::Normalize;
use super::{Prior, PriorError};
use crate::common::Function;
use crate::common::Range;
use crate::common::Validate;

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub struct StudentTPrior {
    pub mean: f64,
    pub sd: f64,
    pub df: f64,
    pub range: (Option<f64>, Option<f64>),
}

impl StudentTPrior {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(mean: f64, sd: f64, df: f64, range: (Option<f64>, Option<f64>)) -> Prior {
        Prior::StudentT(StudentTPrior {
            mean,
            sd,
            df,
            range,
        })
    }
}

impl Validate<PriorError> for StudentTPrior {
    fn validate(&self) -> Result<(), PriorError> {
        let invalid_range = {
            let (lower, upper) = self.range_or_default();
            lower == upper
        };

        let mut errors: Vec<PriorError> = Vec::with_capacity(3);

        if self.sd <= 0.0 {
            errors.push(PriorError::InvalidStandardDeviation(self.sd))
        }

        if self.df <= 1.0 {
            errors.push(PriorError::InvalidDegreesOfFreedom(self.df))
        }
        if invalid_range {
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
        Ok(())
    }
}

impl Function<f64, f64, PriorError> for StudentTPrior {
    fn function(&self, x: f64) -> Result<f64, PriorError> {
        // self.validate()?;
        let mean = self.mean;
        let sd = self.sd;
        let df = self.df;
        if self.has_default_range() {
            return dt_scaled!(x = mean, mean = x, sd = sd, df = df)
                .map_err(PriorError::DistributionError);
        }

        let (lower, upper) = self.range_or_default();

        if x < lower || x > upper {
            Ok(0.0)
        } else {
            let k = 1.0 / self.normalize()?;
            Ok(dt_scaled!(x = mean, mean = x, sd = sd, df = df)
                .map_err(PriorError::DistributionError)?
                * k)
        }
    }
}

impl Normalize for StudentTPrior {
    fn normalize(&self) -> Result<f64, PriorError> {
        let (lower, upper) = self.range_or_default();
        let res = match (lower.is_infinite(), upper.is_infinite()) {
            (true, true) => 1.0,
            (false, true) => pt_scaled!(
                q = lower,
                mean = self.mean,
                sd = self.sd,
                df = self.df,
                lower_tail = false
            )
            .map_err(PriorError::DistributionError)?,
            (true, false) => pt_scaled!(
                q = upper,
                mean = self.mean,
                sd = self.sd,
                df = self.df,
                lower_tail = true
            )
            .map_err(PriorError::DistributionError)?,
            (false, false) => {
                (1.0 - pt_scaled!(
                    q = lower,
                    mean = self.mean,
                    sd = self.sd,
                    df = self.df,
                    lower_tail = true
                )
                .map_err(PriorError::DistributionError)?)
                    - (1.0
                        - (pt_scaled!(
                            q = upper,
                            mean = self.mean,
                            sd = self.sd,
                            df = self.df,
                            lower_tail = true
                        )
                        .map_err(PriorError::DistributionError)?))
            }
        };
        if res == 0.0 {
            Err(PriorError::NormalizingError)?;
        }
        Ok(res)
    }
}

impl Range for StudentTPrior {
    fn range(&self) -> (Option<f64>, Option<f64>) {
        self.range
    }

    fn default_range(&self) -> (f64, f64) {
        (-f64::INFINITY, f64::INFINITY)
    }
}
