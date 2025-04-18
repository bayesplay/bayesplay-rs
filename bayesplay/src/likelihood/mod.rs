use thiserror::Error;
pub(crate) mod binomial;
pub(crate) mod noncentral_d;
pub(crate) mod noncentral_d2;
pub(crate) mod noncentral_t;
pub(crate) mod normal;
pub(crate) mod student_t;

use crate::common::Function;
use crate::common::Validate;

pub use binomial::BinomialLikelihood;
pub use noncentral_d::NoncentralDLikelihood;
pub use noncentral_d2::NoncentralD2Likelihood;
pub use noncentral_t::NoncentralTLikelihood;
pub use normal::NormalLikelihood;
pub use student_t::StudentTLikelihood;
// use crate::plot::Plot;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub enum Likelihood {
    Normal(NormalLikelihood),
    Binomial(BinomialLikelihood),
    StudentT(StudentTLikelihood),
    NoncentralD(NoncentralDLikelihood),
    NoncentralD2(NoncentralD2Likelihood),
    NoncentralT(NoncentralTLikelihood),
}

//
// #[derive(PartialEq, Debug, Serialize, Clone, Copy)]
// pub enum LikelihoodFamily {
//     Normal,
//     NoncentralD,
//     StudentT,
//     NoncentralD2,
//     NoncentralT,
//     Binomial,
// }

/*
// use std::fmt::{Display, Formatter};
* TODO: Decide if we need to implement Display for LikelihoodFamily
impl Display for LikelihoodFamily {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LikelihoodFamily::Normal => write!(f, "normal"),
            LikelihoodFamily::NoncentralD => write!(f, "noncentral_d"),
            LikelihoodFamily::StudentT => write!(f, "student_t"),
            LikelihoodFamily::NoncentralD2 => write!(f, "noncentral_d2"),
            LikelihoodFamily::NoncentralT => write!(f, "noncentral_t"),
            LikelihoodFamily::Binomial => write!(f, "binomial"),
        }
    }
}

impl From<LikelihoodFamily> for String {
    fn from(family: LikelihoodFamily) -> Self {
        format!("{}", family)
    }
}
*/

// impl<'de> Deserialize<'de> for LikelihoodFamily {
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: serde::Deserializer<'de>,
//     {
//         let v = String::deserialize(deserializer)?;
//         LikelihoodFamily::try_from(v.as_str())
//             .map_err(|v| serde::de::Error::custom(format!("Unknown likelihood family: {}", v)))
//     }
// }

// impl TryFrom<&str> for LikelihoodFamily {
//     type Error = String;
//     fn try_from(family: &str) -> Result<Self, Self::Error> {
//         match family {
//             "noncentral_d" => Ok(LikelihoodFamily::NoncentralD),
//             "normal" => Ok(LikelihoodFamily::Normal),
//             "student_t" => Ok(LikelihoodFamily::StudentT),
//             "noncentral_d2" => Ok(LikelihoodFamily::NoncentralD2),
//             "binomial" => Ok(LikelihoodFamily::Binomial),
//             "noncentral_t" => Ok(LikelihoodFamily::NoncentralT),
//             f => Err(f.to_string()),
//         }
//     }
// }

#[derive(Error, Debug, Serialize, Deserialize)]
pub enum LikelihoodError {
    #[error("SD value of {0} is invalid. SD must be positive")]
    InvalidSD(f64),
    #[error("SE value of {0} is invalid. SE must be positive")]
    InvalidSE(f64),
    #[error("N value of {0} is invalid. N must be positive")]
    InvalidN(f64),
    #[error("N1 value of {0} is invalid. N1 must be positive")]
    InvalidN1(f64),
    #[error("N2 value of {0} is invalid. N2 must be positive")]
    InvalidN2(f64),
    #[error("Trials value of {0} is invalid. Trails must be a positive integer")]
    InvalidTrials(f64),
    #[error("Success value of {0} is invalid. Success must be positive integer and less Trials")]
    InvalidSuccess(f64),
    #[error("DF value of {0} is invalid. DF must be positive")]
    InvalidDF(f64),
    #[error("Invalid probability value of {0}. Probability must be between 0 and 1")]
    InvalidProbability(f64),
    #[error("Error with distribution: {0}")]
    DistributionError(&'static str),
    #[error("Multiple errors: {0}, {1}]")]
    MultiError2(Box<LikelihoodError>, Box<LikelihoodError>),
    #[error("Multiple errors: {0}, {1}, {2}]")]
    MultiError3(
        Box<LikelihoodError>,
        Box<LikelihoodError>,
        Box<LikelihoodError>,
    ),
    #[error("Multiple errors: {0}, {1}, {2}, {3}]")]
    MultiError4(
        Box<LikelihoodError>,
        Box<LikelihoodError>,
        Box<LikelihoodError>,
        Box<LikelihoodError>,
    ),
}

pub trait Observation {
    fn update_observation(&mut self, observation: f64);
    fn get_observation(&self) -> Option<f64>;
}

impl Function<f64, f64, LikelihoodError> for Likelihood {
    fn function(&self, x: f64) -> Result<f64, LikelihoodError> {
        match self {
            Likelihood::Normal(n) => n.function(x),
            Likelihood::Binomial(n) => n.function(x),
            Likelihood::StudentT(n) => n.function(x),
            Likelihood::NoncentralD(n) => n.function(x),
            Likelihood::NoncentralD2(n) => n.function(x),
            Likelihood::NoncentralT(n) => n.function(x),
        }
    }
}

impl Function<&[f64], Vec<Option<f64>>, LikelihoodError> for Likelihood {
    fn function(&self, x: &[f64]) -> Result<Vec<Option<f64>>, LikelihoodError> {
        Ok(x.iter()
            .map(|x| self.function(*x).ok())
            .collect::<Vec<Option<_>>>())
    }
}

// impl Plot for Likelihood {
//     fn center(&self) -> f64 {
//         0.0
//     }
//     fn spread(&self) -> f64 {
//         0.0
//     }
// }


impl Validate<LikelihoodError> for Likelihood {
    fn validate(&self) -> Result<(), LikelihoodError> {
        let ok = match self {
            Likelihood::Normal(likelihood) => likelihood.validate(),
            Likelihood::Binomial(likelihood) => likelihood.validate(),
            Likelihood::StudentT(likelihood) => likelihood.validate(),
            Likelihood::NoncentralD(likelihood) => likelihood.validate(),
            Likelihood::NoncentralD2(likelihood) => likelihood.validate(),
            Likelihood::NoncentralT(likelihood) => likelihood.validate(),
        };
        match ok {
            Ok(()) => Ok(()),
            Err(v) => Err(v),
        }
    }
}

impl Likelihood {

    pub fn update_observation(&mut self, observation: f64) {
        match self {
            Likelihood::Normal(likelihood) => likelihood.update_observation(observation),
            Likelihood::Binomial(likelihood) => likelihood.update_observation(observation),
            Likelihood::StudentT(likelihood) => likelihood.update_observation(observation),
            Likelihood::NoncentralD(likelihood) => likelihood.update_observation(observation),
            Likelihood::NoncentralD2(likelihood) => likelihood.update_observation(observation),
            Likelihood::NoncentralT(likelihood) => likelihood.update_observation(observation),
        }
    }

    pub fn get_observation(&self) -> Option<f64> {
        match self {
            Likelihood::Normal(likelihood) => likelihood.get_observation(),
            Likelihood::Binomial(likelihood) => likelihood.get_observation(),
            Likelihood::StudentT(likelihood) => likelihood.get_observation(),
            Likelihood::NoncentralD(likelihood) => likelihood.get_observation(),
            Likelihood::NoncentralD2(likelihood) => likelihood.get_observation(),
            Likelihood::NoncentralT(likelihood) => likelihood.get_observation(),
        }
    }

    // pub fn type_of(&self) -> LikelihoodFamily {
    //     match self {
    //         Likelihood::Normal(_) => LikelihoodFamily::Normal,
    //         Likelihood::Binomial(_) => LikelihoodFamily::Binomial,
    //         Likelihood::StudentT(_) => LikelihoodFamily::StudentT,
    //         Likelihood::NoncentralD(_) => LikelihoodFamily::NoncentralD,
    //         Likelihood::NoncentralD2(_) => LikelihoodFamily::NoncentralD2,
    //         Likelihood::NoncentralT(_) => LikelihoodFamily::NoncentralT,
    //     }
    // }
    //
    // pub fn plot(self) -> Plot {
    //     Plot::Likelihood(self)
    // }
}

// #[macro_export]
// macro_rules! likelihood {
//
//
//     ($family:ident = $f:expr, $($a:tt = $c:expr ),*) => {
//         {
//
//             use $crate::likelihood::*;
//
//             struct LikelihoodArgs {
//               // family: Option<LikelihoodFamily>,
//             mean: Option<f64>,
//             sd: Option<f64>,
//             se: Option<f64>,
//             d: Option<f64>,
//             n: Option<f64>,
//             n1: Option<f64>,
//             n2: Option<f64>,
//             df: Option<f64>,
//             t: Option<f64>,
//             successes: Option<u64>,
//             trials: Option<u64>
//             }
//
//             impl Default for LikelihoodArgs {
//                 fn default() -> Self {
//                     LikelihoodArgs {
//                         // family: None,
//                         mean: None,
//                         sd: None,
//                         se: None,
//                         d: None,
//                         n: None,
//                         n1: None,
//                         n2: None,
//                         df: None,
//                         t: None,
//                         successes: None,
//                         trials: None
//                     }
//                 }
//             }
//
//
//
//             let family: LikelihoodFamily = $f.try_into().expect("Unknown likelihood family");
//             let args = LikelihoodArgs {
//                 // family: Some($f.into()),
//                 $($a: Some($c),)*
//                 ..Default::default()
//             };
//
//             match family {
//                 LikelihoodFamily::Normal => {
//                NormalLikelihood::new(
//                         args.mean.expect("mean is required"),
//                         args.se.expect("se is required")
//                     )
//                 },
//                 LikelihoodFamily::StudentT => {
//                     StudentTLikelihood::new(
//                         args.mean.expect("mean is required"),
//                         args.sd.expect("sd is required"),
//                         args.df.expect("df is required")
//                 )
//                 },
//                 LikelihoodFamily::NoncentralD2 => {
//                     NoncentralD2Likelihood::new(args.d.expect("d is required"),
//                         args.n1.expect("n is required"),
//                         args.n2.expect("n2 is required"))
//
//                 },
//                 LikelihoodFamily::NoncentralT => {
//                     NoncentralTLikelihood::new(
//                         args.t.expect("t is required"),
//                         args.df.expect("df is required")
//         )
//                 },
//             LikelihoodFamily::Binomial => {
//                     BinomialLikelihood::new(
//                         args.successes.expect("successes is required"),
//                         args.trials.expect("trials is required")
//                     )
//                 },
//                 LikelihoodFamily::NoncentralD => {
//                     Likelihood::NoncentralD(NoncentralDLikelihood{
//                         d: args.d.expect("d is required"),
//                         n: args.n.expect("n is required")
//                     })
//                 },
//             }
//         }
//     }
// }

// #[cfg(test)]
// mod likelihood_macro_tests;
//
// #[cfg(test)]
// mod test_traits;
