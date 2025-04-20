use bayesplay::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::Index;
use thiserror::Error;
#[derive(Debug, PartialEq, Serialize, Clone, Copy, Ord, PartialOrd, Eq, Hash)]
pub enum ParameterName {
    Location,
    Scale,
    Sd,
    Mean,
    N1,
    N2,
    N,
    D,
    T,
    Tails,
    Min,
    Max,
    Alpha,
    Beta,
    Point,
    Df,
    LL,
    UL,
    Trials,
    Successes,
    Se,
}

impl TryFrom<String> for ParameterName {
    type Error = &'static str;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "location" => Ok(ParameterName::Location),
            "scale" => Ok(ParameterName::Scale),
            "sd" => Ok(ParameterName::Sd),
            "se" => Ok(ParameterName::Se),
            "mean" => Ok(ParameterName::Mean),
            "n1" => Ok(ParameterName::N1),
            "n2" => Ok(ParameterName::N2),
            "n" => Ok(ParameterName::N),
            "d" => Ok(ParameterName::D),
            "min" => Ok(ParameterName::Min),
            "max" => Ok(ParameterName::Max),
            "alpha" => Ok(ParameterName::Alpha),
            "beta" => Ok(ParameterName::Beta),
            "point" => Ok(ParameterName::Point),
            "tails" => Ok(ParameterName::Tails),
            "ll" => Ok(ParameterName::LL),
            "ul" => Ok(ParameterName::UL),
            "t" => Ok(ParameterName::T),
            "df" => Ok(ParameterName::Df),
            "trials" => Ok(ParameterName::Trials),
            "successes" => Ok(ParameterName::Successes),
            _ => Err("Unknown Parameter"),
        }
    }
}

impl<'de> Deserialize<'de> for ParameterName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        match TryInto::<ParameterName>::try_into(name) {
            Ok(v) => Ok(v),
            _ => Err(serde::de::Error::custom("Unknown parameter")),
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Copy)]
pub struct ParamSetting {
    pub name: ParameterName,
    pub value: Option<f64>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct ParamDefinition(pub Vec<ParamSetting>);

impl Index<ParameterName> for ParamDefinition {
    type Output = Option<f64>;
    fn index(&self, index: ParameterName) -> &Self::Output {
        let i = self.0.iter().position(|r| r.name == index);
        match i {
            None => &None,
            Some(i) => &self.0[i].value,
        }
    }
}

#[derive(PartialEq, Debug, Serialize, Clone, Copy)]
pub enum LikelihoodFamily {
    Normal,
    NoncentralD,
    StudentT,
    NoncentralD2,
    NoncentralT,
    Binomial,
}

impl<'de> Deserialize<'de> for LikelihoodFamily {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = String::deserialize(deserializer)?;
        LikelihoodFamily::try_from(v.as_str())
            .map_err(|v| serde::de::Error::custom(format!("Unknown likelihood family: {}", v)))
    }
}

impl TryFrom<&str> for LikelihoodFamily {
    type Error = String;
    fn try_from(family: &str) -> Result<Self, Self::Error> {
        match family {
            "noncentral_d" => Ok(LikelihoodFamily::NoncentralD),
            "normal" => Ok(LikelihoodFamily::Normal),
            "student_t" => Ok(LikelihoodFamily::StudentT),
            "noncentral_d2" => Ok(LikelihoodFamily::NoncentralD2),
            "binomial" => Ok(LikelihoodFamily::Binomial),
            "noncentral_t" => Ok(LikelihoodFamily::NoncentralT),
            f => Err(f.to_string()),
        }
    }
}

#[derive(Debug, Error)]
pub enum InterfaceError {
    #[error("{0} is missing from interface")]
    MissingLikelihoodParameter(&'static str),
    #[error("{0} is missing from prior interface")]
    MissingPriorParameter(&'static str),
    #[error("{0} could not be parsed")]
    ParseError(&'static str),
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
/// ## Likelihood JSON interface
///
/// External interface for [Likelihood] definitions, used for serializing and
/// deserializing from/to JSON.
/// ```
/// # use interface::LikelihoodInterface;
/// # fn main() {
///   let json_string = r#"
///    {
///      "family": "normal",
///      "params": [
///        {
///          "name": "mean",
///          "value": 0
///        },
///        {
///          "name": "sd",
///          "value": 0.707
///        }
///      ]
///    }
/// "#;
///
/// let likelihood: LikelihoodInterface = serde_json::from_str(&json_string)
///     .expect("Could not parse likelihood interface");
/// #  }
/// ```
///
pub struct LikelihoodInterface {
    /// ## The distribution family for the likelihood
    /// The following distributions families can be used for the likelihood
    ///
    /// - `normal` a normal distribution
    /// - `student_t` a scaled and shifted t-distribution
    /// - `noncentral_t` a non-central t-distribution
    /// - `noncentral_d` a non-central t-distribution (scaled for one-sample d)
    /// - `noncentral_d2` a non-central t-distribution (scaled for two-sample d)
    /// - `binomial` a binomial distribution
    ///
    /// The parameters that need to be specified will be dependent on the family
    pub family: LikelihoodFamily,
    /// ## The parameter list for the likelihood.
    ///
    /// The parameter list for a likelihood should be defined in JSON as an
    /// array of [ParamDefinition].
    ///
    /// Each [ParamDefinition] needs the following fields:
    /// - `name` the name of the parameter
    /// - `value` the value of the parameter
    ///
    /// Valid parameter names will depend on the value for `LikelihoodFamily`.
    /// See below for details.
    ///
    /// ### Normal distribution (`LikelihoodFamily::Normal`)
    /// - `mean` mean of the normal distribution
    /// - `se` standard error of the normal distribution
    ///
    /// ### Student T distribution (`LikelihoodFamily::StudentT`)
    /// - `mean` mean of the t-distribution
    /// - `sd` standard deviation of the t-distribution
    /// - `df` degrees of freedom
    ///
    /// ### Non-central T distribution (`LikelihoodFamily::NoncentralT`)
    /// - `t` the t-value
    /// - `df` degrees of freedom
    ///
    /// ### Non-central D distribution (`LikelihoodFamily::NoncentralD`)
    /// - `d` the d-value (one-sample d)
    /// - `n` the sample size
    ///
    /// ### Non-central D2 distribution (`LikelihoodFamily::NoncentralD2`)
    /// - `d` the d-value (two-sample d)
    /// - `n1` the sample size for group 1
    /// - `n2` the sample size for group 2
    ///
    /// ### Binomial distribution (`LikelihoodFamily::Binomial`)
    /// - `trials` the number of trials
    /// - `successes` the number of successes
    pub params: ParamDefinition,
}

impl TryFrom<LikelihoodInterface> for Likelihood {
    type Error = InterfaceError;
    /// Converts a `LikelihoodInterface` into a `Likelihood`.
    ///
    /// # Examples
    ///
    /// ```
    /// use interface::{LikelihoodInterface, LikelihoodFamily, ParamDefinition, ParamSetting, ParameterName};
    /// use bayesplay::prelude::*;
    /// use std::convert::TryFrom;
    ///
    /// let likelihood_interface = LikelihoodInterface {
    ///     family: LikelihoodFamily::Normal,
    ///     params: ParamDefinition(vec![
    ///         ParamSetting {
    ///             name: ParameterName::Mean,
    ///             value: Some(0.0),
    ///         },
    ///         ParamSetting {
    ///             name: ParameterName::Sd,
    ///             value: Some(1.0),
    ///         },
    ///     ]),
    /// };
    ///
    /// let likelihood = Likelihood::try_from(likelihood_interface)
    ///     .expect("Failed to convert LikelihoodInterface to Likelihood");
    /// assert_eq!(likelihood.get_observation(), Some(0.0));
    /// ```
    fn try_from(value: LikelihoodInterface) -> Result<Self, Self::Error> {
        let likelihood = match value.family {
            LikelihoodFamily::Normal => {
                let mean = value.params[ParameterName::Mean]
                    .ok_or(Self::Error::MissingLikelihoodParameter("mean"))?;
                let se = value.params[ParameterName::Se]
                    .ok_or(Self::Error::MissingLikelihoodParameter("se"))?;
                NormalLikelihood::new(mean, se)
            }
            LikelihoodFamily::StudentT => {
                let mean = value.params[ParameterName::Mean]
                    .ok_or(Self::Error::MissingLikelihoodParameter("mean"))?;
                let sd = value.params[ParameterName::Sd]
                    .ok_or(Self::Error::MissingLikelihoodParameter("sd"))?;
                let df = value.params[ParameterName::Df]
                    .ok_or(Self::Error::MissingLikelihoodParameter("df"))?;
                StudentTLikelihood::new(mean, sd, df)
            }
            LikelihoodFamily::NoncentralD => {
                let d = value.params[ParameterName::D]
                    .ok_or(Self::Error::MissingLikelihoodParameter("d"))?;
                let n = value.params[ParameterName::N]
                    .ok_or(Self::Error::MissingLikelihoodParameter("n"))?;
                NoncentralDLikelihood::new(d, n)
            }
            LikelihoodFamily::NoncentralD2 => {
                let d = value.params[ParameterName::D]
                    .ok_or(Self::Error::MissingLikelihoodParameter("d"))?;
                let n1 = value.params[ParameterName::N1]
                    .ok_or(Self::Error::MissingLikelihoodParameter("n1"))?;
                let n2 = value.params[ParameterName::N2]
                    .ok_or(Self::Error::MissingLikelihoodParameter("n2"))?;
                NoncentralD2Likelihood::new(d, n1, n2)
            }
            LikelihoodFamily::NoncentralT => {
                let t = value.params[ParameterName::T]
                    .ok_or(Self::Error::MissingLikelihoodParameter("t"))?;
                let df = value.params[ParameterName::Df]
                    .ok_or(Self::Error::MissingLikelihoodParameter("df"))?;
                NoncentralTLikelihood::new(t, df)
            }
            LikelihoodFamily::Binomial => {
                let trials: f64 = value.params[ParameterName::Trials]
                    .ok_or(Self::Error::MissingLikelihoodParameter("trials"))?;
                let successes: f64 = value.params[ParameterName::Successes]
                    .ok_or(Self::Error::MissingLikelihoodParameter("successes"))?;
                BinomialLikelihood::new(successes, trials)
            }
        };
        Ok(likelihood)
    }
}

#[derive(PartialEq, Debug, Serialize, Clone, Copy)]
pub enum PriorFamily {
    Cauchy,
    Normal,
    Point,
    Uniform,
    StudentT,
    Beta,
}

impl<'de> Deserialize<'de> for PriorFamily {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = String::deserialize(deserializer)?;
        PriorFamily::try_from(v.as_str())
            .map_err(|v| serde::de::Error::custom(format!("Unknown Prior family: {}", v)))
    }
}

impl TryFrom<&str> for PriorFamily {
    type Error = String;
    fn try_from(family: &str) -> Result<Self, Self::Error> {
        match family {
            "cauchy" => Ok(PriorFamily::Cauchy),
            "normal" => Ok(PriorFamily::Normal),
            "point" => Ok(PriorFamily::Point),
            "uniform" => Ok(PriorFamily::Uniform),
            "student_t" => Ok(PriorFamily::StudentT),
            "beta" => Ok(PriorFamily::Beta),
            f => Err(f.to_string()),
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
/// ## Prior JSON interface
///
/// External interface for [Prior] definitions, used for serializing and
/// deserializing from/to JSON.
/// ```
/// # use interface::PriorInterface;
/// # fn main() {
///    let json_string = r#"
///     {
///       "family": "cauchy",
///       "params": [
///         {
///           "name": "location",
///           "value": 0
///         },
///         {
///           "name": "scale",
///           "value": 0.707
///         },
///         {
///           "name": "ll",
///           "value": null
///         },
///         {
///           "name": "ul",
///           "value": 0
///         }
///       ]
///     }
/// "#;
///
/// let prior: PriorInterface = serde_json::from_str(&json_string)
///     .expect("Could not parse prior interface");
/// #  }
/// ```
pub struct PriorInterface {
    /// ## The distribution family for the prior
    /// The following distributions families can be used for the prior
    ///
    /// - `normal` a normal distribution
    /// - `cauchy` a Cauchy distribution
    //  - `student_t` a scaled and shifted t-distribution
    //  - `uniform` a uniform distribution
    //  - `beta` a beta distribution
    /// - `point` a point
    ///
    /// The parameters that need to be specified will be dependent on the family
    pub family: PriorFamily,
    /// ## The parameter list for the prior.
    ///
    /// The parameter list for a prior should be defined in JSON as an
    /// array of [ParamDefinition].
    ///
    /// Each [ParamDefinition] needs the following fields:
    /// - `name` the name of the parameter
    /// - (optional) `value` the value of the parameter
    ///
    /// If `value` is missing then the default value is used. Default
    /// values are only supported for the `ll` and `ul` parameters
    ///
    ///
    /// Valid parameter names will depend on the value for `PriorFamily`.
    ///  See below for details.
    ///
    /// ### Normal distribution (`PriorFamily::Normal`)
    /// - `mean` mean of the normal distribution
    /// - `sd` standard deviation of the normal distribution
    /// - (optional) `ll` the lower limit (default: neg infinity)
    /// - (optional) `ul` the upper limit (default: pos infinity)
    ///
    /// ### Cauchy distribution (`PriorFamily::Cauchy`)
    /// - `location` the location of the Cauchy distribution
    /// - `scale` the scale of the Cauchy distribution
    /// - `ll` the lower limit (optional: default neg infinity)
    /// - `ul` the upper limit (optional: default pos infinity)
    ///
    /// ### Student T distribution (`PriorFamily::StudentT`)
    /// - `mean` mean of the t-distribution
    /// - `sd` standard deviation of the t-distribution
    /// - `df` degrees of freedom
    /// - (optional) `ll` the lower limit (default: neg infinity)
    /// - (optional) `ul` the upper limit (default: pos infinity)
    ///
    /// ### Uniform distribution (`PriorFamily::Uniform`)
    /// - `min` the lower limit of the uniform distribution
    /// - `max` the upper limit of the uniform distribution
    ///
    /// ### Beta distribution (`PriorFamily::Beta`)
    /// - `shape1` the first shape parameter of the beta distribution
    /// - `shape2` the second shape parameter of the beta distribution
    /// - (optional) `ll` the lower limit (default: 0)
    /// - (optional) `ul` the upper limit (default: 1)
    ///
    /// ### Point (`Prior::Point`)
    /// - `point` the location of the spike
    pub params: ParamDefinition,
}

impl TryFrom<PriorInterface> for Prior {
    type Error = InterfaceError;
    fn try_from(value: PriorInterface) -> Result<Self, Self::Error> {
        let prior = match value.family {
            PriorFamily::Cauchy => {
                let range = (
                    value.params[ParameterName::LL],
                    value.params[ParameterName::UL],
                );
                let location = value.params[ParameterName::Location]
                    .ok_or(InterfaceError::MissingPriorParameter("location"))?;
                let scale = value.params[ParameterName::Scale]
                    .ok_or(InterfaceError::MissingPriorParameter("scale"))?;
                CauchyPrior::new(location, scale, range)
            }
            PriorFamily::Point => {
                let point = value.params[ParameterName::Point]
                    .ok_or(InterfaceError::MissingPriorParameter("point"))?;
                PointPrior::new(point)
            }

            PriorFamily::Normal => {
                let range = (
                    value.params[ParameterName::LL],
                    value.params[ParameterName::UL],
                );

                let mean = value.params[ParameterName::Mean]
                    .ok_or(InterfaceError::MissingPriorParameter("mean"))?;

                let sd = value.params[ParameterName::Sd]
                    .ok_or(InterfaceError::MissingPriorParameter("sd"))?;
                NormalPrior::new(mean, sd, range)
            }
            PriorFamily::StudentT => {
                let range = (
                    value.params[ParameterName::LL],
                    value.params[ParameterName::UL],
                );

                let mean = value.params[ParameterName::Mean]
                    .ok_or(InterfaceError::MissingPriorParameter("mean"))?;
                let sd = value.params[ParameterName::Sd]
                    .ok_or(InterfaceError::MissingPriorParameter("sd"))?;
                let df = value.params[ParameterName::Df]
                    .ok_or(InterfaceError::MissingPriorParameter("df"))?;
                StudentTPrior::new(mean, sd, df, range)
            }
            PriorFamily::Uniform => {
                let min = value.params[ParameterName::Min]
                    .ok_or(InterfaceError::MissingPriorParameter("min"))?;
                let max = value.params[ParameterName::Max]
                    .ok_or(InterfaceError::MissingPriorParameter("max"))?;
                UniformPrior::new(min, max)
            }
            PriorFamily::Beta => {
                let range = (
                    value.params[ParameterName::LL],
                    value.params[ParameterName::UL],
                );
                let shape1 = value.params[ParameterName::Alpha]
                    .ok_or(InterfaceError::MissingPriorParameter("shape1"))?;
                let shape2 = value.params[ParameterName::Beta]
                    .ok_or(InterfaceError::MissingPriorParameter("shape2"))?;
                BetaPrior::new(shape1, shape2, range)
            }
        };

        Ok(prior)
    }
}
