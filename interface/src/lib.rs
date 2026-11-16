use bayesplay::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::Index;
use thiserror::Error;

/// A parameter name as it appears in JSON (`"mean"`, `"n1"`, `"ll"`, ...).
#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Copy, Ord, PartialOrd, Eq, Hash)]
#[serde(rename_all = "snake_case")]
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
    #[serde(rename = "ll")]
    LL,
    #[serde(rename = "ul")]
    UL,
    Trials,
    Successes,
    Se,
    U,
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
            "u" => Ok(ParameterName::U),
            _ => Err("Unknown Parameter"),
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

/// A likelihood family as it appears in JSON (`"normal"`, `"noncentral_d2"`, ...).
#[derive(PartialEq, Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum LikelihoodFamily {
    Normal,
    NoncentralD,
    StudentT,
    NoncentralD2,
    NoncentralT,
    Binomial,
    VarianceRatio,
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
            "variance_ratio" => Ok(LikelihoodFamily::VarianceRatio),
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
    #[error(transparent)]
    Prior(#[from] PriorError),
    #[error(transparent)]
    Likelihood(#[from] LikelihoodError),
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
///          "name": "se",
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
    /// - `variance_ratio` a likelihood for a variance ratio against a reference
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
    ///
    /// ### Variance ratio (`LikelihoodFamily::VarianceRatio`)
    /// - `u` the observed variance ratio
    /// - `df` degrees of freedom
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
    ///             name: ParameterName::Se,
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
                NormalLikelihood::new(mean, se)?.into()
            }
            LikelihoodFamily::StudentT => {
                let mean = value.params[ParameterName::Mean]
                    .ok_or(Self::Error::MissingLikelihoodParameter("mean"))?;
                let sd = value.params[ParameterName::Sd]
                    .ok_or(Self::Error::MissingLikelihoodParameter("sd"))?;
                let df = value.params[ParameterName::Df]
                    .ok_or(Self::Error::MissingLikelihoodParameter("df"))?;
                StudentTLikelihood::new(mean, sd, df)?.into()
            }
            LikelihoodFamily::NoncentralD => {
                let d = value.params[ParameterName::D]
                    .ok_or(Self::Error::MissingLikelihoodParameter("d"))?;
                let n = value.params[ParameterName::N]
                    .ok_or(Self::Error::MissingLikelihoodParameter("n"))?;
                NoncentralDLikelihood::new(d, n)?.into()
            }
            LikelihoodFamily::NoncentralD2 => {
                let d = value.params[ParameterName::D]
                    .ok_or(Self::Error::MissingLikelihoodParameter("d"))?;
                let n1 = value.params[ParameterName::N1]
                    .ok_or(Self::Error::MissingLikelihoodParameter("n1"))?;
                let n2 = value.params[ParameterName::N2]
                    .ok_or(Self::Error::MissingLikelihoodParameter("n2"))?;
                NoncentralD2Likelihood::new(d, n1, n2)?.into()
            }
            LikelihoodFamily::NoncentralT => {
                let t = value.params[ParameterName::T]
                    .ok_or(Self::Error::MissingLikelihoodParameter("t"))?;
                let df = value.params[ParameterName::Df]
                    .ok_or(Self::Error::MissingLikelihoodParameter("df"))?;
                NoncentralTLikelihood::new(t, df)?.into()
            }
            LikelihoodFamily::Binomial => {
                let trials: f64 = value.params[ParameterName::Trials]
                    .ok_or(Self::Error::MissingLikelihoodParameter("trials"))?;
                let successes: f64 = value.params[ParameterName::Successes]
                    .ok_or(Self::Error::MissingLikelihoodParameter("successes"))?;
                BinomialLikelihood::new(successes, trials)?.into()
            }
            LikelihoodFamily::VarianceRatio => {
                let u = value.params[ParameterName::U]
                    .ok_or(Self::Error::MissingLikelihoodParameter("u"))?;
                let df = value.params[ParameterName::Df]
                    .ok_or(Self::Error::MissingLikelihoodParameter("df"))?;
                VarianceRatioLikelihood::new(u, df)?.into()
            }
        };
        Ok(likelihood)
    }
}

/// A prior family as it appears in JSON (`"cauchy"`, `"beta_prime"`, ...).
#[derive(PartialEq, Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum PriorFamily {
    Cauchy,
    Normal,
    Point,
    Uniform,
    StudentT,
    Beta,
    BetaPrime,
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
            "beta_prime" => Ok(PriorFamily::BetaPrime),
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
    /// - `student_t` a scaled and shifted t-distribution
    /// - `uniform` a uniform distribution
    /// - `beta` a beta distribution
    /// - `beta_prime` a symmetric beta-prime distribution (for positive ratios)
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
    /// - `alpha` the first shape parameter of the beta distribution
    /// - `beta` the second shape parameter of the beta distribution
    /// - (optional) `ll` the lower limit (default: 0)
    /// - (optional) `ul` the upper limit (default: 1)
    ///
    /// ### Beta-prime distribution (`PriorFamily::BetaPrime`)
    /// - `alpha` the shape parameter (both shapes are equal)
    /// - (optional) `ll` the lower limit (default: 0)
    /// - (optional) `ul` the upper limit (default: pos infinity)
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
                CauchyPrior::new(location, scale, range)?.into()
            }
            PriorFamily::Point => {
                let point = value.params[ParameterName::Point]
                    .ok_or(InterfaceError::MissingPriorParameter("point"))?;
                PointPrior::new(point)?.into()
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
                NormalPrior::new(mean, sd, range)?.into()
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
                StudentTPrior::new(mean, sd, df, range)?.into()
            }
            PriorFamily::Uniform => {
                let min = value.params[ParameterName::Min]
                    .ok_or(InterfaceError::MissingPriorParameter("min"))?;
                let max = value.params[ParameterName::Max]
                    .ok_or(InterfaceError::MissingPriorParameter("max"))?;
                UniformPrior::new(min, max)?.into()
            }
            PriorFamily::Beta => {
                let range = (
                    value.params[ParameterName::LL],
                    value.params[ParameterName::UL],
                );
                let alpha = value.params[ParameterName::Alpha]
                    .ok_or(InterfaceError::MissingPriorParameter("alpha"))?;
                let beta = value.params[ParameterName::Beta]
                    .ok_or(InterfaceError::MissingPriorParameter("beta"))?;
                BetaPrior::new(alpha, beta, range)?.into()
            }
            PriorFamily::BetaPrime => {
                let range = (
                    value.params[ParameterName::LL],
                    value.params[ParameterName::UL],
                );
                let alpha = value.params[ParameterName::Alpha]
                    .ok_or(InterfaceError::MissingPriorParameter("alpha"))?;
                BetaPrimePrior::new(alpha, range)?.into()
            }
        };

        Ok(prior)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_beta_prior_json_supports_density_and_marginal() {
        let prior = prior_from_json(
            r#"{"family":"beta","params":[{"name":"alpha","value":2},{"name":"beta","value":1},{"name":"ll","value":0.5}]}"#,
        );
        assert_eq!(prior.function(0.25).unwrap(), 0.0);
        assert!((prior.function(0.75).unwrap() - 2.0).abs() < 1e-12);
        let likelihood: Likelihood = BinomialLikelihood::new(1.0, 1.0).unwrap().into();
        assert!(((likelihood * prior).integral().unwrap().value - 7.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn interfaces_round_trip_through_json() {
        let likelihood = r#"{"family":"noncentral_d2","params":[{"name":"d","value":0.5},{"name":"n1","value":20.0},{"name":"n2","value":25.0}]}"#;
        let parsed: LikelihoodInterface = serde_json::from_str(likelihood).unwrap();
        assert_eq!(serde_json::to_string(&parsed).unwrap(), likelihood);

        let prior = r#"{"family":"cauchy","params":[{"name":"location","value":0.0},{"name":"scale","value":0.707},{"name":"ll","value":null},{"name":"ul","value":0.0}]}"#;
        let parsed: PriorInterface = serde_json::from_str(prior).unwrap();
        assert_eq!(serde_json::to_string(&parsed).unwrap(), prior);
    }

    #[test]
    fn serde_names_match_try_from_names() {
        use ParameterName::*;
        for name in [
            Location, Scale, Sd, Mean, N1, N2, N, D, T, Tails, Min, Max, Alpha, Beta, Point, Df,
            LL, UL, Trials, Successes, Se, U,
        ] {
            let json = serde_json::to_string(&name).unwrap();
            let text: String = serde_json::from_str(&json).unwrap();
            assert_eq!(ParameterName::try_from(text), Ok(name));
        }
        for family in [
            "normal",
            "noncentral_d",
            "student_t",
            "noncentral_d2",
            "noncentral_t",
            "binomial",
            "variance_ratio",
        ] {
            let parsed = LikelihoodFamily::try_from(family).unwrap();
            assert_eq!(
                serde_json::to_string(&parsed).unwrap(),
                format!("\"{family}\"")
            );
        }
        for family in [
            "cauchy",
            "normal",
            "point",
            "uniform",
            "student_t",
            "beta",
            "beta_prime",
        ] {
            let parsed = PriorFamily::try_from(family).unwrap();
            assert_eq!(
                serde_json::to_string(&parsed).unwrap(),
                format!("\"{family}\"")
            );
        }
    }

    #[test]
    fn variance_ratio_and_beta_prime_are_reachable_from_json() {
        let likelihood: LikelihoodInterface = serde_json::from_str(
            r#"{"family":"variance_ratio","params":[{"name":"u","value":1.6},{"name":"df","value":9}]}"#,
        )
        .unwrap();
        let likelihood = Likelihood::try_from(likelihood).unwrap();
        let prior = prior_from_json(
            r#"{"family":"beta_prime","params":[{"name":"alpha","value":0.5},{"name":"ll","value":1}]}"#,
        );
        assert!(matches!(likelihood, Likelihood::VarianceRatio(_)));
        assert!(matches!(prior, Prior::BetaPrime(_)));
        assert!((likelihood * prior).integral().unwrap().value > 0.0);
    }

    #[test]
    fn unknown_names_are_rejected() {
        assert!(serde_json::from_str::<ParameterName>(r#""nope""#).is_err());
        assert!(serde_json::from_str::<LikelihoodFamily>(r#""nope""#).is_err());
        assert!(serde_json::from_str::<PriorFamily>(r#""nope""#).is_err());
    }

    #[test]
    fn invalid_parameters_are_reported_by_the_interface() {
        let prior: PriorInterface = serde_json::from_str(
            r#"{"family":"normal","params":[{"name":"mean","value":0},{"name":"sd","value":-1}]}"#,
        )
        .unwrap();
        assert!(matches!(
            Prior::try_from(prior),
            Err(InterfaceError::Prior(PriorError::InvalidStandardDeviation(
                _
            )))
        ));
        let likelihood: LikelihoodInterface = serde_json::from_str(
            r#"{"family":"binomial","params":[{"name":"trials","value":10},{"name":"successes","value":11}]}"#,
        )
        .unwrap();
        assert!(matches!(
            Likelihood::try_from(likelihood),
            Err(InterfaceError::Likelihood(LikelihoodError::InvalidSuccess(
                _
            )))
        ));
    }

    fn prior_from_json(json: &str) -> Prior {
        let interface: PriorInterface =
            serde_json::from_str(json).expect("prior JSON should deserialize");
        Prior::try_from(interface).expect("prior interface should convert")
    }
}
