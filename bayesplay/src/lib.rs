//! # Bayesplay
//!
//! A Rust library for Bayesian inference, providing tools for creating prior and
//! likelihood distributions, combining them into models, and computing posteriors
//! and Bayes factors.
//!
//! ## Overview
//!
//! Bayesplay provides a type-safe API for Bayesian statistical analysis. The main
//! workflow involves:
//!
//! 1. Creating a **likelihood** from observed data
//! 2. Specifying a **prior** distribution representing beliefs before seeing data
//! 3. Combining them into a **model**
//! 4. Computing the **posterior** distribution or Bayes factors
//!
//! ## Quick Start
//!
//! ```rust
//! use bayesplay::prelude::*;
//!
//! // Create a likelihood from observed data (mean = 0.5, standard error = 0.2)
//! let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
//!
//! // Create a prior distribution (normal with mean = 0, sd = 1)
//! let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
//!
//! // Combine into a model using multiplication
//! let model: Model = likelihood * prior;
//!
//! // Compute the posterior distribution
//! let posterior = model.posterior().unwrap();
//!
//! // Evaluate the posterior at a point
//! let density = posterior.function(0.5).unwrap();
//! ```
//!
//! ## Available Distributions
//!
//! ### Likelihoods
//!
//! - [`prelude::NormalLikelihood`] - For continuous data with known standard error
//! - [`prelude::BinomialLikelihood`] - For count data (successes out of trials)
//! - [`prelude::StudentTLikelihood`] - For t-distributed data
//! - [`prelude::NoncentralDLikelihood`] - For one-sample effect sizes (Cohen's d)
//! - [`prelude::NoncentralD2Likelihood`] - For two-sample effect sizes
//! - [`prelude::NoncentralTLikelihood`] - For noncentral t-distributed data
//!
//! ### Priors
//!
//! - [`prelude::NormalPrior`] - Normal (Gaussian) prior
//! - [`prelude::CauchyPrior`] - Cauchy prior (heavy tails)
//! - [`prelude::StudentTPrior`] - Student's t prior
//! - [`prelude::UniformPrior`] - Uniform prior over an interval
//! - [`prelude::BetaPrior`] - Beta prior (for probabilities)
//! - [`prelude::PointPrior`] - Point mass prior (for null hypotheses)
//!
//! ## Truncated Distributions
//!
//! Many priors support truncation to restrict the parameter space:
//!
//! ```rust
//! use bayesplay::prelude::*;
//!
//! // Normal prior truncated to positive values only
//! let positive_normal = NormalPrior::new(0.0, 1.0, (Some(0.0), None));
//!
//! // Cauchy prior truncated between -2 and 2
//! let bounded_cauchy = CauchyPrior::new(0.0, 0.707, (Some(-2.0), Some(2.0)));
//! ```
//!
//! ## Computing Bayes Factors
//!
//! Compare models by computing the ratio of their marginal likelihoods:
//!
//! ```rust
//! use bayesplay::prelude::*;
//!
//! let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
//!
//! // Alternative hypothesis: effect exists
//! let h1_prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
//! let h1_model: Model = likelihood * h1_prior;
//!
//! // Null hypothesis: effect is exactly zero
//! let h0_prior: Prior = PointPrior::new(0.0).into();
//! let h0_model: Model = likelihood * h0_prior;
//!
//! // Bayes factor = marginal likelihood of H1 / marginal likelihood of H0
//! let bf = h1_model.integral().unwrap() / h0_model.integral().unwrap();
//! ```

mod common;
mod compute;
mod likelihood;
mod prior;

/// The prelude module re-exports the most commonly used types and traits.
///
/// Import everything from the prelude to get started quickly:
///
/// ```rust
/// use bayesplay::prelude::*;
/// ```
pub mod prelude {

    // Common traits
    pub use crate::common::Function;
    pub use crate::common::Integrate;
    pub use crate::common::Range;
    pub use crate::common::Validate;

    // General likelihood
    pub use crate::likelihood::Likelihood;
    pub use crate::likelihood::LikelihoodError;
    pub use crate::likelihood::LikelihoodResult;
    pub use crate::likelihood::Observation;

    // Specific likelihoods
    pub use crate::likelihood::BinomialLikelihood;
    pub use crate::likelihood::NoncentralD2Likelihood;
    pub use crate::likelihood::NoncentralDLikelihood;
    pub use crate::likelihood::NoncentralTLikelihood;
    pub use crate::likelihood::NormalLikelihood;
    pub use crate::likelihood::StudentTLikelihood;

    // General prior
    pub use crate::prior::Normalize;
    pub use crate::prior::Prior;
    pub use crate::prior::PriorError;
    pub use crate::prior::PriorFamily;
    pub use crate::prior::PriorResult;
    pub use crate::prior::TypeOf;

    // Specific prior
    pub use crate::prior::BetaPrior;
    pub use crate::prior::CauchyPrior;
    pub use crate::prior::NormalPrior;
    pub use crate::prior::PointPrior;
    pub use crate::prior::StudentTPrior;
    pub use crate::prior::UniformPrior;

    // For computing Bayes factors
    pub use crate::compute::model::Model;

    pub use crate::compute::model::estimate_marginal;
    pub use crate::compute::model::ApproximateModel;
    pub use crate::compute::model::IntegralError;
    pub use crate::compute::model::MarginalResult;
    pub use crate::compute::model::Posterior;
}
