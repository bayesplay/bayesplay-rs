// mod compute;
// mod plot;
mod common;
mod likelihood;
mod prior;
mod compute;

pub mod prelude {

    // Common traits
    pub use crate::common::Function;
    pub use crate::common::Validate;
    pub use crate::common::Range;
    pub use crate::common::Integrate;

    // General likelihood
    pub use crate::likelihood::Likelihood;
    pub use crate::likelihood::Observation;
    pub use crate::likelihood::LikelihoodError;
   
    // Specific likelihoods
    pub use crate::likelihood::NormalLikelihood;
    pub use crate::likelihood::BinomialLikelihood;
    pub use crate::likelihood::StudentTLikelihood;
    pub use crate::likelihood::NoncentralDLikelihood;
    pub use crate::likelihood::NoncentralD2Likelihood;
    pub use crate::likelihood::NoncentralTLikelihood;

    // General prior
    pub use crate::prior::Prior;
    pub use crate::prior::PriorError;
    pub use crate::prior::Normalize;
    pub use crate::prior::TypeOf;

    // Specific prior
    pub use crate::prior::NormalPrior;
    pub use crate::prior::PointPrior;
    pub use crate::prior::CauchyPrior;
    pub use crate::prior::UniformPrior;
    pub use crate::prior::StudentTPrior;
    pub use crate::prior::BetaPrior;

    // Names of prior
    pub use crate::prior::PriorFamily;

    // For computing Bayes factors
    pub use crate::compute::model::Model;

    pub use crate::compute::model::Posterior;
}

