//! Helpers shared by the integration tests.

use bayesplay::prelude::*;

/// BF10 for `likelihood` under `alternative` (H1) versus `null` (H0).
pub fn bayes_factor(likelihood: Likelihood, alternative: Prior, null: Prior) -> f64 {
    let h1 = (likelihood * alternative).integral().unwrap();
    let h0 = (likelihood * null).integral().unwrap();
    h1 / h0
}
