//! Integration tests based on examples from the bayesplay R package documentation.
//!
//! These tests are derived from:
//! - https://bayesplay.github.io/bayesplay/articles/basic.html
//! - https://bayesplay.github.io/bayesplay/articles/advanced.html
//! - https://bayesplay.github.io/bayesplay/articles/default_ttests.html
//!
//! The R package is the gold standard. Test expectations should match R output exactly.
//! If tests fail, it means the Rust implementation needs fixing, not the test expectations.

#[cfg(test)]
mod basic_examples {
    //! Tests from the basic usage article (basic.html)

    use crate::prelude::*;
    use approx::assert_relative_eq;

    /// Helper function to compute Bayes factor (H1/H0)
    fn bayes_factor(likelihood: Likelihood, alt_prior: Prior, null_prior: Prior) -> f64 {
        let m1 = (likelihood * alt_prior).integral().unwrap();
        let m0 = (likelihood * null_prior).integral().unwrap();
        m1 / m0
    }

    #[test]
    fn test_example1_half_normal_prior() {
        // From basic.html Example 1 (Dienes & Mclatchie 2018):
        // data_mod <- likelihood(family = "normal", mean = 5.5, sd = 32.35)
        // h0_mod <- prior(family = "point", point = 0)
        // h1_mod <- prior(family = "normal", mean = 0, sd = 13.3, range = c(0, Inf))
        // bf <- m1 / m0
        // bf
        // #> 0.9745934

        let likelihood: Likelihood = NormalLikelihood::new(5.5, 32.35).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = NormalPrior::new(0.0, 13.3, (Some(0.0), None)).into(); // Half-normal (0 to Inf)

        let bf = bayes_factor(likelihood, alt_prior, null_prior);

        // R gives: 0.9745934
        assert_relative_eq!(bf, 0.9745934, epsilon = 0.001);
    }

    #[test]
    fn test_example2_uniform_prior() {
        // From basic.html Example 2 (Dienes 2014):
        // data_mod <- likelihood(family = "normal", mean = 5, sd = 10)
        // h1_mod <- prior(family = "uniform", min = 0, max = 20)
        // h0_mod <- prior(family = "point", point = 0)
        // bf <- integral(data_mod * h1_mod) / integral(data_mod * h0_mod)
        // bf
        // #> 0.8871298

        let likelihood: Likelihood = NormalLikelihood::new(5.0, 10.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = UniformPrior::new(0.0, 20.0).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);

        // R gives: 0.8871298
        assert_relative_eq!(bf, 0.8871298, epsilon = 0.001);
    }

    #[test]
    fn test_example3_noncentral_d_cauchy() {
        // From basic.html Example 3 (Rouder et al. 2009):
        // d <- 2.03 / sqrt(80) # convert t to d
        // data_model <- likelihood(family = "noncentral_d", d, 80)
        // h0_mod <- prior(family = "point", point = 0)
        // h1_mod <- prior(family = "cauchy", scale = 1)
        // bf <- integral(data_model * h0_mod) / integral(data_model * h1_mod)
        // bf
        // #> 1.557447
        //
        // Note: R computes BF as H0/H1 = 1.557447, so H1/H0 = 1/1.557447 = 0.6420768

        let d = 2.03 / 80.0_f64.sqrt();
        let likelihood: Likelihood = NoncentralDLikelihood::new(d, 80.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);

        // R gives BF(H0/H1) = 1.557447, so BF(H1/H0) = 1/1.557447 = 0.6420768
        assert_relative_eq!(bf, 1.0 / 1.557447, epsilon = 0.001);
    }

    #[test]
    fn test_example3_noncentral_d_normal() {
        // From basic.html Example 3 with unit-information prior (Rouder et al. 2009):
        // d <- 2.03 / sqrt(80)
        // data_model <- likelihood(family = "noncentral_d", d, 80)
        // h0_mod <- prior(family = "point", point = 0)
        // h1_mod <- prior(family = "normal", mean = 0, sd = 1)
        // bf <- integral(data_model * h0_mod) / integral(data_model * h1_mod)
        // bf
        // #> 1.208093
        //
        // Note: R computes BF as H0/H1 = 1.208093, so H1/H0 = 1/1.208093 = 0.8277

        let d = 2.03 / 80.0_f64.sqrt();
        let likelihood: Likelihood = NoncentralDLikelihood::new(d, 80.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);

        // R gives BF(H0/H1) = 1.208093, so BF(H1/H0) = 1/1.208093 = 0.8277
        assert_relative_eq!(bf, 1.0 / 1.208093, epsilon = 0.001);
    }

    #[test]
    fn test_example3_noncentral_t() {
        // From basic.html Example 3 using noncentral_t parametrization:
        // data_model <- likelihood(family = "noncentral_t", 2.03, 79)
        // h0_mod <- prior(family = "point", point = 0)
        // h1_mod <- prior(family = "cauchy", location = 0, scale = 1 * sqrt(80))
        // bf <- integral(data_model * h0_mod) / integral(data_model * h1_mod)
        // bf
        // #> 1.557447
        //
        // This should give the same result as noncentral_d with Cauchy(0, 1)

        let t = 2.03;
        let n = 80.0;
        let df = n - 1.0;

        let likelihood: Likelihood = NoncentralTLikelihood::new(t, df).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0 * n.sqrt(), (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);

        // R gives BF(H0/H1) = 1.557447, so BF(H1/H0) = 1/1.557447 = 0.6420768
        assert_relative_eq!(bf, 1.0 / 1.557447, epsilon = 0.001);
    }
}

#[cfg(test)]
mod ttest_examples {
    //! Tests from the default t-tests article (default_ttests.html)
    //!
    //! Includes one-sample, paired, and independent samples t-test examples

    use crate::prelude::*;
    use approx::assert_relative_eq;

    /// Helper function to compute Bayes factor (H1/H0)
    fn bayes_factor(likelihood: Likelihood, alt_prior: Prior, null_prior: Prior) -> f64 {
        let m1 = (likelihood * alt_prior).integral().unwrap();
        let m0 = (likelihood * null_prior).integral().unwrap();
        m1 / m0
    }

    #[test]
    fn test_one_sample_ttest_noncentral_t() {
        // From default_ttests.html:
        // t <- 2.03
        // n <- 80
        // data_model <- likelihood("noncentral_t", t = t, df = n - 1)
        // alt_prior <- prior("cauchy", location = 0, scale = 1 * sqrt(n))
        // null_prior <- prior("point", point = 0)
        // bf_onesample_1 <- integral(data_model * alt_prior) / integral(data_model * null_prior)
        // summary(bf_onesample_1)
        // #> A BF of 0.6421 indicates: Anecdotal evidence

        let t = 2.03;
        let n = 80.0;
        let df = n - 1.0;

        let likelihood: Likelihood = NoncentralTLikelihood::new(t, df).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0 * n.sqrt(), (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);

        // R gives: 0.6421
        assert_relative_eq!(bf, 0.6421, epsilon = 0.001);
    }

    #[test]
    fn test_one_sample_ttest_noncentral_d() {
        // From default_ttests.html:
        // d <- t / sqrt(n)  # = 2.03 / sqrt(80) = 0.2269588
        // data_model2 <- likelihood("noncentral_d", d = d, n = n)
        // alt_prior2 <- prior("cauchy", location = 0, scale = 1)
        // bf_onesample_2 <- integral(data_model2 * alt_prior2) / integral(data_model2 * null_prior)
        // summary(bf_onesample_2)
        // #> A BF of 0.6421 indicates: Anecdotal evidence
        //
        // This should give the same result as noncentral_t with scaled Cauchy

        let t: f64 = 2.03;
        let n: f64 = 80.0;
        let d = t / n.sqrt();

        let likelihood: Likelihood = NoncentralDLikelihood::new(d, n).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);

        // R gives: 0.6421 (same as noncentral_t approach)
        assert_relative_eq!(bf, 0.6421, epsilon = 0.001);
    }

    #[test]
    fn test_independent_samples_ttest_noncentral_d2() {
        // From default_ttests.html:
        // d = -0.6441105, n1 = 15, n2 = 16
        // data_model4 <- likelihood("noncentral_d2", d = d, n1 = n1, n2 = n2)
        // alt_prior4 <- prior("cauchy", location = 0, scale = 1)
        // bf_independent_2 <- integral(data_model4 * alt_prior4) / integral(data_model4 * null_prior)
        // summary(bf_independent_2)
        // #> A BF of 0.9709 indicates: Anecdotal evidence
        //
        // Also verified against BayesFactor::ttestBF: 0.9709424

        let d = -0.6441105;
        let n1 = 15.0;
        let n2 = 16.0;

        let likelihood: Likelihood = NoncentralD2Likelihood::new(d, n1, n2).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);

        // R gives: 0.9709424
        assert_relative_eq!(bf, 0.9709424, epsilon = 0.001);
    }

    #[test]
    fn test_independent_samples_ttest_noncentral_t() {
        // From default_ttests.html:
        // t = -1.792195, df = 29, n1 = 15, n2 = 16
        // scale = sqrt((n1 * n2) / (n1 + n2))
        // data_model3 <- likelihood("noncentral_t", t = t, df = df)
        // alt_prior3 <- prior("cauchy", location = 0, scale = 1 * sqrt((n1 * n2) / (n1 + n2)))
        // bf_independent_1 <- integral(data_model3 * alt_prior3) / integral(data_model3 * null_prior)
        // summary(bf_independent_1)
        // #> A BF of 0.9709 indicates: Anecdotal evidence

        let t: f64 = -1.792195;
        let df: f64 = 29.0;
        let n1: f64 = 15.0;
        let n2: f64 = 16.0;
        let scale = ((n1 * n2) / (n1 + n2)).sqrt();

        let likelihood: Likelihood = NoncentralTLikelihood::new(t, df).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, scale, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);

        // R gives: 0.9709 (same as noncentral_d2 approach)
        assert_relative_eq!(bf, 0.9709, epsilon = 0.001);
    }


    #[test]
    fn test_noncentral_d2_likelihood_function() {
        // Test noncentral_d2 likelihood for two-sample case

        let likelihood = NoncentralD2Likelihood::new(0.5, 10.0, 15.0);

        // Evaluate at several points
        let l_at_0 = likelihood.function(0.0).unwrap();
        let l_at_05 = likelihood.function(0.5).unwrap();

        // Likelihood should be higher near the observed d
        assert!(l_at_05 > l_at_0);
    }
}

#[cfg(test)]
mod model_construction_tests {
    //! Tests for model construction and basic operations

    use crate::prelude::*;

    #[test]
    fn test_likelihood_times_prior() {
        let likelihood = NormalLikelihood::new(0.0, 1.0);
        let prior = NormalPrior::new(0.0, 1.0, (None, None));

        let model = Likelihood::from(likelihood) * Prior::from(prior);

        assert!(model.integral().is_ok());
    }

    #[test]
    fn test_prior_times_likelihood() {
        let likelihood: Likelihood = NormalLikelihood::new(0.0, 1.0).into();
        let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();

        // Both orderings should work
        let model1 = likelihood * prior;
        let model2 = prior * likelihood;

        let integral1 = model1.integral().unwrap();
        let integral2 = model2.integral().unwrap();

        assert_eq!(integral1, integral2);
    }

    #[test]
    fn test_point_prior_returns_likelihood_at_point() {
        let likelihood = NormalLikelihood::new(0.0, 1.0);
        let prior = PointPrior::new(0.5);

        let model = Likelihood::from(likelihood) * Prior::from(prior);
        let integral = model.integral().unwrap();

        // For point prior, integral should equal likelihood at that point
        let expected = likelihood.function(0.5).unwrap();
        assert_eq!(integral, expected);
    }

    #[test]
    fn test_model_get_observation() {
        let likelihood = NormalLikelihood::new(2.5, 1.0);
        let prior = NormalPrior::new(0.0, 1.0, (None, None));

        let model = Likelihood::from(likelihood) * Prior::from(prior);

        assert_eq!(model.get_observation(), Some(2.5));
    }

    #[test]
    fn test_posterior_creation() {
        let likelihood = NormalLikelihood::new(0.0, 1.0);
        let prior = NormalPrior::new(0.0, 1.0, (None, None));

        let model = Likelihood::from(likelihood) * Prior::from(prior);
        let posterior = model.posterior();

        assert!(posterior.is_ok());
    }

    #[test]
    fn test_posterior_integrates_to_one() {
        let likelihood = NormalLikelihood::new(0.0, 1.0);
        let prior = NormalPrior::new(0.0, 1.0, (Some(-10.0), Some(10.0)));

        let model = Likelihood::from(likelihood) * Prior::from(prior);
        let posterior = model.posterior().unwrap();

        // Posterior should integrate to approximately 1
        let integral = posterior.integral().unwrap();
        assert!((integral - 1.0).abs() < 0.01);
    }
}

#[cfg(test)]
mod validation_tests {
    //! Tests for validation of likelihood and prior parameters

    use crate::prelude::*;

    #[test]
    fn test_invalid_normal_likelihood_se() {
        let likelihood: Likelihood = NormalLikelihood {
            mean: 0.0,
            se: -1.0,
        }
        .into();
        let prior = NormalPrior::new(0.0, 1.0, (None, None));

        let model = Likelihood::from(likelihood) * Prior::from(prior);
        let result = model.integral();

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_normal_prior_sd() {
        let likelihood = NormalLikelihood::new(0.0, 1.0);
        let prior = NormalPrior::new(0.0, -1.0, (None, None));

        let model = Likelihood::from(likelihood) * Prior::from(prior);
        let result = model.integral();

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_binomial_likelihood() {
        let likelihood = BinomialLikelihood {
            successes: 15.0,
            trials: 10.0,
        };
        let prior = BetaPrior::new(1.0, 1.0, (None, None));

        let model = Likelihood::from(likelihood) * Prior::from(prior);
        let result = model.integral();

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_cauchy_prior_scale() {
        let likelihood = NormalLikelihood::new(0.0, 1.0);
        let prior = CauchyPrior::new(0.0, -1.0, (None, None));

        let model = Likelihood::from(likelihood) * Prior::from(prior);
        let result = model.integral();

        assert!(result.is_err());
    }
}
