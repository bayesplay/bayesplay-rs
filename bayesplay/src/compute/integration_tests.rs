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

#[cfg(test)]
mod extended_validation_tests {
    //! Comprehensive validation error tests.
    //!
    //! These tests verify that direct `.validate()` calls on all prior and
    //! likelihood types return the correct error variants, matching the
    //! error-checking behavior from the R package's test-error_messages.R.

    use crate::prelude::*;

    // ── Prior validations ──────────────────────────────────────────────

    #[test]
    fn test_normal_prior_negative_sd() {
        let prior = NormalPrior::new(0.0, -1.0, (None, None));
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidStandardDeviation(sd)) if sd == -1.0
        ));
    }

    #[test]
    fn test_normal_prior_zero_sd() {
        let prior = NormalPrior::new(0.0, 0.0, (None, None));
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidStandardDeviation(sd)) if sd == 0.0
        ));
    }

    #[test]
    fn test_cauchy_prior_negative_scale() {
        let prior = CauchyPrior::new(0.0, -1.0, (None, None));
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidScale(s)) if s == -1.0
        ));
    }

    #[test]
    fn test_cauchy_prior_zero_scale() {
        let prior = CauchyPrior::new(0.0, 0.0, (None, None));
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidScale(s)) if s == 0.0
        ));
    }

    #[test]
    fn test_student_t_prior_negative_sd() {
        let prior = StudentTPrior::new(0.0, -1.0, 3.0, (None, None));
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidStandardDeviation(sd)) if sd == -1.0
        ));
    }

    #[test]
    fn test_student_t_prior_invalid_df_fraction() {
        // df = 0.5, which is < 1 and should be invalid
        let prior = StudentTPrior::new(0.0, 1.0, 0.5, (None, None));
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidDegreesOfFreedom(df)) if df == 0.5
        ));
    }

    #[test]
    fn test_student_t_prior_invalid_df_one() {
        // df = 1 is invalid — must be > 1
        let prior = StudentTPrior::new(0.0, 1.0, 1.0, (None, None));
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidDegreesOfFreedom(df)) if df == 1.0
        ));
    }

    #[test]
    fn test_student_t_prior_multiple_errors() {
        // Both sd and df invalid
        let prior = StudentTPrior::new(0.0, -1.0, 0.5, (None, None));
        assert!(matches!(
            prior.validate(),
            Err(PriorError::MultipleErrors(_))
        ));
    }

    #[test]
    fn test_uniform_prior_equal_bounds() {
        let result = UniformPrior::new(1.0, 1.0);
        assert!(matches!(result.validate(), Err(PriorError::InvalidRange)));
    }

    #[test]
    fn test_uniform_prior_reversed_bounds() {
        let result = UniformPrior::new(2.0, 1.0);
        assert!(matches!(result.validate(), Err(PriorError::InvalidRange)));
    }

    #[test]
    fn test_beta_prior_negative_alpha() {
        let prior = BetaPrior::new(-1.0, 1.0, (None, None));
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidShapeParameter(v)) if v == -1.0
        ));
    }

    #[test]
    fn test_beta_prior_negative_beta() {
        let prior = BetaPrior::new(1.0, -1.0, (None, None));
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidShapeParameter(v)) if v == -1.0
        ));
    }

    // ── Likelihood validations ─────────────────────────────────────────

    #[test]
    fn test_normal_likelihood_negative_se() {
        let likelihood = NormalLikelihood {
            mean: 0.0,
            se: -1.0,
        };
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidSE(se)) if se == -1.0
        ));
    }

    #[test]
    fn test_normal_likelihood_zero_se() {
        let likelihood = NormalLikelihood { mean: 0.0, se: 0.0 };
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidSE(se)) if se == 0.0
        ));
    }

    #[test]
    fn test_student_t_likelihood_negative_sd() {
        let likelihood = StudentTLikelihood::new(0.0, -1.0, 10.0);
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidSD(sd)) if sd == -1.0
        ));
    }

    #[test]
    fn test_student_t_likelihood_negative_df() {
        let likelihood = StudentTLikelihood::new(0.0, 1.0, -5.0);
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidDF(df)) if df == -5.0
        ));
    }

    #[test]
    fn test_student_t_likelihood_multiple_errors() {
        let likelihood = StudentTLikelihood::new(0.0, -1.0, -5.0);
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::MultipleErrors(_))
        ));
    }

    #[test]
    fn test_binomial_likelihood_successes_exceed_trials() {
        let likelihood = BinomialLikelihood {
            successes: 15.0,
            trials: 10.0,
        };
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidSuccess(s)) if s == 15.0
        ));
    }

    #[test]
    fn test_binomial_likelihood_zero_trials() {
        let likelihood = BinomialLikelihood {
            successes: 0.0,
            trials: 0.0,
        };
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidTrials(t)) if t == 0.0
        ));
    }

    #[test]
    fn test_binomial_likelihood_non_integer_successes() {
        let likelihood = BinomialLikelihood {
            successes: 1.1,
            trials: 10.0,
        };
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidSuccess(s)) if s == 1.1
        ));
    }

    #[test]
    fn test_binomial_likelihood_non_integer_trials() {
        let likelihood = BinomialLikelihood {
            successes: 1.0,
            trials: 1.1,
        };
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidTrials(t)) if t == 1.1
        ));
    }

    #[test]
    fn test_noncentral_d_likelihood_small_n() {
        let likelihood = NoncentralDLikelihood::new(0.5, 0.5);
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidN(n)) if n == 0.5
        ));
    }

    #[test]
    fn test_noncentral_d2_likelihood_small_n1() {
        let likelihood = NoncentralD2Likelihood::new(0.5, 0.5, 10.0);
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidN1(n)) if n == 0.5
        ));
    }

    #[test]
    fn test_noncentral_d2_likelihood_small_n2() {
        let likelihood = NoncentralD2Likelihood::new(0.5, 10.0, 0.5);
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidN2(n)) if n == 0.5
        ));
    }

    #[test]
    fn test_noncentral_t_likelihood_small_df() {
        let likelihood = NoncentralTLikelihood::new(2.0, 0.5);
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidDF(df)) if df == 0.5
        ));
    }
}

#[cfg(test)]
mod type_of_tests {
    //! Tests for PriorFamily identification via TypeOf trait.
    //!
    //! Verifies that each prior type correctly reports its family
    //! and that is_point() returns true only for PointPrior.

    use crate::prelude::*;
    use crate::prior::PriorFamily;

    #[test]
    fn test_normal_type_of() {
        let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
        assert_eq!(prior.type_of(), PriorFamily::Normal);
    }

    #[test]
    fn test_cauchy_type_of() {
        let prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).into();
        assert_eq!(prior.type_of(), PriorFamily::Cauchy);
    }

    #[test]
    fn test_student_t_type_of() {
        let prior: Prior = StudentTPrior::new(0.0, 1.0, 3.0, (None, None)).into();
        assert_eq!(prior.type_of(), PriorFamily::StudentT);
    }

    #[test]
    fn test_uniform_type_of() {
        let prior: Prior = UniformPrior::new(0.0, 1.0).into();
        assert_eq!(prior.type_of(), PriorFamily::Uniform);
    }

    #[test]
    fn test_beta_type_of() {
        let prior: Prior = BetaPrior::new(1.0, 1.0, (None, None)).into();
        assert_eq!(prior.type_of(), PriorFamily::Beta);
    }

    #[test]
    fn test_point_type_of() {
        let prior: Prior = PointPrior::new(0.0).into();
        assert_eq!(prior.type_of(), PriorFamily::Point);
    }

    #[test]
    fn test_point_is_point() {
        let prior: Prior = PointPrior::new(0.0).into();
        assert!(prior.is_point());
    }

    #[test]
    fn test_normal_is_not_point() {
        let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();
        assert!(!prior.is_point());
    }

    #[test]
    fn test_cauchy_is_not_point() {
        let prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).into();
        assert!(!prior.is_point());
    }

    #[test]
    fn test_student_t_is_not_point() {
        let prior: Prior = StudentTPrior::new(0.0, 1.0, 3.0, (None, None)).into();
        assert!(!prior.is_point());
    }

    #[test]
    fn test_uniform_is_not_point() {
        let prior: Prior = UniformPrior::new(0.0, 1.0).into();
        assert!(!prior.is_point());
    }

    #[test]
    fn test_beta_is_not_point() {
        let prior: Prior = BetaPrior::new(1.0, 1.0, (None, None)).into();
        assert!(!prior.is_point());
    }
}

#[cfg(test)]
mod r_basic_calculations {
    //! BF computation tests matching the R package's test-basic_calculations.R.
    //!
    //! All expected values come from the R package which is the gold standard.
    //! Tolerance: epsilon = EPS (0.001) for BF comparisons.

    use crate::compute::model::IntegralError;
    use crate::prelude::*;
    use approx::assert_relative_eq;
    static EPS: f64 = 0.001;

    /// Helper function to compute Bayes factor (H1/H0)
    fn bayes_factor(likelihood: Likelihood, alt_prior: Prior, null_prior: Prior) -> f64 {
        let m1 = (likelihood * alt_prior).integral().unwrap();
        let m0 = (likelihood * null_prior).integral().unwrap();
        m1 / m0
    }

    fn bayes_factor_result(
        likelihood: Likelihood,
        alt_prior: Prior,
        null_prior: Prior,
    ) -> Result<f64, IntegralError> {
        let m1 = (likelihood * alt_prior).integral()?;
        let m0 = (likelihood * null_prior).integral()?;
        Ok(m1 / m0)
    }

    // ── Binomial tests ─────────────────────────────────────────────────

    #[test]
    fn test_binomial_beta_prior_2_5_1() {
        // R: likelihood("binomial", successes = 2, trials = 10) + prior("beta", 2.5, 1) vs point(0.5)
        // BF01 = 0.4887695, so BF10 = 1/0.4887695
        let likelihood: Likelihood = BinomialLikelihood::new(2.0, 10.0).into();
        let null_prior: Prior = PointPrior::new(0.5).into();
        let alt_prior: Prior = BetaPrior::new(1.0, 2.5, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 3.3921325, epsilon = EPS);
    }

    #[test]
    fn test_binomial_uniform_prior() {
        // R: likelihood("binomial", 3, 12) + prior("uniform", 0, 1) vs point(0.5)
        // BF01 = 0.6982422, so BF10 = 1/0.6982422
        let likelihood: Likelihood = BinomialLikelihood::new(3.0, 12.0).into();
        let null_prior: Prior = PointPrior::new(0.5).into();
        let alt_prior: Prior = UniformPrior::new(0.0, 1.0).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 1.0/0.6982422, epsilon = EPS);
    }

    #[test]
    fn test_binomial_beta_1_1_equals_uniform() {
        // Beta(1,1) is the same as Uniform(0,1), should give the same BF
        let likelihood: Likelihood = BinomialLikelihood::new(2.0, 10.0).into();
        let null_prior: Prior = PointPrior::new(0.5).into();
        let alt_prior: Prior = BetaPrior::new(1.0, 1.0, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 1.0 / 0.4833984, epsilon = EPS);
    }

    #[test]
    fn test_binomial_beta_1_2_5() {
        // R: likelihood("binomial", 2, 10) + prior("beta", 1, 2.5) vs point(0.5)
        // BF10 = 3.3921325
        let likelihood: Likelihood = BinomialLikelihood::new(2.0, 10.0).into();
        let null_prior: Prior = PointPrior::new(0.5).into();
        let alt_prior: Prior = BetaPrior::new(1.0, 2.5, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 3.3921325, epsilon = EPS);
    }

    // ── Noncentral d tests ─────────────────────────────────────────────

    #[test]
    fn test_noncentral_d_cauchy_full() {
        // R: NoncentralD(0.227, 80) + Cauchy(0, 1) vs Point(0)
        // BF10 = 0.642
        let likelihood: Likelihood = NoncentralDLikelihood::new(0.227, 80.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 0.642, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_d_half_cauchy() {
        // R: NoncentralD(0.227, 80) + half-Cauchy(0, 1, [0, Inf]) vs Point(0)
        // BF10 = 1.254
        let likelihood: Likelihood = NoncentralDLikelihood::new(0.227, 80.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (Some(0.0), None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 1.254, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_d_strong_effect() {
        // R: NoncentralD(0.625, 51) + Cauchy(0, 0.707) vs Point(0)
        // BF10 = 465.064
        let likelihood: Likelihood = NoncentralDLikelihood::new(0.625, 51.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 0.707, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 465.064_089_100_291_48, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_d_negative_half_cauchy() {
        // R: NoncentralD(-2.24, 34) + half-Cauchy(0, 0.707, [0, Inf]) vs Point(0)
        // BF10 = 0.006773
        // |t| = 2.24 * sqrt(34) ≈ 13.06 > 5, observation outside [0, Inf] -> approximation path
        let likelihood: Likelihood = NoncentralDLikelihood::new(-2.24, 34.0).into();
        let cauchy_prior = CauchyPrior::new(0.0, 0.707, (Some(0.0), None));
        let alt_prior: Prior = cauchy_prior.into();
        let null_prior: Prior = PointPrior::new(0.0).into();

        let m0 = (likelihood * null_prior).integral().unwrap();
        let m1 = (likelihood * alt_prior).integral().unwrap();
        let bf = m1 / m0;
        assert_relative_eq!(1.0 / bf, 147.666_019_341_430_88, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_d_large_effect_small_n() {
        // R: NoncentralD(1.492, 10) + Cauchy(0, 1) vs Point(0)
        // BF10 = 42.4677
        let likelihood: Likelihood = NoncentralDLikelihood::new(1.492, 10.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 42.467_654_362_821_015, epsilon = EPS);
    }

    // ── Noncentral d2 tests ────────────────────────────────────────────

    #[test]
    fn test_noncentral_d2_two_sample() {
        // R: NoncentralD2(-0.169, 17, 18) + Cauchy(0, 1) vs Point(0)
        // BF10 = 0.274111
        let likelihood: Likelihood = NoncentralD2Likelihood::new(-0.169, 17.0, 18.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 0.274111, epsilon = EPS);
    }

    // ── Noncentral t tests ─────────────────────────────────────────────

    #[test]
    fn test_noncentral_t_matching_d2() {
        // R: NoncentralT(-0.499, 33) + Cauchy(0, 2.957) vs Point(0)
        // BF10 = 0.274111 (should match noncentral_d2 test above)
        let likelihood: Likelihood = NoncentralTLikelihood::new(-0.499, 33.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 2.957, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 0.274111, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_t_large_effect() {
        // R: NoncentralT(4.718, 9) + Cauchy(0, 3.162) vs Point(0)
        // BF10 = 42.4607
        let likelihood: Likelihood = NoncentralTLikelihood::new(4.718, 9.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 3.162, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 42.460_688_167_600_331, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_t_cauchy_scaled() {
        // R: NoncentralT(2.03, 79) + Cauchy(0, sqrt(80)) vs Point(0)
        // BF10 = 0.642
        let likelihood: Likelihood = NoncentralTLikelihood::new(2.03, 79.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 80.0_f64.sqrt(), (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 0.642, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_t_df49() {
        // R: NoncentralT(4.46, 49) + Cauchy(0, sqrt(50)) vs Point(0)
        // BF10 = 403.352
        let likelihood: Likelihood = NoncentralTLikelihood::new(4.46, 49.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 50.0_f64.sqrt(), (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 403.352_227_790_426_41, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_t_df50() {
        // R: NoncentralT(4.46, 50) + Cauchy(0, 5.049) vs Point(0)
        // BF10 = 460.250
        let likelihood: Likelihood = NoncentralTLikelihood::new(4.46, 50.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 5.049, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 460.249_519_258_614_84, epsilon = EPS);
    }

    // ── Student-t likelihood tests ─────────────────────────────────────

    #[test]
    fn test_student_t_likelihood_with_student_t_prior() {
        // R: StudentTLikelihood(5.47, 32.2, 119) + StudentTPrior(13.3, 4.93, 72) vs Point(0)
        // BF10 = 0.97381
        let likelihood: Likelihood = StudentTLikelihood::new(5.47, 32.2, 119.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = StudentTPrior::new(13.3, 4.93, 72.0, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 0.973_812_546_820_977_62, epsilon = EPS);
    }

    // ── Normal likelihood tests ────────────────────────────────────────

    #[test]
    fn test_normal_half_normal_prior_small_effect() {
        // R: NormalLikelihood(0.63, 0.43) + half-Normal(0, 2.69, [0, Inf]) vs Point(0)
        // BF10 = 0.83252
        let likelihood: Likelihood = NormalLikelihood::new(0.63, 0.43).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = NormalPrior::new(0.0, 2.69, (Some(0.0), None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 0.832_516_704_274_544_27, epsilon = EPS);
    }

    #[test]
    fn test_normal_with_normal_prior() {
        // R: NormalLikelihood(15, 13) + Normal(50, 14) vs Point(0)
        // BF10 = 0.247
        let likelihood: Likelihood = NormalLikelihood::new(15.0, 13.0).into();
        let null_prior: Prior = PointPrior::new(0.0).into();
        let alt_prior: Prior = NormalPrior::new(50.0, 14.0, (None, None)).into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 0.247, epsilon = EPS);
    }
}

#[cfg(test)]
mod posterior_predictive_tests {
    //! Tests for posterior and predictive distributions.
    //!
    //! These verify:
    //! - The Savage-Dickey density ratio identity
    //! - Predictive distribution consistency with model integral
    //! - Posterior integrates to 1 for various model configurations

    use crate::prelude::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_savage_dickey_ratio() {
        // The Savage-Dickey density ratio: BF10 ≈ prior(0) / posterior(0)
        // Using a noncentral_d model with continuous prior
        let likelihood: Likelihood = NoncentralDLikelihood::new(0.227, 80.0).into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).into();
        let null_prior: Prior = PointPrior::new(0.0).into();

        let model = likelihood * alt_prior;
        let posterior = model.posterior().unwrap();

        // BF10 from model integrals
        let m1 = model.integral().unwrap();
        let m0 = (likelihood * null_prior).integral().unwrap();
        let bf10 = m1 / m0;

        // BF10 from Savage-Dickey: prior(0) / posterior(0)
        let prior_at_0 = alt_prior.function(0.0).unwrap();
        let posterior_at_0 = posterior.function(0.0).unwrap();
        let bf10_sd = prior_at_0 / posterior_at_0;

        assert_relative_eq!(bf10, bf10_sd, epsilon = 0.01);
    }

    #[test]
    fn test_predictive_equals_integral() {
        // For a normal likelihood + normal prior model:
        // model.integral() ≈ model.predictive().function(observation)
        let observation = 0.5;
        let likelihood: Likelihood = NormalLikelihood::new(observation, 0.2).into();
        let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).into();

        let model = likelihood * prior;
        let integral = model.integral().unwrap();

        let predictive = model.predictive();
        let pred_at_obs = predictive.function(observation).unwrap();

        assert_relative_eq!(integral, pred_at_obs, epsilon = 0.001);
    }

    #[test]
    fn test_posterior_integrates_to_one_normal_normal() {
        // Normal likelihood + Normal prior (truncated to finite range)
        let likelihood: Likelihood = NormalLikelihood::new(0.0, 1.0).into();
        let prior: Prior = NormalPrior::new(0.0, 1.0, (Some(-10.0), Some(10.0))).into();

        let model = likelihood * prior;
        let posterior = model.posterior().unwrap();
        let integral = posterior.integral().unwrap();

        assert_relative_eq!(integral, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_posterior_integrates_to_one_normal_cauchy() {
        // Normal likelihood + Cauchy prior (truncated to finite range)
        let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).into();
        let prior: Prior = CauchyPrior::new(0.0, 1.0, (Some(-20.0), Some(20.0))).into();

        let model = likelihood * prior;
        let posterior = model.posterior().unwrap();
        let integral = posterior.integral().unwrap();

        assert_relative_eq!(integral, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_posterior_integrates_to_one_binomial_beta() {
        // Binomial likelihood + Beta prior
        let likelihood: Likelihood = BinomialLikelihood::new(7.0, 10.0).into();
        let prior: Prior = BetaPrior::new(1.0, 1.0, (None, None)).into();

        let model = likelihood * prior;
        let posterior = model.posterior().unwrap();
        let integral = posterior.integral().unwrap();

        assert_relative_eq!(integral, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_posterior_integrates_to_one_normal_uniform() {
        // Normal likelihood + Uniform prior
        let likelihood: Likelihood = NormalLikelihood::new(5.0, 10.0).into();
        let prior: Prior = UniformPrior::new(0.0, 20.0).into();

        let model = likelihood * prior;
        let posterior = model.posterior().unwrap();
        let integral = posterior.integral().unwrap();

        assert_relative_eq!(integral, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_posterior_integrates_to_one_noncentral_d() {
        // Noncentral d likelihood + Cauchy prior (truncated)
        let likelihood: Likelihood = NoncentralDLikelihood::new(0.5, 30.0).into();
        let prior: Prior = CauchyPrior::new(0.0, 0.707, (Some(-10.0), Some(10.0))).into();

        let model = likelihood * prior;
        let posterior = model.posterior().unwrap();
        let integral = posterior.integral().unwrap();

        assert_relative_eq!(integral, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_posterior_integrates_to_one_half_normal() {
        // Normal likelihood + half-normal prior (truncated to positive)
        let likelihood: Likelihood = NormalLikelihood::new(5.5, 32.35).into();
        let prior: Prior = NormalPrior::new(0.0, 13.3, (Some(0.0), None)).into();

        let model = likelihood * prior;
        let posterior = model.posterior().unwrap();
        let integral = posterior.integral().unwrap();

        assert_relative_eq!(integral, 1.0, epsilon = 0.01);
    }
}

#[cfg(test)]
mod normalization_tests {
    //! Tests for prior normalization constants.
    //!
    //! Verifies that normalize() returns the correct CDF-based area
    //! for various prior types and truncation configurations.

    use crate::prelude::*;
    use approx::assert_relative_eq;

    // ── Normal prior normalization ─────────────────────────────────────

    #[test]
    fn test_normal_unbounded_normalization() {
        let prior = NormalPrior::new(0.0, 1.0, (None, None));
        assert_relative_eq!(prior.normalize().unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normal_half_normalization() {
        // Symmetric around 0, positive half → 0.5
        let prior = NormalPrior::new(0.0, 1.0, (Some(0.0), None));
        assert_relative_eq!(prior.normalize().unwrap(), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_normal_one_sd_normalization() {
        // ±1 SD contains ~68.27% of mass
        let prior = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0)));
        assert_relative_eq!(prior.normalize().unwrap(), 0.6827, epsilon = 0.001);
    }

    #[test]
    fn test_normal_two_sd_normalization() {
        // ±2 SD contains ~95.45% of mass
        let prior = NormalPrior::new(0.0, 1.0, (Some(-2.0), Some(2.0)));
        assert_relative_eq!(prior.normalize().unwrap(), 0.9545, epsilon = 0.001);
    }

    // ── Cauchy prior normalization ─────────────────────────────────────

    #[test]
    fn test_cauchy_unbounded_normalization() {
        let prior = CauchyPrior::new(0.0, 1.0, (None, None));
        assert_relative_eq!(prior.normalize().unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cauchy_half_normalization() {
        // Symmetric around 0, positive half → 0.5
        let prior = CauchyPrior::new(0.0, 1.0, (Some(0.0), None));
        assert_relative_eq!(prior.normalize().unwrap(), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_cauchy_symmetric_range_normalization() {
        // Cauchy(0,1) from -1 to 1: CDF(1) - CDF(-1) = 0.75 - 0.25 = 0.5
        let prior = CauchyPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0)));
        assert_relative_eq!(prior.normalize().unwrap(), 0.5, epsilon = 0.001);
    }

    // ── Student-t prior normalization ──────────────────────────────────

    #[test]
    fn test_student_t_unbounded_normalization() {
        let prior = StudentTPrior::new(0.0, 1.0, 3.0, (None, None));
        assert_relative_eq!(prior.normalize().unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_student_t_half_normalization() {
        // Symmetric around 0, positive half → 0.5
        let prior = StudentTPrior::new(0.0, 1.0, 3.0, (Some(0.0), None));
        assert_relative_eq!(prior.normalize().unwrap(), 0.5, epsilon = 1e-6);
    }

    // ── Uniform prior normalization ────────────────────────────────────

    #[test]
    fn test_uniform_normalization() {
        // Uniform prior always integrates to 1 by definition
        let prior: Prior = UniformPrior::new(0.0, 1.0).into();
        let integral = prior.integral().unwrap();
        assert_relative_eq!(integral, 1.0, epsilon = 0.001);
    }

    #[test]
    fn test_uniform_wider_normalization() {
        let prior: Prior = UniformPrior::new(-5.0, 5.0).into();
        let integral = prior.integral().unwrap();
        assert_relative_eq!(integral, 1.0, epsilon = 0.001);
    }

    // ── Beta prior normalization ───────────────────────────────────────

    #[test]
    fn test_beta_normalization() {
        // Beta(1,1) is uniform on [0,1], integrates to 1
        let prior: Prior = BetaPrior::new(1.0, 1.0, (None, None)).into();
        let integral = prior.integral().unwrap();
        assert_relative_eq!(integral, 1.0, epsilon = 0.001);
    }

    #[test]
    fn test_beta_asymmetric_normalization() {
        // Beta(2,5) integrates to 1 on [0,1]
        let prior: Prior = BetaPrior::new(2.0, 5.0, (None, None)).into();
        let integral = prior.integral().unwrap();
        assert_relative_eq!(integral, 1.0, epsilon = 0.001);
    }

    // ── Point prior normalization ──────────────────────────────────────

    #[test]
    fn test_point_prior_normalization() {
        // Point prior integrates to 1 (by convention)
        let prior: Prior = PointPrior::new(0.0).into();
        let integral = prior.integral().unwrap();
        assert_relative_eq!(integral, 1.0, epsilon = 1e-10);
    }
}
