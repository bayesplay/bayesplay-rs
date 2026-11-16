//! Bayes factors checked against values from the bayesplay R package
//! documentation and R calculations. See also `bayesfactor_ttest.rs` for the
//! t-test grid checked against BayesFactor and a high-precision reference.
//!
//! These tests are derived from:
//! - https://bayesplay.github.io/bayesplay/articles/basic.html
//! - https://bayesplay.github.io/bayesplay/articles/advanced.html
//! - https://bayesplay.github.io/bayesplay/articles/default_ttests.html

mod common;

mod basic_examples {
    //! Tests from the basic usage article (basic.html)

    use approx::assert_relative_eq;
    use bayesplay::prelude::*;

    /// Helper function to compute Bayes factor (H1/H0)
    use crate::common::bayes_factor;

    #[test]
    fn test_example1_half_normal_prior() {
        // From basic.html Example 1 (Dienes & Mclatchie 2018):
        // data_mod <- likelihood(family = "normal", mean = 5.5, sd = 32.35)
        // h0_mod <- prior(family = "point", point = 0)
        // h1_mod <- prior(family = "normal", mean = 0, sd = 13.3, range = c(0, Inf))
        // bf <- m1 / m0
        // bf
        // #> 0.9745934

        let likelihood: Likelihood = NormalLikelihood::new(5.5, 32.35).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = NormalPrior::new(0.0, 13.3, (Some(0.0), None))
            .unwrap()
            .into(); // Half-normal (0 to Inf)

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

        let likelihood: Likelihood = NormalLikelihood::new(5.0, 10.0).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = UniformPrior::new(0.0, 20.0).unwrap().into();

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
        let likelihood: Likelihood = NoncentralDLikelihood::new(d, 80.0).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).unwrap().into();

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
        let likelihood: Likelihood = NoncentralDLikelihood::new(d, 80.0).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap().into();

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

        let likelihood: Likelihood = NoncentralTLikelihood::new(t, df).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0 * n.sqrt(), (None, None))
            .unwrap()
            .into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);

        // R gives BF(H0/H1) = 1.557447, so BF(H1/H0) = 1/1.557447 = 0.6420768
        assert_relative_eq!(bf, 1.0 / 1.557447, epsilon = 0.001);
    }
}

mod ttest_examples {
    //! Tests from the default t-tests article (default_ttests.html)
    //!
    //! Includes one-sample, paired, and independent samples t-test examples

    use approx::assert_relative_eq;
    use bayesplay::prelude::*;

    /// Helper function to compute Bayes factor (H1/H0)
    use crate::common::bayes_factor;

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

        let likelihood: Likelihood = NoncentralTLikelihood::new(t, df).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0 * n.sqrt(), (None, None))
            .unwrap()
            .into();

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

        let likelihood: Likelihood = NoncentralDLikelihood::new(d, n).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).unwrap().into();

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

        let likelihood: Likelihood = NoncentralD2Likelihood::new(d, n1, n2).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).unwrap().into();

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

        let likelihood: Likelihood = NoncentralTLikelihood::new(t, df).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, scale, (None, None)).unwrap().into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);

        // R gives: 0.9709 (same as noncentral_d2 approach)
        assert_relative_eq!(bf, 0.9709, epsilon = 0.001);
    }

    #[test]
    fn test_noncentral_d2_likelihood_function() {
        // Test noncentral_d2 likelihood for two-sample case

        let likelihood = NoncentralD2Likelihood::new(0.5, 10.0, 15.0).unwrap();

        // Evaluate at several points
        let l_at_0 = likelihood.function(0.0).unwrap();
        let l_at_05 = likelihood.function(0.5).unwrap();

        // Likelihood should be higher near the observed d
        assert!(l_at_05 > l_at_0);
    }
}

mod r_basic_calculations {
    //! BF computation tests matching the R package's test-basic_calculations.R.
    //!
    //! All expected values come from the R package which is the gold standard.
    //! Tolerance: epsilon = EPS (0.001) for BF comparisons.

    use approx::assert_relative_eq;
    use bayesplay::prelude::*;
    static EPS: f64 = 0.00001;

    /// Helper function to compute Bayes factor (H1/H0)
    use crate::common::bayes_factor;

    // ── Binomial tests ─────────────────────────────────────────────────

    #[test]
    fn test_binomial_beta_prior_2_5_1() {
        // R:
        // data_model <- likelihood(family = "binomial", successes = 2L, trials = 10L)
        // alt_prior <- prior(family = "beta", alpha = 1L, beta = 2.5)
        // null_prior <- prior(family = "point", point = 0.5)
        // BF01 = 0.4887695, so BF10 = 1/0.4887695
        let likelihood: Likelihood = BinomialLikelihood::new(2.0, 10.0).unwrap().into();
        let alt_prior: Prior = BetaPrior::new(1.0, 2.5, (None, None)).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.5).unwrap().into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 3.3921325, epsilon = EPS);
    }

    #[test]
    fn test_binomial_uniform_prior() {
        // R:
        //
        // data_model <- likelihood(family = "binomial", 3L, 12L)
        // alt_pripr <- prior(family = "uniform", min = 0L, max = 1L)
        // null_prior <- prior(family = "point", point = 0.5)
        // BF01 = 0.6982422, so BF10 = 1/0.6982422
        let likelihood: Likelihood = BinomialLikelihood::new(3.0, 12.0).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.5).unwrap().into();
        let alt_prior: Prior = UniformPrior::new(0.0, 1.0).unwrap().into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 1.0 / 0.6982422, epsilon = EPS);
    }

    #[test]
    fn test_binomial_beta_1_1_equals_uniform() {
        // Beta(1,1) is the same as Uniform(0,1), should give the same BF
        let likelihood: Likelihood = BinomialLikelihood::new(3.0, 12.0).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.5).unwrap().into();
        let alt_prior: Prior = BetaPrior::new(1.0, 1.0, (None, None)).unwrap().into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 1.0 / 0.6982422, epsilon = EPS);
    }

    #[test]
    fn test_binomial_beta_1_2_5() {
        // R:
        // data_model <- likelihood(family = "binomial", successes = 2L, trials = 10L)
        // alt_prior <- prior(family = "beta", alpha = 1L, beta = 2.5)
        // null_prior <- prior(family = "point", point = 0.5)
        // BF10 = 3.3921325
        let likelihood: Likelihood = BinomialLikelihood::new(2.0, 10.0).unwrap().into();
        let alt_prior: Prior = BetaPrior::new(1.0, 2.5, (None, None)).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.5).unwrap().into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 3.3921325, epsilon = EPS);
    }

    // ── Noncentral d tests ─────────────────────────────────────────────

    #[test]
    fn test_noncentral_d_cauchy_full() {
        // R:
        // data_model <- likelihood(family = "noncentral_d", d = 0.22696089971622863, n = 80L)
        // null_prior <- prior("point", 0L)
        // alt_prior <- prior("cauchy", location = 0L, scale = 1L)
        // BF10 = 0.64207642378841778275
        let likelihood: Likelihood = NoncentralDLikelihood::new(0.226_960_899_716_228_63, 80.0)
            .unwrap()
            .into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).unwrap().into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 0.642_076_423_788_417_8, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_d_half_cauchy() {
        // R:
        // data_model <- likelihood(family = "noncentral_d", d = 0.22696089971622863, n = 80L)
        // null_prior <- prior("point", 0L)
        // alt_prior <- prior("cauchy", location = 0L, scale = 1L, range = c(0L, Inf))
        // BF10 = 1.25399711580663364430
        let likelihood: Likelihood = NoncentralDLikelihood::new(0.226_960_899_716_228_63, 80.0)
            .unwrap()
            .into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (Some(0.0), None))
            .unwrap()
            .into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 1.253_997_115_806_633_6, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_d_strong_effect() {
        // R:
        // data_model <- likelihood("noncentral_d", 0.624524917476492, 51L)
        // alt_prior <- prior("cauchy", 0L, 0.707)
        // null_prior <- prior("point", 0L)
        // BF10 = 460.24969127083818420942
        let likelihood: Likelihood = NoncentralDLikelihood::new(0.624524917476492, 51.0)
            .unwrap()
            .into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 0.707, (None, None)).unwrap().into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 460.249_691_270_838_2, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_d_negative_half_cauchy() {
        // NoncentralD(-2.24, 34) + half-Cauchy(0, 0.707, [0, Inf]) vs Point(0).
        // The observed effect lies far outside the prior's interval. R bayesplay
        // (BF01 = 147.67, via its estimate_marginal heuristic) and BayesFactor
        // (BF01 = 1.51) are both wrong here; 35.134908074 is the high-precision
        // reference from tests/data/bayesfactor_ttest.R.
        let likelihood: Likelihood = NoncentralDLikelihood::new(-2.24, 34.0).unwrap().into();
        let cauchy_prior = CauchyPrior::new(0.0, 0.707, (Some(0.0), None)).unwrap();
        let alt_prior: Prior = cauchy_prior.into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();

        let m0 = (likelihood * null_prior).integral().unwrap();
        let m1 = (likelihood * alt_prior).integral().unwrap();
        let bf = m1 / m0;
        assert_relative_eq!(1.0 / bf, 35.134_908_074_210_1, max_relative = 1e-7);
    }

    #[test]
    fn test_noncentral_d_large_effect_small_n() {
        // R: NoncentralD(1.492, 10) + Cauchy(0, 1) vs Point(0)
        // BF10 = 42.4677
        let likelihood: Likelihood = NoncentralDLikelihood::new(1.492, 10.0).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).unwrap().into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 42.467_654_362_821_015, epsilon = EPS);
    }

    // ── Noncentral d2 tests ────────────────────────────────────────────

    #[test]
    fn test_noncentral_d2_two_sample() {
        // R:
        // data_model <- likelihood("noncentral_d2", -0.16866426138921944422, 17L, 18L)
        // alt_prior <- prior("cauchy", location = 0L, scale = 1L)
        // null_prior <- prior("point", 0L)
        // BF10 = 0.27411111221911949043
        let likelihood: Likelihood =
            NoncentralD2Likelihood::new(-0.168_664_261_389_219_44, 17.0, 18.0)
                .unwrap()
                .into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).unwrap().into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 0.274_111_112_219_119_5, epsilon = EPS);
    }

    // ── Noncentral t tests ─────────────────────────────────────────────

    #[test]
    fn test_noncentral_t_matching_d2() {
        // R:
        // data_model <- likelihood("noncentral_t", t = -0.49871193286623344276, df = 33L)
        // alt_prior <- prior("cauchy", location = 0L, scale = 2.95683228182748658597)
        // null_prior <- prior("point", 0L)
        let likelihood: Likelihood = NoncentralTLikelihood::new(-0.498_711_932_866_233_44, 33.0)
            .unwrap()
            .into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 2.956_832_281_827_486_6, (None, None))
            .unwrap()
            .into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 0.274_111_112_132_553_7, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_t_large_effect() {
        // R:
        //
        // data_model <- likelihood("noncentral_t", t = 4.71774600375525743345, df = 9L)
        // alt_prior <- prior("cauchy", location = 0L, scale = 3.16227766016837952279)
        // null_prior <- prior("point", 0L)
        // BF10 = 42.44813154457972359523
        let likelihood: Likelihood = NoncentralTLikelihood::new(4.717_746_003_755_257, 9.0)
            .unwrap()
            .into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 3.162_277_660_168_379_5, (None, None))
            .unwrap()
            .into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 42.448_131_544_579_724, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_t_cauchy_scaled() {
        // R:
        // data_model <- likelihood(family = "noncentral_t", t = 2.03, df = 79L)
        // alt_prior <- prior(family = "cauchy", location = 0L, scale = 8.94427190999915922021)
        // null_prior <- prior("point", 0L)
        // BF10 = 0.64207648806954098486
        let likelihood: Likelihood = NoncentralTLikelihood::new(2.03, 79.0).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 8.944_271_909_999_16, (None, None))
            .unwrap()
            .into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 0.642_076_488_069_541, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_t_df49() {
        // R: NoncentralT(4.46, 49) + Cauchy(0, sqrt(50)) vs Point(0)
        // BF10 = 403.352
        let likelihood: Likelihood = NoncentralTLikelihood::new(4.46, 49.0).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 50.0_f64.sqrt(), (None, None))
            .unwrap()
            .into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 403.352_227_790_426_4, epsilon = EPS);
    }

    #[test]
    fn test_noncentral_t_df50() {
        // R: NoncentralT(4.46, 50) + Cauchy(0, 5.049) vs Point(0)
        // BF10 = 460.250
        let likelihood: Likelihood = NoncentralTLikelihood::new(4.46, 50.0).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 5.049, (None, None)).unwrap().into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 460.249_519_258_614_84, epsilon = EPS);
    }

    // ── Student-t likelihood tests ─────────────────────────────────────

    #[test]
    fn test_student_t_likelihood_with_student_t_prior() {
        // R: StudentTLikelihood(5.47, 32.2, 119) + StudentTPrior(13.3, 4.93, 72) vs Point(0)
        // BF10 = 0.97381
        let likelihood: Likelihood = StudentTLikelihood::new(5.47, 32.2, 119.0).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = StudentTPrior::new(13.3, 4.93, 72.0, (None, None))
            .unwrap()
            .into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 0.973_812_546_820_977_6, epsilon = EPS);
    }

    // ── Normal likelihood tests ────────────────────────────────────────

    #[test]
    fn test_normal_half_normal_prior_small_effect() {
        // R: NormalLikelihood(0.63, 0.43) + half-Normal(0, 2.69, [0, Inf]) vs Point(0)
        // BF10 = 0.83252
        let likelihood: Likelihood = NormalLikelihood::new(0.63, 0.43).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = NormalPrior::new(0.0, 2.69, (Some(0.0), None))
            .unwrap()
            .into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 0.832_516_704_274_544_3, epsilon = EPS);
    }

    #[test]
    fn test_normal_with_normal_prior() {
        // R:
        // data_model <- likelihood(family = "normal", mean = 15L, sd = 13L)
        // alt_prior <- prior(family = "normal", mean = 50L, sd = 14L)
        // null_prior <- prior(family = "point", point = 0L)
        // BF10 = 0.24723869370658554945
        let likelihood: Likelihood = NormalLikelihood::new(15.0, 13.0).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();
        let alt_prior: Prior = NormalPrior::new(50.0, 14.0, (None, None)).unwrap().into();

        let bf = bayes_factor(likelihood, alt_prior, null_prior);
        assert_relative_eq!(bf, 0.247_238_693_706_585_55, epsilon = EPS);
    }
}

mod variance_model_tests {
    use approx::assert_relative_eq;

    use bayesplay::prelude::*;

    use crate::common::bayes_factor;

    #[test]
    fn variance_ratio_bayes_factors_match_r_references() {
        let references = [
            (6.0, 1.6, 0.5, 0.321_085),
            (6.0, 1.6, 4.5, 0.776_653),
            (29.0, 1.25, 0.5, 0.152_961),
            (29.0, 1.25, 2.16, 0.357_114),
            (99.0, 0.8, 0.5, 0.177_146),
            (9.0, 3.0, 4.5, 11.854_365),
        ];

        for (df, u, alpha, expected) in references {
            let likelihood: Likelihood = VarianceRatioLikelihood::new(u, df).unwrap().into();
            let alternative: Prior = BetaPrimePrior::new(alpha, (None, None)).unwrap().into();
            let null: Prior = PointPrior::new(1.0).unwrap().into();
            assert_relative_eq!(
                bayes_factor(likelihood, alternative, null),
                expected,
                epsilon = 1e-5,
                max_relative = 1e-5
            );
        }
    }

    #[test]
    fn directional_variance_ratio_bayes_factors_mix_to_the_two_sided_result() {
        let likelihood: Likelihood = VarianceRatioLikelihood::new(1.9575, 24.0).unwrap().into();
        let null: Prior = PointPrior::new(1.0).unwrap().into();
        let two_sided = bayes_factor(
            likelihood,
            BetaPrimePrior::new(0.5, (None, None)).unwrap().into(),
            null,
        );
        let above = bayes_factor(
            likelihood,
            BetaPrimePrior::new(0.5, (Some(1.0), None)).unwrap().into(),
            null,
        );
        let below = bayes_factor(
            likelihood,
            BetaPrimePrior::new(0.5, (Some(0.0), Some(1.0)))
                .unwrap()
                .into(),
            null,
        );

        assert_relative_eq!(two_sided, 0.5 * above + 0.5 * below, epsilon = 1e-8);
        assert!(above > two_sided);
        assert!(below < two_sided);
    }
}
