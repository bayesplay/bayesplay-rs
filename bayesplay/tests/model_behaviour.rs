//! Behaviour of models, posteriors and predictives: construction, normalization,
//! identities that must hold (Savage-Dickey, predictive = marginal, ...).

mod model_construction_tests {
    //! Tests for model construction and basic operations

    use approx::assert_relative_eq;
    use bayesplay::prelude::*;

    #[test]
    fn test_likelihood_times_prior() {
        let likelihood = NormalLikelihood::new(0.0, 1.0).unwrap();
        let prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();

        let model = Likelihood::from(likelihood) * Prior::from(prior);

        assert!(model.integral().is_ok());
    }

    #[test]
    fn test_prior_times_likelihood() {
        let likelihood: Likelihood = NormalLikelihood::new(0.0, 1.0).unwrap().into();
        let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap().into();

        // Both orderings should work
        let model1 = likelihood * prior;
        let model2 = prior * likelihood;

        let integral1 = model1.integral().unwrap();
        let integral2 = model2.integral().unwrap();

        assert_eq!(integral1, integral2);
    }

    #[test]
    fn test_point_prior_returns_likelihood_at_point() {
        let likelihood = NormalLikelihood::new(0.0, 1.0).unwrap();
        let prior = PointPrior::new(0.5).unwrap();

        let model = Likelihood::from(likelihood) * Prior::from(prior);
        let integral = model.integral().unwrap();

        // For point prior, integral should equal likelihood at that point
        let expected = likelihood.function(0.5).unwrap();
        assert_eq!(integral, expected);
    }

    #[test]
    fn test_model_get_observation() {
        let likelihood = NormalLikelihood::new(2.5, 1.0).unwrap();
        let prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();

        let model = Likelihood::from(likelihood) * Prior::from(prior);

        assert_eq!(model.get_observation(), Some(2.5));
    }

    #[test]
    fn test_posterior_creation() {
        let likelihood = NormalLikelihood::new(0.0, 1.0).unwrap();
        let prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();

        let model = Likelihood::from(likelihood) * Prior::from(prior);
        let posterior = model.posterior();

        assert!(posterior.is_ok());
    }

    #[test]
    fn test_posterior_integrates_to_one() {
        let likelihood = NormalLikelihood::new(0.0, 1.0).unwrap();
        let prior = NormalPrior::new(0.0, 1.0, (Some(-10.0), Some(10.0))).unwrap();

        let model = Likelihood::from(likelihood) * Prior::from(prior);
        let posterior = model.posterior().unwrap();

        // Posterior should integrate to approximately 1
        let integral = posterior.integral().unwrap();
        assert!((integral - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_large_unbounded_cauchy_matches_direct_quadrature() {
        let likelihood: Likelihood = NoncentralDLikelihood::new(2.0, 100.0).unwrap().into();
        let prior: Prior = CauchyPrior::new(0.0, 0.707, (None, None)).unwrap().into();
        let model = likelihood * prior;

        let marginal = model.integral().unwrap();

        let exact = rmath::Quadrature {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
            points: vec![2.0],
            ..Default::default()
        }
        .integrate(move |x| model.function(x).unwrap())
        .unwrap()
        .value;

        assert!(marginal.value.is_finite());
        assert!(marginal.value > 0.0);
        assert_relative_eq!(marginal.value, exact, epsilon = 1e-12);
    }

    #[test]
    fn test_large_noncentral_t_matches_effect_size_parameterization() {
        // With ncp = delta * sqrt(n), a Cauchy(0, r * sqrt(n)) prior on ncp is
        // the same model as Cauchy(0, r) on delta, so the marginals agree.
        let n: f64 = 34.0;
        for (t, range) in [
            (16.0, (None, None)),
            (16.0, (Some(0.0), None)),
            (-25.0, (Some(0.0), None)),
            (20.0, (Some(-1.0), Some(1.0))),
        ] {
            let scale = |x: Option<f64>| x.map(|x: f64| x * n.sqrt());
            let t_model = Likelihood::from(NoncentralTLikelihood::new(t, n - 1.0).unwrap())
                * Prior::from(
                    CauchyPrior::new(0.0, 0.707 * n.sqrt(), (scale(range.0), scale(range.1)))
                        .unwrap(),
                );
            let d_model = Likelihood::from(NoncentralDLikelihood::new(t / n.sqrt(), n).unwrap())
                * Prior::from(CauchyPrior::new(0.0, 0.707, range).unwrap());
            assert_relative_eq!(
                t_model.integral().unwrap().value,
                d_model.integral().unwrap().value,
                max_relative = 1e-7
            );
        }
    }

    #[test]
    fn test_narrow_truncated_interval_far_from_observation() {
        let n: f64 = 1000.0;
        let likelihood: Likelihood = NoncentralDLikelihood::new(6.0 / n.sqrt(), n)
            .unwrap()
            .into();
        let prior: Prior = CauchyPrior::new(0.05, 1.0, (Some(0.0), Some(0.15)))
            .unwrap()
            .into();
        let model = likelihood * prior;

        let marginal = model.integral().unwrap().value;

        let exact = rmath::integrate(move |x| model.function(x).unwrap(), 0.0, 0.15)
            .unwrap()
            .value;

        assert_relative_eq!(marginal, exact, max_relative = 1e-8);
    }
}

mod posterior_predictive_tests {
    //! Tests for posterior and predictive distributions.
    //!
    //! These verify:
    //! - The Savage-Dickey density ratio identity
    //! - Predictive distribution consistency with model integral
    //! - Posterior integrates to 1 for various model configurations

    use approx::assert_relative_eq;
    use bayesplay::prelude::*;
    static EPS: f64 = 0.000001;

    #[test]
    fn test_savage_dickey_ratio() {
        // The Savage-Dickey density ratio: BF10 ≈ prior(0) / posterior(0)
        // Using a noncentral_d model with continuous prior
        let likelihood: Likelihood = NoncentralDLikelihood::new(0.227, 80.0).unwrap().into();
        let alt_prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).unwrap().into();
        let null_prior: Prior = PointPrior::new(0.0).unwrap().into();

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

        assert_relative_eq!(bf10, bf10_sd, epsilon = EPS);
    }

    #[test]
    fn test_predictive_equals_integral() {
        // For a normal likelihood + normal prior model:
        // model.integral() ≈ model.predictive().function(observation)
        let observation = 0.5;
        let likelihood: Likelihood = NormalLikelihood::new(observation, 0.2).unwrap().into();
        let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap().into();

        let model = likelihood * prior;
        let integral = model.integral().unwrap();

        let predictive = model.predictive();
        let pred_at_obs = predictive.function(observation).unwrap();

        assert_relative_eq!(integral, pred_at_obs, epsilon = 0.001);
    }

    #[test]
    fn test_posterior_integrates_to_one_normal_normal() {
        // Normal likelihood + Normal prior (truncated to finite range)
        let likelihood: Likelihood = NormalLikelihood::new(0.0, 1.0).unwrap().into();
        let prior: Prior = NormalPrior::new(0.0, 1.0, (Some(-10.0), Some(10.0)))
            .unwrap()
            .into();

        let model = likelihood * prior;
        let posterior = model.posterior().unwrap();
        let integral = posterior.integral().unwrap();

        assert_relative_eq!(integral, 1.0, epsilon = EPS);
    }

    #[test]
    fn test_posterior_integrates_to_one_normal_cauchy() {
        // Normal likelihood + Cauchy prior (truncated to finite range)
        let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).unwrap().into();
        let prior: Prior = CauchyPrior::new(0.0, 1.0, (Some(-20.0), Some(20.0)))
            .unwrap()
            .into();

        let model = likelihood * prior;
        let posterior = model.posterior().unwrap();
        let integral = posterior.integral().unwrap();

        assert_relative_eq!(integral, 1.0, epsilon = EPS);
    }

    #[test]
    fn test_posterior_integrates_to_one_binomial_beta() {
        // Binomial likelihood + Beta prior
        let likelihood: Likelihood = BinomialLikelihood::new(7.0, 10.0).unwrap().into();
        let prior: Prior = BetaPrior::new(1.0, 1.0, (None, None)).unwrap().into();

        let model = likelihood * prior;
        let posterior = model.posterior().unwrap();
        let integral = posterior.integral().unwrap();

        assert_relative_eq!(integral, 1.0, epsilon = EPS);
    }

    #[test]
    fn test_posterior_integrates_to_one_normal_uniform() {
        // Normal likelihood + Uniform prior
        let likelihood: Likelihood = NormalLikelihood::new(5.0, 10.0).unwrap().into();
        let prior: Prior = UniformPrior::new(0.0, 20.0).unwrap().into();

        let model = likelihood * prior;
        let posterior = model.posterior().unwrap();
        let integral = posterior.integral().unwrap();

        assert_relative_eq!(integral, 1.0, epsilon = EPS);
    }

    #[test]
    fn test_posterior_integrates_to_one_noncentral_d() {
        // Noncentral d likelihood + Cauchy prior (truncated)
        let likelihood: Likelihood = NoncentralDLikelihood::new(0.5, 30.0).unwrap().into();
        let prior: Prior = CauchyPrior::new(0.0, 0.707, (Some(-10.0), Some(10.0)))
            .unwrap()
            .into();

        let model = likelihood * prior;
        let posterior = model.posterior().unwrap();
        let integral = posterior.integral().unwrap();

        assert_relative_eq!(integral, 1.0, epsilon = EPS);
    }

    #[test]
    fn test_posterior_integrates_to_one_half_normal() {
        // Normal likelihood + half-normal prior (truncated to positive)
        let likelihood: Likelihood = NormalLikelihood::new(5.5, 32.35).unwrap().into();
        let prior: Prior = NormalPrior::new(0.0, 13.3, (Some(0.0), None))
            .unwrap()
            .into();

        let model = likelihood * prior;
        let posterior = model.posterior().unwrap();
        let integral = posterior.integral().unwrap();

        assert_relative_eq!(integral, 1.0, epsilon = EPS);
    }
}

mod normalization_tests {
    //! Tests for prior normalization constants.
    //!
    //! Verifies that normalize() returns the correct CDF-based area
    //! for various prior types and truncation configurations.

    use approx::assert_relative_eq;
    use bayesplay::prelude::*;

    // ── Normal prior normalization ─────────────────────────────────────

    #[test]
    fn test_normal_unbounded_normalization() {
        let prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();
        assert_relative_eq!(prior.normalize().unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normal_half_normalization() {
        // Symmetric around 0, positive half → 0.5
        let prior = NormalPrior::new(0.0, 1.0, (Some(0.0), None)).unwrap();
        assert_relative_eq!(prior.normalize().unwrap(), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_normal_one_sd_normalization() {
        // ±1 SD contains ~68.27% of mass
        let prior = NormalPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).unwrap();
        assert_relative_eq!(prior.normalize().unwrap(), 0.6827, epsilon = 0.001);
    }

    #[test]
    fn test_normal_two_sd_normalization() {
        // ±2 SD contains ~95.45% of mass
        let prior = NormalPrior::new(0.0, 1.0, (Some(-2.0), Some(2.0))).unwrap();
        assert_relative_eq!(prior.normalize().unwrap(), 0.9545, epsilon = 0.001);
    }

    // ── Cauchy prior normalization ─────────────────────────────────────

    #[test]
    fn test_cauchy_unbounded_normalization() {
        let prior = CauchyPrior::new(0.0, 1.0, (None, None)).unwrap();
        assert_relative_eq!(prior.normalize().unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cauchy_half_normalization() {
        // Symmetric around 0, positive half → 0.5
        let prior = CauchyPrior::new(0.0, 1.0, (Some(0.0), None)).unwrap();
        assert_relative_eq!(prior.normalize().unwrap(), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_cauchy_symmetric_range_normalization() {
        // Cauchy(0,1) from -1 to 1: CDF(1) - CDF(-1) = 0.75 - 0.25 = 0.5
        let prior = CauchyPrior::new(0.0, 1.0, (Some(-1.0), Some(1.0))).unwrap();
        assert_relative_eq!(prior.normalize().unwrap(), 0.5, epsilon = 0.001);
    }

    // ── Student-t prior normalization ──────────────────────────────────

    #[test]
    fn test_student_t_unbounded_normalization() {
        let prior = StudentTPrior::new(0.0, 1.0, 3.0, (None, None)).unwrap();
        assert_relative_eq!(prior.normalize().unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_student_t_half_normalization() {
        // Symmetric around 0, positive half → 0.5
        let prior = StudentTPrior::new(0.0, 1.0, 3.0, (Some(0.0), None)).unwrap();
        assert_relative_eq!(prior.normalize().unwrap(), 0.5, epsilon = 1e-6);
    }

    // ── Uniform prior normalization ────────────────────────────────────

    #[test]
    fn test_uniform_normalization() {
        // Uniform prior always integrates to 1 by definition
        let prior: Prior = UniformPrior::new(0.0, 1.0).unwrap().into();
        let integral = prior.integral().unwrap();
        assert_relative_eq!(integral, 1.0, epsilon = 0.001);
    }

    #[test]
    fn test_uniform_wider_normalization() {
        let prior: Prior = UniformPrior::new(-5.0, 5.0).unwrap().into();
        let integral = prior.integral().unwrap();
        assert_relative_eq!(integral, 1.0, epsilon = 0.001);
    }

    // ── Beta prior normalization ───────────────────────────────────────

    #[test]
    fn test_beta_normalization() {
        // Beta(1,1) is uniform on [0,1], integrates to 1
        let prior: Prior = BetaPrior::new(1.0, 1.0, (None, None)).unwrap().into();
        let integral = prior.integral().unwrap();
        assert_relative_eq!(integral, 1.0, epsilon = 0.001);
    }

    #[test]
    fn test_beta_asymmetric_normalization() {
        // Beta(2,5) integrates to 1 on [0,1]
        let prior: Prior = BetaPrior::new(2.0, 5.0, (None, None)).unwrap().into();
        let integral = prior.integral().unwrap();
        assert_relative_eq!(integral, 1.0, epsilon = 0.001);
    }

    // ── Point prior normalization ──────────────────────────────────────

    #[test]
    fn test_point_prior_normalization() {
        // Point prior integrates to 1 (by convention)
        let prior: Prior = PointPrior::new(0.0).unwrap().into();
        let integral = prior.integral().unwrap();
        assert_relative_eq!(integral, 1.0, epsilon = 1e-10);
    }
}

mod audit_regression_tests {
    use approx::assert_relative_eq;
    use bayesplay::prelude::*;
    use statrs::distribution::{Continuous, StudentsT};

    #[test]
    fn scalar_point_predictives_normalize_in_observation_space() {
        let cases: [(Likelihood, f64, f64); 7] = [
            (
                NormalLikelihood::new(0.2, 1.3).unwrap().into(),
                0.3,
                f64::NEG_INFINITY,
            ),
            (
                StudentTLikelihood::new(0.2, 1.3, 8.0).unwrap().into(),
                0.3,
                f64::NEG_INFINITY,
            ),
            (BinomialLikelihood::new(3.0, 8.0).unwrap().into(), 0.3, 0.0),
            (
                NoncentralTLikelihood::new(0.2, 8.0).unwrap().into(),
                0.2,
                f64::NEG_INFINITY,
            ),
            (
                NoncentralDLikelihood::new(0.2, 9.0).unwrap().into(),
                0.1,
                f64::NEG_INFINITY,
            ),
            (
                NoncentralD2Likelihood::new(0.2, 12.0, 18.0).unwrap().into(),
                0.1,
                f64::NEG_INFINITY,
            ),
            (
                VarianceRatioLikelihood::new(1.4, 8.0).unwrap().into(),
                1.3,
                0.0,
            ),
        ];
        for (likelihood, point, lower) in cases {
            let predictive =
                (likelihood * Prior::from(PointPrior::new(point).unwrap())).predictive();
            let area = if let Likelihood::Binomial(_) = likelihood {
                (0..=8)
                    .map(|x| predictive.function(x as f64).unwrap())
                    .sum()
            } else {
                // Quadrature samples the supported open interval, not invalid endpoints.
                rmath::integrate(
                    move |x| predictive.function(x).unwrap(),
                    lower,
                    f64::INFINITY,
                )
                .unwrap()
                .value
            };
            assert!((area - 1.0).abs() < 1e-7, "{likelihood:?}: area = {area}");
        }
    }

    #[test]
    fn continuous_effect_predictives_match_normal_mixture_identity() {
        let observation = 0.2;
        let prior_sd: f64 = 0.15;
        for (likelihood, effective_n, df) in [
            (
                Likelihood::from(NoncentralDLikelihood::new(observation, 9.0).unwrap()),
                9.0,
                8.0,
            ),
            (
                Likelihood::from(NoncentralD2Likelihood::new(observation, 12.0, 18.0).unwrap()),
                7.2,
                28.0,
            ),
        ] {
            // D = (Z / sqrt(effective_n) + delta) / sqrt(V / df).
            // Mixing delta ~ N(0, prior_sd^2) gives a scaled central t, including
            // its observation-density normalizer (not just a Bayes-factor kernel).
            let scale = (1.0 / effective_n + prior_sd.powi(2)).sqrt();
            let expected = StudentsT::new(0.0, scale, df).unwrap().pdf(observation);
            let prior: Prior = NormalPrior::new(0.0, prior_sd, (None, None))
                .unwrap()
                .into();
            assert_relative_eq!(
                (likelihood * prior)
                    .predictive()
                    .function(observation)
                    .unwrap(),
                expected,
                epsilon = 1e-8,
                max_relative = 1e-7
            );
        }
    }

    #[test]
    fn beta_prime_marginal_matches_beta_rho_coordinates() {
        let likelihood: Likelihood = VarianceRatioLikelihood::new(1.4, 8.0).unwrap().into();
        let beta_prime: Prior = BetaPrimePrior::new(2.3, (None, None)).unwrap().into();
        let beta = BetaPrior::new(2.3, 2.3, (None, None)).unwrap();
        // rho = r / (1 + r); p(r) = p(rho) / (1 + r)^2.
        let r = 1.7;
        assert_relative_eq!(
            beta_prime.function(r).unwrap(),
            beta.function(r / (1.0 + r)).unwrap() / (1.0 + r).powi(2),
            epsilon = 1e-12
        );
        let transformed = rmath::integrate(
            move |rho| {
                likelihood.function(rho / (1.0 - rho)).unwrap() * beta.function(rho).unwrap()
            },
            0.0,
            1.0,
        )
        .unwrap()
        .value;
        assert_relative_eq!(
            (likelihood * beta_prime).integrate(None, None).unwrap(),
            transformed,
            epsilon = 1e-9,
            max_relative = 1e-8
        );
    }
}

mod type_of_tests {
    //! Tests for PriorFamily identification via the Family trait.
    //!
    //! Verifies that each prior type correctly reports its family
    //! and that is_point() returns true only for PointPrior.

    use bayesplay::prelude::*;

    #[test]
    fn test_normal_type_of() {
        let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap().into();
        assert_eq!(prior.family(), PriorFamily::Normal);
    }

    #[test]
    fn test_cauchy_type_of() {
        let prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).unwrap().into();
        assert_eq!(prior.family(), PriorFamily::Cauchy);
    }

    #[test]
    fn test_student_t_type_of() {
        let prior: Prior = StudentTPrior::new(0.0, 1.0, 3.0, (None, None))
            .unwrap()
            .into();
        assert_eq!(prior.family(), PriorFamily::StudentT);
    }

    #[test]
    fn test_uniform_type_of() {
        let prior: Prior = UniformPrior::new(0.0, 1.0).unwrap().into();
        assert_eq!(prior.family(), PriorFamily::Uniform);
    }

    #[test]
    fn test_beta_type_of() {
        let prior: Prior = BetaPrior::new(1.0, 1.0, (None, None)).unwrap().into();
        assert_eq!(prior.family(), PriorFamily::Beta);
    }

    #[test]
    fn test_point_type_of() {
        let prior: Prior = PointPrior::new(0.0).unwrap().into();
        assert_eq!(prior.family(), PriorFamily::Point);
    }

    #[test]
    fn test_point_is_point() {
        let prior: Prior = PointPrior::new(0.0).unwrap().into();
        assert!(prior.is_point());
    }

    #[test]
    fn test_normal_is_not_point() {
        let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap().into();
        assert!(!prior.is_point());
    }

    #[test]
    fn test_cauchy_is_not_point() {
        let prior: Prior = CauchyPrior::new(0.0, 1.0, (None, None)).unwrap().into();
        assert!(!prior.is_point());
    }

    #[test]
    fn test_student_t_is_not_point() {
        let prior: Prior = StudentTPrior::new(0.0, 1.0, 3.0, (None, None))
            .unwrap()
            .into();
        assert!(!prior.is_point());
    }

    #[test]
    fn test_uniform_is_not_point() {
        let prior: Prior = UniformPrior::new(0.0, 1.0).unwrap().into();
        assert!(!prior.is_point());
    }

    #[test]
    fn test_beta_is_not_point() {
        let prior: Prior = BetaPrior::new(1.0, 1.0, (None, None)).unwrap().into();
        assert!(!prior.is_point());
    }
}
