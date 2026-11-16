//! Parameter validation: invalid priors and likelihoods must be reported by
//! `validate()` and surface as errors (never panics or silent zeros).

mod validation_tests {
    //! Tests for validation of likelihood and prior parameters

    use bayesplay::prelude::*;

    #[test]
    fn test_invalid_normal_likelihood_se() {
        let likelihood: Likelihood = NormalLikelihood {
            mean: 0.0,
            se: -1.0,
        }
        .into();
        let prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();

        let model = likelihood * Prior::from(prior);
        let result = model.integral();

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_normal_prior_sd() {
        let likelihood = NormalLikelihood::new(0.0, 1.0).unwrap();
        let prior = NormalPrior {
            mean: 0.0,
            sd: -1.0,
            range: (None, None),
        };

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
        let prior = BetaPrior::new(1.0, 1.0, (None, None)).unwrap();

        let model = Likelihood::from(likelihood) * Prior::from(prior);
        let result = model.integral();

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_cauchy_prior_scale() {
        let likelihood = NormalLikelihood::new(0.0, 1.0).unwrap();
        let prior = CauchyPrior {
            location: 0.0,
            scale: -1.0,
            range: (None, None),
        };

        let model = Likelihood::from(likelihood) * Prior::from(prior);
        let result = model.integral();

        assert!(result.is_err());
    }
}

mod extended_validation_tests {
    //! Comprehensive validation error tests.
    //!
    //! These tests verify that direct `.validate()` calls on all prior and
    //! likelihood types return the correct error variants, matching the
    //! error-checking behavior from the R package's test-error_messages.R.

    use bayesplay::prelude::*;

    // ── Prior validations ──────────────────────────────────────────────

    #[test]
    fn test_normal_prior_negative_sd() {
        let prior = NormalPrior {
            mean: 0.0,
            sd: -1.0,
            range: (None, None),
        };
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidStandardDeviation(sd)) if sd == -1.0
        ));
    }

    #[test]
    fn test_normal_prior_zero_sd() {
        let prior = NormalPrior {
            mean: 0.0,
            sd: 0.0,
            range: (None, None),
        };
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidStandardDeviation(sd)) if sd == 0.0
        ));
    }

    #[test]
    fn test_cauchy_prior_negative_scale() {
        let prior = CauchyPrior {
            location: 0.0,
            scale: -1.0,
            range: (None, None),
        };
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidScale(s)) if s == -1.0
        ));
    }

    #[test]
    fn test_cauchy_prior_zero_scale() {
        let prior = CauchyPrior {
            location: 0.0,
            scale: 0.0,
            range: (None, None),
        };
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidScale(s)) if s == 0.0
        ));
    }

    #[test]
    fn test_student_t_prior_negative_sd() {
        let prior = StudentTPrior {
            mean: 0.0,
            sd: -1.0,
            df: 3.0,
            range: (None, None),
        };
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidStandardDeviation(sd)) if sd == -1.0
        ));
    }

    #[test]
    fn test_student_t_prior_invalid_df_fraction() {
        // df = 0.5, which is < 1 and should be invalid
        let prior = StudentTPrior {
            mean: 0.0,
            sd: 1.0,
            df: 0.5,
            range: (None, None),
        };
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidDegreesOfFreedom(df)) if df == 0.5
        ));
    }

    #[test]
    fn test_student_t_prior_invalid_df_one() {
        // df = 1 is invalid - must be > 1
        let prior = StudentTPrior {
            mean: 0.0,
            sd: 1.0,
            df: 1.0,
            range: (None, None),
        };
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidDegreesOfFreedom(df)) if df == 1.0
        ));
    }

    #[test]
    fn test_student_t_prior_multiple_errors() {
        // Both sd and df invalid
        let prior = StudentTPrior {
            mean: 0.0,
            sd: -1.0,
            df: 0.5,
            range: (None, None),
        };
        assert!(matches!(
            prior.validate(),
            Err(PriorError::MultipleErrors(_))
        ));
    }

    #[test]
    fn test_uniform_prior_equal_bounds() {
        let result = UniformPrior { min: 1.0, max: 1.0 };
        assert!(matches!(result.validate(), Err(PriorError::InvalidRange)));
    }

    #[test]
    fn test_uniform_prior_reversed_bounds() {
        let result = UniformPrior { min: 2.0, max: 1.0 };
        assert!(matches!(result.validate(), Err(PriorError::InvalidRange)));
    }

    #[test]
    fn test_beta_prior_negative_alpha() {
        let prior = BetaPrior {
            alpha: -1.0,
            beta: 1.0,
            range: (None, None),
        };
        assert!(matches!(
            prior.validate(),
            Err(PriorError::InvalidShapeParameter(v)) if v == -1.0
        ));
    }

    #[test]
    fn test_beta_prior_negative_beta() {
        let prior = BetaPrior {
            alpha: 1.0,
            beta: -1.0,
            range: (None, None),
        };
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
        let likelihood = StudentTLikelihood {
            mean: 0.0,
            sd: -1.0,
            df: 10.0,
        };
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidSD(sd)) if sd == -1.0
        ));
    }

    #[test]
    fn test_student_t_likelihood_negative_df() {
        let likelihood = StudentTLikelihood {
            mean: 0.0,
            sd: 1.0,
            df: -5.0,
        };
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidDF(df)) if df == -5.0
        ));
    }

    #[test]
    fn test_student_t_likelihood_multiple_errors() {
        let likelihood = StudentTLikelihood {
            mean: 0.0,
            sd: -1.0,
            df: -5.0,
        };
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
        let likelihood = NoncentralDLikelihood { d: 0.5, n: 0.5 };
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidN(n)) if n == 0.5
        ));
    }

    #[test]
    fn test_noncentral_d2_likelihood_small_n1() {
        let likelihood = NoncentralD2Likelihood {
            d: 0.5,
            n1: 0.5,
            n2: 10.0,
        };
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidN1(n)) if n == 0.5
        ));
    }

    #[test]
    fn test_noncentral_d2_likelihood_small_n2() {
        let likelihood = NoncentralD2Likelihood {
            d: 0.5,
            n1: 10.0,
            n2: 0.5,
        };
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidN2(n)) if n == 0.5
        ));
    }

    #[test]
    fn test_noncentral_t_likelihood_small_df() {
        let likelihood = NoncentralTLikelihood { t: 2.0, df: 0.5 };
        assert!(matches!(
            likelihood.validate(),
            Err(LikelihoodError::InvalidDF(df)) if df == 0.5
        ));
    }
}

mod invalid_input_tests {
    //! Invalid parameters must be reported by `validate()` and surface as
    //! errors (never panics or silent zeros) from integration.
    use bayesplay::prelude::*;

    fn normal_likelihood() -> Likelihood {
        NormalLikelihood::new(0.5, 0.2).unwrap().into()
    }

    #[test]
    fn reversed_and_nan_prior_ranges_are_invalid() {
        let priors: [Prior; 8] = [
            NormalPrior {
                mean: 0.0,
                sd: 1.0,
                range: (Some(5.0), Some(1.0)),
            }
            .into(),
            CauchyPrior {
                location: 0.0,
                scale: 1.0,
                range: (Some(5.0), Some(1.0)),
            }
            .into(),
            StudentTPrior {
                mean: 0.0,
                sd: 1.0,
                df: 3.0,
                range: (Some(5.0), Some(1.0)),
            }
            .into(),
            NormalPrior {
                mean: 0.0,
                sd: 1.0,
                range: (Some(f64::NAN), Some(1.0)),
            }
            .into(),
            CauchyPrior {
                location: 0.0,
                scale: 1.0,
                range: (None, Some(f64::NAN)),
            }
            .into(),
            UniformPrior {
                min: f64::NAN,
                max: 1.0,
            }
            .into(),
            UniformPrior {
                min: f64::NEG_INFINITY,
                max: f64::INFINITY,
            }
            .into(),
            UniformPrior {
                min: 0.0,
                max: f64::INFINITY,
            }
            .into(),
        ];
        for prior in priors {
            assert!(prior.validate().is_err(), "{prior:?} validated");
            assert!(
                matches!(
                    (normal_likelihood() * prior).integral(),
                    Err(IntegralError::Prior(_))
                ),
                "{prior:?} integrated"
            );
            assert!(prior.integral().is_err(), "{prior:?} prior integrated");
        }
    }

    #[test]
    fn non_finite_prior_parameters_are_invalid() {
        let priors: [Prior; 6] = [
            NormalPrior {
                mean: 0.0,
                sd: f64::NAN,
                range: (None, None),
            }
            .into(),
            NormalPrior {
                mean: 0.0,
                sd: f64::INFINITY,
                range: (None, None),
            }
            .into(),
            NormalPrior {
                mean: f64::NAN,
                sd: 1.0,
                range: (None, None),
            }
            .into(),
            CauchyPrior {
                location: 0.0,
                scale: f64::INFINITY,
                range: (None, None),
            }
            .into(),
            CauchyPrior {
                location: f64::INFINITY,
                scale: 1.0,
                range: (None, None),
            }
            .into(),
            StudentTPrior {
                mean: 0.0,
                sd: f64::NAN,
                df: 3.0,
                range: (None, None),
            }
            .into(),
        ];
        for prior in priors {
            assert!(prior.validate().is_err(), "{prior:?} validated");
            assert!(prior.integral().is_err(), "{prior:?} prior integrated");
        }
    }

    #[test]
    fn invalid_priors_and_bounds_do_not_panic_when_integrated() {
        let prior: Prior = NormalPrior {
            mean: 0.0,
            sd: -1.0,
            range: (None, None),
        }
        .into();
        assert!(matches!(prior.integral(), Err(IntegralError::Prior(_))));

        // Bounds outside the beta's support are clamped to it.
        let beta: Prior = BetaPrior::new(2.0, 2.0, (None, None)).unwrap().into();
        let mass = beta.integrate(Some(-1.0), Some(2.0)).unwrap();
        assert!((mass - 1.0).abs() < 1e-10);
        assert!(matches!(
            beta.integrate(Some(0.8), Some(0.2)),
            Err(IntegralError::InvalidBounds(..))
        ));
    }

    #[test]
    fn negative_and_fractional_successes_are_invalid() {
        for successes in [-1.0, 2.5, 11.0, f64::NAN] {
            let likelihood = BinomialLikelihood {
                successes,
                trials: 10.0,
            };
            assert!(likelihood.validate().is_err(), "successes={successes}");
            assert!(likelihood.function(0.3).is_err(), "successes={successes}");
        }
        assert!(
            BinomialLikelihood::new(0.0, 10.0)
                .unwrap()
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn effect_size_likelihoods_need_at_least_one_degree_of_freedom() {
        let likelihoods: [Likelihood; 5] = [
            NoncentralDLikelihood { d: 0.5, n: 1.0 }.into(),
            NoncentralDLikelihood {
                d: 0.5,
                n: f64::NAN,
            }
            .into(),
            NoncentralD2Likelihood {
                d: 0.5,
                n1: 1.0,
                n2: 1.0,
            }
            .into(),
            NoncentralD2Likelihood {
                d: 0.5,
                n1: f64::NAN,
                n2: 10.0,
            }
            .into(),
            NoncentralTLikelihood {
                t: 0.5,
                df: f64::NAN,
            }
            .into(),
        ];
        let prior: Prior = CauchyPrior::new(0.0, 0.707, (None, None)).unwrap().into();
        for likelihood in likelihoods {
            assert!(likelihood.validate().is_err(), "{likelihood:?} validated");
            assert!(
                matches!(
                    (likelihood * prior).integral(),
                    Err(IntegralError::Likelihood(_))
                ),
                "{likelihood:?} integrated"
            );
        }
        assert!(
            NoncentralDLikelihood::new(0.5, 2.0)
                .unwrap()
                .validate()
                .is_ok()
        );
        assert!(
            NoncentralD2Likelihood::new(0.5, 1.0, 2.0)
                .unwrap()
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn nan_scale_likelihoods_are_invalid() {
        let prior: Prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap().into();
        for likelihood in [
            Likelihood::from(NormalLikelihood {
                mean: 0.5,
                se: f64::NAN,
            }),
            StudentTLikelihood {
                mean: 0.5,
                sd: f64::NAN,
                df: 10.0,
            }
            .into(),
            StudentTLikelihood {
                mean: 0.5,
                sd: 1.0,
                df: f64::NAN,
            }
            .into(),
        ] {
            assert!(likelihood.validate().is_err(), "{likelihood:?} validated");
            assert!((likelihood * prior).integral().is_err());
        }
    }
}

mod constructor_tests {
    //! Constructors validate, so invalid parameters are rejected up front with
    //! the same error `validate()` reports.
    use bayesplay::prelude::*;

    #[test]
    fn prior_constructors_reject_invalid_parameters() {
        let rejected: [(PriorResult<Prior>, PriorError); 8] = [
            (
                NormalPrior::new(0.0, -1.0, (None, None)).map(Prior::from),
                PriorError::InvalidStandardDeviation(-1.0),
            ),
            (
                CauchyPrior::new(0.0, 0.0, (None, None)).map(Prior::from),
                PriorError::InvalidScale(0.0),
            ),
            (
                StudentTPrior::new(0.0, 1.0, 0.5, (None, None)).map(Prior::from),
                PriorError::InvalidDegreesOfFreedom(0.5),
            ),
            (
                UniformPrior::new(1.0, 0.0).map(Prior::from),
                PriorError::InvalidRange,
            ),
            (
                BetaPrior::new(-1.0, 1.0, (None, None)).map(Prior::from),
                PriorError::InvalidShapeParameter(-1.0),
            ),
            (
                BetaPrimePrior::new(0.5, (Some(-1.0), None)).map(Prior::from),
                PriorError::InvalidPositiveRangeBounds,
            ),
            (
                NormalPrior::new(0.0, 1.0, (Some(5.0), Some(1.0))).map(Prior::from),
                PriorError::InvalidRange,
            ),
            (
                PointPrior::new(f64::NAN).map(Prior::from),
                PriorError::InvalidValue(f64::NAN, f64::NEG_INFINITY, f64::INFINITY),
            ),
        ];
        for (result, expected) in rejected {
            let error = result.unwrap_err();
            assert_eq!(
                std::mem::discriminant(&error),
                std::mem::discriminant(&expected),
                "{error} vs {expected}"
            );
        }
    }

    #[test]
    fn likelihood_constructors_reject_invalid_parameters() {
        let rejected: [(LikelihoodResult<Likelihood>, LikelihoodError); 7] = [
            (
                NormalLikelihood::new(0.0, 0.0).map(Likelihood::from),
                LikelihoodError::InvalidSE(0.0),
            ),
            (
                StudentTLikelihood::new(0.0, -1.0, 10.0).map(Likelihood::from),
                LikelihoodError::InvalidSD(-1.0),
            ),
            (
                BinomialLikelihood::new(-1.0, 10.0).map(Likelihood::from),
                LikelihoodError::InvalidSuccess(-1.0),
            ),
            (
                NoncentralDLikelihood::new(0.5, 1.0).map(Likelihood::from),
                LikelihoodError::InvalidN(1.0),
            ),
            (
                NoncentralD2Likelihood::new(0.5, 0.5, 10.0).map(Likelihood::from),
                LikelihoodError::InvalidN1(0.5),
            ),
            (
                NoncentralTLikelihood::new(2.0, 0.5).map(Likelihood::from),
                LikelihoodError::InvalidDF(0.5),
            ),
            (
                VarianceRatioLikelihood::new(0.0, 9.0).map(Likelihood::from),
                LikelihoodError::InvalidU(0.0),
            ),
        ];
        for (result, expected) in rejected {
            let error = result.unwrap_err();
            assert_eq!(
                std::mem::discriminant(&error),
                std::mem::discriminant(&expected),
                "{error} vs {expected}"
            );
        }
    }

    #[test]
    fn mutated_parameters_are_still_caught_at_use() {
        // Fields are public, so a valid value can be made invalid after
        // construction; integration validates again.
        let mut prior = NormalPrior::new(0.0, 1.0, (None, None)).unwrap();
        prior.sd = -1.0;
        let likelihood: Likelihood = NormalLikelihood::new(0.5, 0.2).unwrap().into();
        assert!(matches!(
            (likelihood * Prior::from(prior)).integral(),
            Err(IntegralError::Prior(PriorError::InvalidStandardDeviation(
                _
            )))
        ));
    }
}
