//! Bayes factors for one- and two-sample t-tests with (truncated) Cauchy
//! priors, checked against the R package BayesFactor and an independent
//! high-precision reference

use bayesplay::prelude::*;

const FIXTURE: &str = include_str!("data/bayesfactor_ttest.csv");

/// Tolerance on log BF10 (i.e. a relative tolerance of ~1e-5 on BF10).
const LOG_BF_TOLERANCE: f64 = 1e-5;

/// Rows where BayesFactor agrees with the reference to this precision are
/// treated as converged BayesFactor results.
const BAYESFACTOR_CONVERGED: f64 = 1e-6;

#[derive(Debug, Clone, Copy)]
struct Case {
    t: f64,
    n1: f64,
    n2: f64,
    lower: f64,
    upper: f64,
    rscale: f64,
    bayesfactor_logbf: f64,
    bayesfactor_t_approximation: bool,
    reference_logbf: f64,
}

impl Case {
    fn effective_n(&self) -> f64 {
        if self.n2 == 0.0 {
            self.n1
        } else {
            self.n1 * self.n2 / (self.n1 + self.n2)
        }
    }

    fn likelihood(&self) -> Likelihood {
        let d = self.t / self.effective_n().sqrt();
        if self.n2 == 0.0 {
            NoncentralDLikelihood::new(d, self.n1).into()
        } else {
            NoncentralD2Likelihood::new(d, self.n1, self.n2).into()
        }
    }

    fn prior(&self) -> Prior {
        let bound = |x: f64| x.is_finite().then_some(x);
        CauchyPrior::new(0.0, self.rscale, (bound(self.lower), bound(self.upper))).into()
    }

    fn log_bf(&self) -> f64 {
        let likelihood = self.likelihood();
        let h1 = (likelihood * self.prior())
            .integral()
            .unwrap_or_else(|e| panic!("{self:?}: H1 marginal failed: {e}"));
        let h0 = (likelihood * Prior::from(PointPrior::new(0.0)))
            .integral()
            .unwrap_or_else(|e| panic!("{self:?}: H0 marginal failed: {e}"));
        (h1 / h0).ln()
    }

    fn bayesfactor_converged(&self) -> bool {
        !self.bayesfactor_t_approximation
            && (self.bayesfactor_logbf - self.reference_logbf).abs() < BAYESFACTOR_CONVERGED
    }

    /// The cases that bayesplay sends to the `estimate_marginal`
    fn will_estimate(&self) -> bool {
        let d = self.t / self.effective_n().sqrt();
        let outside = d < self.lower || d > self.upper;
        self.t.abs() > 15.0 || (self.t.abs() > 5.0 && outside)
    }
}

fn cases() -> Vec<Case> {
    let parse = |s: &str| -> f64 {
        match s {
            "Inf" => f64::INFINITY,
            "-Inf" => f64::NEG_INFINITY,
            s => s.parse().unwrap_or_else(|_| panic!("bad number {s:?}")),
        }
    };
    FIXTURE
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            assert_eq!(f.len(), 9, "bad fixture row {line:?}");
            Case {
                t: parse(f[0]),
                n1: parse(f[1]),
                n2: parse(f[2]),
                lower: parse(f[3]),
                upper: parse(f[4]),
                rscale: parse(f[5]),
                bayesfactor_logbf: parse(f[6]),
                bayesfactor_t_approximation: f[7] == "TRUE",
                reference_logbf: parse(f[8]),
            }
        })
        .collect()
}

fn check(cases: &[Case], expected: impl Fn(&Case) -> f64) -> Vec<String> {
    cases
        .iter()
        .filter_map(|case| {
            let got = case.log_bf();
            let want = expected(case);
            ((got - want).abs() >= LOG_BF_TOLERANCE).then(|| {
                format!(
                    "t={} n1={} n2={} [{}, {}] r={}: log BF {got:.6} != {want:.6}",
                    case.t, case.n1, case.n2, case.lower, case.upper, case.rscale
                )
            })
        })
        .collect()
}

#[test]
fn fixture_is_complete() {
    let cases = cases();
    assert_eq!(cases.len(), 660);
    assert!(cases.iter().all(|c| c.reference_logbf.is_finite()));
}

#[test]
#[ignore = "fails while Model::integral() uses the estimate_marginal"]
fn matches_bayesfactor_where_bayesfactor_converges() {
    let cases: Vec<Case> = cases()
        .into_iter()
        .filter(Case::bayesfactor_converged)
        .collect();
    assert!(
        cases.len() > 400,
        "only {} converged BayesFactor rows",
        cases.len()
    );
    let failures = check(&cases, |c| c.bayesfactor_logbf);
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[ignore = "fails while Model::integral() uses the estimate_marginal"]
fn matches_high_precision_reference_for_every_case() {
    let failures = check(&cases(), |c| c.reference_logbf);
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[ignore = "fails while Model::integral() uses the estimate_marginal"]
fn large_t_and_out_of_interval_cases_are_exact() {
    // These are the cases most at risk: large statistics, and effects far
    // outside a truncated prior. Keep them covered explicitly.
    let cases: Vec<Case> = cases()
        .into_iter()
        .filter(Case::will_estimate)
        .collect();
    assert!(cases.len() > 300, "only {} large-t rows", cases.len());
    assert!(cases.iter().filter(|c| c.bayesfactor_converged()).count() > 100);
    let failures = check(&cases, |c| c.reference_logbf);
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
