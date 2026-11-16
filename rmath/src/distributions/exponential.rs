use crate::error::{Distribution, Error, Result, require_positive};
use statrs::distribution::Continuous;

pub fn dexp(x: f64, rate: f64, log: bool) -> Result<f64> {
    require_positive(Distribution::Exponential, "rate", rate)?;
    let dist = statrs::distribution::Exp::new(rate).map_err(|_| Error::InvalidParameter {
        distribution: Distribution::Exponential,
        parameter: "rate".into(),
        value: rate,
        requirement: "rejected by statrs".into(),
    })?;

    match log {
        true => Ok(dist.ln_pdf(x)),
        false => Ok(dist.pdf(x)),
    }
}

pub fn pexp(q: f64, rate: f64, lower_tail: bool, log_p: bool) -> Result<f64> {
    require_positive(Distribution::Exponential, "rate", rate)?;

    if q < 0.0 {
        return match (lower_tail, log_p) {
            (true, true) => Ok(f64::NEG_INFINITY),
            (true, false) => Ok(0.0),
            (false, true) => Ok(0.0),
            (false, false) => Ok(1.0),
        };
    }

    // Use -expm1(-rate*q) instead of 1 - exp(-rate*q) to avoid catastrophic
    // cancellation when rate*q is very small (CDF ≈ 0) or very large (CDF ≈ 1).
    // -expm1(-x) = -(exp(-x) - 1) = 1 - exp(-x), computed accurately by expm1.
    let neg_rate_q = -(rate * q);
    let cdf = -neg_rate_q.exp_m1(); // = 1 - exp(-rate*q), accurate for all q
    let sf = neg_rate_q.exp(); // = exp(-rate*q) = 1 - CDF

    match (lower_tail, log_p) {
        // log(CDF) = log(1 - exp(-rate*q)) = log(-expm1(-rate*q))
        (true, true) => Ok(cdf.ln()),
        (true, false) => Ok(cdf),
        // log(SF) = log(exp(-rate*q)) = -rate*q
        (false, true) => Ok(sf.ln()),
        (false, false) => Ok(sf),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pexp_rejects_invalid_rates() {
        for rate in [0.0, -1.0, f64::NAN] {
            match pexp(1.0, rate, true, false) {
                Err(Error::InvalidParameter {
                    distribution: Distribution::Exponential,
                    parameter,
                    value,
                    ..
                }) => {
                    assert_eq!(parameter, "rate");
                    assert!(value == rate || (value.is_nan() && rate.is_nan()));
                }
                other => panic!("rate={rate}: {other:?}"),
            }
        }
    }

    #[test]
    fn pexp_valid_rate_returns_consistent_tails() {
        let cdf = pexp(1.0, 2.0, true, false).unwrap();
        let survival = pexp(1.0, 2.0, false, false).unwrap();
        let log_survival = pexp(1.0, 2.0, false, true).unwrap();

        assert!((cdf + survival - 1.0).abs() < f64::EPSILON);
        assert!((log_survival + 2.0).abs() < f64::EPSILON);
    }
}
