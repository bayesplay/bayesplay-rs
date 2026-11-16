use crate::error::{Distribution, Error, Result, require_positive};
use statrs::distribution::ContinuousCDF;

/// `ln B(a, b)`, the log of the beta function.
pub fn lbeta(a: f64, b: f64) -> Result<f64> {
    require_positive(Distribution::Beta, "a", a)?;
    require_positive(Distribution::Beta, "b", b)?;
    Ok(statrs::function::beta::ln_beta(a, b))
}

fn beta(shape1: f64, shape2: f64) -> Result<statrs::distribution::Beta> {
    require_positive(Distribution::Beta, "shape1", shape1)?;
    require_positive(Distribution::Beta, "shape2", shape2)?;
    statrs::distribution::Beta::new(shape1, shape2).map_err(|_| Error::InvalidParameter {
        distribution: Distribution::Beta,
        parameter: "shape1, shape2".into(),
        value: shape1,
        requirement: "rejected by statrs".into(),
    })
}

/// Beta distribution function. Accurate in both tails: probabilities below
/// `f64::MIN_POSITIVE` are still available on the log scale (`log_p`).
pub fn pbeta(q: f64, shape1: f64, shape2: f64, lower_tail: bool, log_p: bool) -> Result<f64> {
    let dist = beta(shape1, shape2)?;
    if q.is_nan() {
        return Ok(f64::NAN);
    }
    // The upper tail at q is the lower tail of Beta(shape2, shape1) at 1 - q.
    let (x, a, b) = if lower_tail {
        (q, shape1, shape2)
    } else {
        (1.0 - q, shape2, shape1)
    };
    let ln_p = if x <= 0.0 {
        f64::NEG_INFINITY
    } else if x >= 1.0 {
        0.0
    } else if let Some(ln_p) = ln_lower_tail_series(x, a, b) {
        ln_p
    } else if lower_tail {
        dist.cdf(q).ln()
    } else {
        dist.sf(q).ln()
    };
    Ok(if log_p { ln_p } else { ln_p.exp() })
}

/// `ln I_x(a, b)` (the regularized incomplete beta function) from the series
///
///   I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * sum_n [(a + b)_n / (a + 1)_n] x^n
///
/// (DLMF 8.17.8), whose terms are all positive. Used where it converges
/// quickly, which covers the far lower tail where `I_x` underflows; returns
/// `None` elsewhere.
fn ln_lower_tail_series(x: f64, a: f64, b: f64) -> Option<f64> {
    // Term ratios are (a + b + n) / (a + 1 + n) * x, bounded by this.
    let ratio_bound = x * ((a + b) / (a + 1.0)).max(1.0);
    if ratio_bound > 0.5 {
        return None;
    }
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 0..200 {
        let n = n as f64;
        term *= (a + b + n) / (a + 1.0 + n) * x;
        sum += term;
        if term <= sum * f64::EPSILON {
            break;
        }
    }
    Some(a * x.ln() + b * (-x).ln_1p() - a.ln() - statrs::function::beta::ln_beta(a, b) + sum.ln())
}

/// Beta density.
///
/// Computed directly as `(a - 1) ln x + (b - 1) ln(1 - x) - ln B(a, b)`:
/// statrs treats any `x` within machine epsilon of 0 or 1 as exactly 0 or 1,
/// so e.g. its log density of Beta(0.5, 0.5) at 1e-300 is `-inf` rather
/// than +344.
pub fn dbeta(x: f64, shape1: f64, shape2: f64, log: bool) -> Result<f64> {
    beta(shape1, shape2)?;
    let (a, b) = (shape1, shape2);
    let ln_density = if x.is_nan() {
        f64::NAN
    } else if !(0.0..=1.0).contains(&x) {
        f64::NEG_INFINITY
    } else if x == 0.0 || x == 1.0 {
        // At an endpoint the density is 0, finite or infinite depending on
        // whether the corresponding shape is above, at or below 1.
        let shape = if x == 0.0 { a } else { b };
        if shape < 1.0 {
            f64::INFINITY
        } else if shape > 1.0 {
            f64::NEG_INFINITY
        } else {
            -statrs::function::beta::ln_beta(a, b)
        }
    } else {
        (a - 1.0) * x.ln() + (b - 1.0) * (-x).ln_1p() - statrs::function::beta::ln_beta(a, b)
    };
    Ok(if log { ln_density } else { ln_density.exp() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_tails_match_r() {
        // R: pbeta(q, shape1, shape2, lower.tail, log.p = TRUE)
        for (q, a, b, lower_tail, want) in [
            (1e-70, 5.0, 1.0, true, -805.904_782_547_916),
            (1e-200, 3.0, 3.0, true, -1_379.248_470_703_433_4),
            (1e-20, 0.5, 0.5, true, -23.477_433_635_229_91),
            (0.3, 2.0, 5.0, true, -0.545_028_945_107_488_7),
            (1e-5, 200.0, 3.0, true, -2_292.666_687_469_742),
            (0.999_999, 2.0, 300.0, false, -4_138.946_058_112_584),
        ] {
            let got = pbeta(q, a, b, lower_tail, true).unwrap();
            assert!(
                (got / want - 1.0).abs() < 1e-12,
                "pbeta({q}, {a}, {b}, {lower_tail}): {got} vs {want}"
            );
        }
        assert_eq!(pbeta(1e-200, 3.0, 3.0, false, true).unwrap(), 0.0);
    }

    #[test]
    fn log_density_matches_r_near_the_endpoints() {
        // R: dbeta(x, shape1, shape2, log = TRUE)
        for (x, a, b, want) in [
            (5e-71, 5.0, 1.0, -645.886_976_848_138_4),
            (1e-300, 0.5, 0.5, 344.243_034_063_257_4),
            (1e-200, 2.0, 3.0, -458.032_111_949_021_2),
            (0.3, 2.0, 5.0, 0.770_524_801_581_289_8),
            (1.0 - 1e-12, 2.0, 0.5, 13.527_839_546_493_9),
        ] {
            let got = dbeta(x, a, b, true).unwrap();
            assert!(
                (got / want - 1.0).abs() < 1e-11,
                "dbeta({x}, {a}, {b}): {got} vs {want}"
            );
        }
        // R: dbeta(0, 1, 3) = 3, dbeta(0, 0.5, 3) = Inf, dbeta(0, 2, 3) = 0, dbeta(1, 2, 1) = 2
        assert!((dbeta(0.0, 1.0, 3.0, false).unwrap() / 3.0 - 1.0).abs() < 1e-12);
        assert_eq!(dbeta(0.0, 0.5, 3.0, false).unwrap(), f64::INFINITY);
        assert_eq!(dbeta(0.0, 2.0, 3.0, false).unwrap(), 0.0);
        assert!((dbeta(1.0, 2.0, 1.0, false).unwrap() / 2.0 - 1.0).abs() < 1e-12);
        assert_eq!(dbeta(1.5, 2.0, 1.0, false).unwrap(), 0.0);
    }

    #[test]
    fn linear_scale_and_endpoints() {
        assert_eq!(pbeta(0.0, 2.0, 3.0, true, false).unwrap(), 0.0);
        assert_eq!(pbeta(1.0, 2.0, 3.0, true, false).unwrap(), 1.0);
        assert_eq!(pbeta(-1.0, 2.0, 3.0, false, false).unwrap(), 1.0);
        // Beta(2, 1) has CDF x^2.
        for x in [1e-3, 0.25, 0.5, 0.9] {
            let lower = pbeta(x, 2.0, 1.0, true, false).unwrap();
            let upper = pbeta(x, 2.0, 1.0, false, false).unwrap();
            assert!((lower / (x * x) - 1.0).abs() < 1e-13);
            assert!((lower + upper - 1.0).abs() < 1e-14);
        }
    }
}
