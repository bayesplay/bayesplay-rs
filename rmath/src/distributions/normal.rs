use crate::error::{Distribution, Error, Result, require_not_nan, require_positive};
use statrs::distribution::{Continuous, ContinuousCDF};

fn normal(mean: f64, sd: f64) -> Result<statrs::distribution::Normal> {
    require_not_nan(Distribution::Normal, "mean", mean)?;
    require_positive(Distribution::Normal, "sd", sd)?;
    statrs::distribution::Normal::new(mean, sd).map_err(|_| Error::InvalidParameter {
        distribution: Distribution::Normal,
        parameter: "sd".into(),
        value: sd,
        requirement: "rejected by statrs".into(),
    })
}

pub fn pnorm(q: f64, mean: f64, sd: f64, lower_tail: bool, log_p: bool) -> Result<f64> {
    let dist = normal(mean, sd)?;

    match (lower_tail, log_p) {
        (true, true) => Ok(if q > mean {
            (-dist.sf(q)).ln_1p()
        } else {
            ln_upper_tail((mean - q) / sd)
        }),
        (true, false) => Ok(dist.cdf(q)),
        (false, true) => Ok(if q < mean {
            (-dist.cdf(q)).ln_1p()
        } else {
            ln_upper_tail((q - mean) / sd)
        }),
        (false, false) => Ok(dist.sf(q)),
    }
}

/// Beyond this many standard deviations `Q(z) = P(Z > z)` is evaluated by its
/// asymptotic expansion, since statrs' value underflows near z = 38.
const ASYMPTOTIC_TAIL_Z: f64 = 30.0;

/// `ln P(Z > z)` for a standard normal `Z` and `z >= 0`, accurate however far
/// into the tail `z` is.
fn ln_upper_tail(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z < ASYMPTOTIC_TAIL_Z {
        let standard = statrs::distribution::Normal::new(0.0, 1.0).expect("standard normal");
        return standard.sf(z).ln();
    }
    if z.is_infinite() {
        return f64::NEG_INFINITY;
    }
    // Q(z) = phi(z) / z * (1 - 1/z^2 + 3/z^4 - 15/z^6 + 105/z^8 - ...); at
    // z >= 30 the terms after these are below 1e-14 relative.
    let w = 1.0 / (z * z);
    let series = 1.0 - w * (1.0 - 3.0 * w * (1.0 - 5.0 * w * (1.0 - 7.0 * w * (1.0 - 9.0 * w))));
    -0.5 * z * z - 0.5 * (2.0 * core::f64::consts::PI).ln() - z.ln() + series.ln()
}

pub fn dnorm(x: f64, mean: f64, sd: f64, log: bool) -> Result<f64> {
    let dist = normal(mean, sd)?;

    match log {
        true => Ok(dist.ln_pdf(x)),
        false => Ok(dist.pdf(x)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extreme_tails_and_truncation_mass() {
        let small = pnorm(10.0, 0.0, 1.0, false, false).unwrap();
        let want = 7.619_853_024_160_47e-24;
        assert!((small / want - 1.0).abs() < 1e-9);
        assert_eq!(pnorm(-10.0, 0.0, 1.0, true, false).unwrap(), small);
        assert_eq!(pnorm(-10.0, 0.0, 1.0, true, true).unwrap(), small.ln());
        assert_eq!(pnorm(10.0, 0.0, 1.0, false, true).unwrap(), small.ln());
        assert_eq!(pnorm(10.0, 0.0, 1.0, true, true).unwrap(), (-small).ln_1p());
        assert_eq!(
            pnorm(-10.0, 0.0, 1.0, false, true).unwrap(),
            (-small).ln_1p()
        );
        let mass = small - pnorm(11.0, 0.0, 1.0, false, false).unwrap();
        assert!(mass > 0.0 && mass < small);
    }

    #[test]
    fn log_tail_probabilities_do_not_underflow() {
        // R: pnorm(q, log.p = TRUE)
        for (q, want) in [
            (-30.0, -4.543_212_439_563_432e2),
            (-37.5, -7.076_689_893_175_072e2),
            (-38.5, -7.456_952_702_904_11e2),
            (-40.0, -8.046_084_420_137_538e2),
            (-100.0, -5.005_524_208_694_205e3),
            (-1e4, -5.000_001_012_927_891e7),
        ] {
            let got = pnorm(q, 0.0, 1.0, true, true).unwrap();
            assert!((got / want - 1.0).abs() < 1e-14, "q={q}: {got} vs {want}");
            let upper = pnorm(-q, 0.0, 1.0, false, true).unwrap();
            assert!(
                (upper / want - 1.0).abs() < 1e-14,
                "q={q}: {upper} vs {want}"
            );
        }
        // Scaled and shifted.
        let got = pnorm(-79.0, 1.0, 2.0, true, true).unwrap();
        assert!((got / -8.046_084_420_137_538e2 - 1.0).abs() < 1e-14);
        assert_eq!(
            pnorm(f64::NEG_INFINITY, 0.0, 1.0, true, true).unwrap(),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn log_density_does_not_underflow() {
        let got = dnorm(40.0, 0.0, 1.0, true).unwrap();
        let want = -800.0 - 0.5 * (2.0 * core::f64::consts::PI).ln();
        assert!((got - want).abs() < 1e-12);
    }
}
