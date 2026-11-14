use statrs::distribution::{Continuous, ContinuousCDF};

pub struct Dnorm {
    pub x: Option<f64>,
    pub mean: Option<f64>,
    pub sd: Option<f64>,
    pub log: Option<bool>,
}

impl Default for Dnorm {
    fn default() -> Self {
        Dnorm {
            x: None,
            mean: Some(0.0),
            sd: Some(1.0),
            log: Some(false),
        }
    }
}

pub struct Pnorm {
    pub q: Option<f64>,
    pub mean: Option<f64>,
    pub sd: Option<f64>,
    pub lower_tail: Option<bool>,
    pub log_p: Option<bool>,
}

impl Default for Pnorm {
    fn default() -> Self {
        Pnorm {
            q: None,
            mean: Some(0.0),
            sd: Some(1.0),
            lower_tail: Some(true),
            log_p: Some(false),
        }
    }
}

pub fn pnorm(
    q: f64,
    mean: f64,
    sd: f64,
    lower_tail: bool,
    log_p: bool,
) -> Result<f64, &'static str> {
    let dist = statrs::distribution::Normal::new(mean, sd)
        .map_err(|_| "Error creating Normal distribution")?;

    match (lower_tail, log_p) {
        (true, true) => Ok(if q > mean {
            (-dist.sf(q)).ln_1p()
        } else {
            dist.cdf(q).ln()
        }),
        (true, false) => Ok(dist.cdf(q)),
        (false, true) => Ok(if q < mean {
            (-dist.cdf(q)).ln_1p()
        } else {
            dist.sf(q).ln()
        }),
        (false, false) => Ok(dist.sf(q)),
    }
}

pub fn dnorm(x: f64, mean: f64, sd: f64, log: bool) -> Result<f64, &'static str> {
    let dist = statrs::distribution::Normal::new(mean, sd)
        .map_err(|_| "Error creating Normal distribution")?;

    match log {
        true => Ok(dist.ln_pdf(x)),
        false => Ok(dist.pdf(x)),
    }
}

#[allow(dead_code)]
pub fn safe_dnorm(args: Dnorm) -> Result<f64, &'static str> {
    dnorm(
        args.x.ok_or("argument \"x\" is missing, with no default")?,
        args.mean
            .ok_or("argument \"mean\" is missing, with no default")?,
        args.sd
            .ok_or("argument \"sd\" is missing, with no default")?,
        args.log
            .ok_or("argument \"log\" is missing, with no default")?,
    )
}

#[allow(dead_code)]
pub fn safe_pnorm(args: Pnorm) -> Result<f64, &'static str> {
    pnorm(
        args.q.ok_or("argument \"q\" is missing, with no default")?,
        args.mean
            .ok_or("argument \"mean\" is missing, with no default")?,
        args.sd
            .ok_or("argument \"sd\" is missing, with no default")?,
        args.lower_tail
            .ok_or("argument \"lower_tail\" is missing, with no default")?,
        args.log_p
            .ok_or("argument \"give_log\" is missing, with no default")?,
    )
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
    fn log_density_does_not_underflow() {
        let got = dnorm(40.0, 0.0, 1.0, true).unwrap();
        let want = -800.0 - 0.5 * (2.0 * core::f64::consts::PI).ln();
        assert!((got - want).abs() < 1e-12);
    }
}
