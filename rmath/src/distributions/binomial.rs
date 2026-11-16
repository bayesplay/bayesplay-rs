use statrs::distribution::{Discrete, DiscreteCDF};

#[derive(Debug)]
pub struct Dbinom {
    pub x: Option<f64>,
    pub size: Option<f64>,
    pub prob: Option<f64>,
    pub log: Option<bool>,
}

impl Default for Dbinom {
    fn default() -> Self {
        Dbinom {
            x: None,
            size: None,
            prob: None,
            log: Some(false),
        }
    }
}

pub struct Pbinom {
    pub q: Option<f64>,
    pub size: Option<f64>,
    pub prob: Option<f64>,
    pub lower_tail: Option<bool>,
    pub log_p: Option<bool>,
}

impl Default for Pbinom {
    fn default() -> Self {
        Pbinom {
            q: None,
            size: None,
            prob: None,
            lower_tail: Some(true),
            log_p: Some(false),
        }
    }
}

fn binomial(size: f64, prob: f64) -> Result<statrs::distribution::Binomial, &'static str> {
    // `as u64` would silently truncate (or saturate) a non-integer or negative size.
    if !(size >= 0.0 && size.fract() == 0.0 && size <= u64::MAX as f64) {
        return Err("Binomial size must be a non-negative whole number");
    }
    statrs::distribution::Binomial::new(prob, size as u64)
        .map_err(|_| "Error creating Binomial distribution")
}

/// Binomial probability mass. As in R, the mass is 0 at any `x` that is not a
/// whole number in `0..=size`.
pub fn dbinom(x: f64, size: f64, prob: f64, log: bool) -> Result<f64, &'static str> {
    let dist = binomial(size, prob)?;
    if !(0.0..=size).contains(&x) || x.fract() != 0.0 {
        return Ok(if log { f64::NEG_INFINITY } else { 0.0 });
    }
    Ok(if log {
        dist.ln_pmf(x as u64)
    } else {
        dist.pmf(x as u64)
    })
}

#[allow(dead_code)]
pub fn safe_dbinom(args: Dbinom) -> Result<f64, &'static str> {
    dbinom(
        args.x.ok_or("argument \"x\" is missing, with no default")?,
        args.size
            .ok_or("argument \"size\" is missing, with no default")?,
        args.prob
            .ok_or("argument \"prob\" is missing, with no default")?,
        args.log
            .ok_or("argument \"log\" is missing, with no default")?,
    )
}

pub fn pbinom(
    x: f64,
    size: f64,
    prob: f64,
    lower_tail: bool,
    log_p: bool,
) -> Result<f64, &'static str> {
    let dist = binomial(size, prob)?;
    let (cdf, sf) = if x < 0.0 {
        (0.0, 1.0)
    } else if x >= size {
        (1.0, 0.0)
    } else {
        let k = x.floor() as u64;
        (dist.cdf(k), dist.sf(k))
    };
    let p = if lower_tail { cdf } else { sf };
    Ok(if log_p { p.ln() } else { p })
}

#[allow(dead_code)]
pub fn safe_pbinom(args: Pbinom) -> Result<f64, &'static str> {
    pbinom(
        args.q.ok_or("argument \"q\" is missing, with no default")?,
        args.size
            .ok_or("argument \"size\" is missing, with no default")?,
        args.prob
            .ok_or("argument \"prob\" is missing, with no default")?,
        args.lower_tail
            .ok_or("argument \"lower_tail\" is missing, with no default")?,
        args.log_p
            .ok_or("argument \"log_p\" is missing, with no default")?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mass_is_zero_off_the_integer_support() {
        let at_two = dbinom(2.0, 10.0, 0.3, false).unwrap();
        assert!((at_two - 0.233_474_440_5).abs() < 1e-9);
        for x in [2.7, -1.0, 11.0, f64::NAN] {
            assert_eq!(dbinom(x, 10.0, 0.3, false).unwrap(), 0.0);
            assert_eq!(dbinom(x, 10.0, 0.3, true).unwrap(), f64::NEG_INFINITY);
        }
    }

    #[test]
    fn invalid_sizes_are_errors() {
        for size in [-1.0, 10.5, f64::NAN, f64::INFINITY] {
            assert!(dbinom(1.0, size, 0.3, false).is_err());
            assert!(pbinom(1.0, size, 0.3, true, false).is_err());
        }
    }

    #[test]
    fn cdf_steps_at_integers_and_tails_complement() {
        assert_eq!(
            pbinom(2.5, 10.0, 0.3, true, false),
            pbinom(2.0, 10.0, 0.3, true, false)
        );
        assert_eq!(pbinom(-0.5, 10.0, 0.3, true, false).unwrap(), 0.0);
        assert_eq!(pbinom(10.0, 10.0, 0.3, true, false).unwrap(), 1.0);
        let lower = pbinom(3.0, 10.0, 0.3, true, false).unwrap();
        let upper = pbinom(3.0, 10.0, 0.3, false, false).unwrap();
        assert!((lower + upper - 1.0).abs() < 1e-15);
    }
}
