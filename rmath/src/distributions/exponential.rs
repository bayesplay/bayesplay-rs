use statrs::distribution::{Continuous, ContinuousCDF};

pub struct Dexp {
    pub x: Option<f64>,
    pub rate: Option<f64>,
    pub log: Option<bool>,
}

impl Default for Dexp {
    fn default() -> Self {
        Dexp {
            x: None,
            rate: Some(1.0),
            log: Some(false),
        }
    }
}

pub struct Pexp {
    pub q: Option<f64>,
    pub rate: Option<f64>,
    pub lower_tail: Option<bool>,
    pub log_p: Option<bool>,
}

impl Default for Pexp {
    fn default() -> Self {
        Pexp {
            q: None,
            rate: Some(1.0),
            lower_tail: Some(true),
            log_p: Some(false),
        }
    }
}

pub fn dexp(x: f64, rate: f64, log: bool) -> Result<f64, &'static str> {
    let dist = statrs::distribution::Exp::new(rate)
        .map_err(|_| "Error creating Exponential distribution")?;

    match log {
        true => Ok(dist.pdf(x).ln()),
        false => Ok(dist.pdf(x)),
    }
}

pub fn pexp(q: f64, rate: f64, lower_tail: bool, log_p: bool) -> Result<f64, &'static str> {
    let dist = statrs::distribution::Exp::new(rate)
        .map_err(|_| "Error creating Exponential distribution")?;

    match (lower_tail, log_p) {
        (true, true) => Ok(dist.cdf(q).ln()),
        (true, false) => Ok(dist.cdf(q)),
        (false, true) => Ok((1.0 - dist.cdf(q)).ln()),
        (false, false) => Ok(1.0 - dist.cdf(q)),
    }
}

#[allow(dead_code)]
pub fn safe_dexp(args: Dexp) -> Result<f64, &'static str> {
    dexp(
        args.x.ok_or("argument \"x\" is missing, with no default")?,
        args.rate
            .ok_or("argument \"rate\" is missing, with no default")?,
        args.log
            .ok_or("argument \"log\" is missing, with no default")?,
    )
}

#[allow(dead_code)]
pub fn safe_pexp(args: Pexp) -> Result<f64, &'static str> {
    pexp(
        args.q.ok_or("argument \"q\" is missing, with no default")?,
        args.rate
            .ok_or("argument \"rate\" is missing, with no default")?,
        args.lower_tail
            .ok_or("argument \"lower_tail\" is missing, with no default")?,
        args.log_p
            .ok_or("argument \"log_p\" is missing, with no default")?,
    )
}
