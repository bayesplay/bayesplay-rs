use statrs::distribution::{Continuous, ContinuousCDF};

pub struct Dunif {
    pub x: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub log: Option<bool>,
}

impl Default for Dunif {
    fn default() -> Self {
        Dunif {
            x: None,
            min: Some(0.0),
            max: Some(1.0),
            log: Some(false),
        }
    }
}

pub struct Punif {
    pub q: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub lower_tail: Option<bool>,
    pub log_p: Option<bool>,
}

impl Default for Punif {
    fn default() -> Self {
        Punif {
            q: None,
            min: Some(0.0),
            max: Some(1.0),
            lower_tail: Some(true),
            log_p: Some(false),
        }
    }
}

pub fn punif(
    x: f64,
    min: f64,
    max: f64,
    lower_tail: bool,
    log_p: bool,
) -> Result<f64, &'static str> {
    let dist = statrs::distribution::Uniform::new(min, max)
        .map_err(|_| "Error creating Uniform distribution")?;

    match (lower_tail, log_p) {
        (true, true) => Ok(dist.cdf(x).ln()),
        (true, false) => Ok(dist.cdf(x)),
        (false, true) => Ok((1.0 - dist.cdf(x)).ln()),
        (false, false) => Ok(1.0 - dist.cdf(x)),
    }
}

pub fn dunif(x: f64, min: f64, max: f64, log: bool) -> Result<f64, &'static str> {
    let dist = statrs::distribution::Uniform::new(min, max)
        .map_err(|_| "Error creating Uniform distribution")?;

    match log {
        true => Ok(dist.pdf(x).ln()),
        false => Ok(dist.pdf(x)),
    }
}

#[allow(dead_code)]
pub fn safe_dunif(args: Dunif) -> Result<f64, &'static str> {
    dunif(
        args.x.ok_or("argument \"x\" is missing, with no default")?,
        args.min
            .ok_or("argument \"min\" is missing, with no default")?,
        args.max
            .ok_or("argument \"max\" is missing, with no default")?,
        args.log
            .ok_or("argument \"log\" is missing, with no default")?,
    )
}

#[allow(dead_code)]
pub fn safe_punif(args: Punif) -> Result<f64, &'static str> {
    punif(
        args.q.ok_or("argument \"q\" is missing, with no default")?,
        args.min
            .ok_or("argument \"min\" is missing, with no default")?,
        args.max
            .ok_or("argument \"max\" is missing, with no default")?,
        args.lower_tail
            .ok_or("argument \"lower_tail\" is missing, with no default")?,
        args.log_p
            .ok_or("argument \"log_p\" is missing, with no default")?,
    )
}
