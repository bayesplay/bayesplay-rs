use statrs::distribution::{Continuous, ContinuousCDF};

pub fn pbeta(
    q: f64,
    shape1: f64,
    shape2: f64,
    lower_tail: bool,
    log_p: bool,
) -> Result<f64, &'static str> {
    let dist = statrs::distribution::Beta::new(shape1, shape2)
        .map_err(|_| "Error creating Beta distribution")?;

    match (lower_tail, log_p) {
        (true, true) => Ok(dist.cdf(q).ln()),
        (true, false) => Ok(dist.cdf(q)),
        (false, true) => Ok((1.0 - dist.cdf(q)).ln()),
        (false, false) => Ok(1.0 - dist.cdf(q)),
    }
}

pub fn dbeta(x: f64, shape1: f64, shape2: f64, log: bool) -> Result<f64, &'static str> {
    let dist = statrs::distribution::Beta::new(shape1, shape2)
        .map_err(|_| "Error creating Beta distribution")?;
    match log {
        true => Ok(dist.pdf(x).ln()),
        false => Ok(dist.pdf(x)),
    }
}

#[derive(Debug)]
pub struct Dbeta {
    pub x: Option<f64>,
    pub shape1: Option<f64>,
    pub shape2: Option<f64>,
    pub ncp: Option<f64>,
    pub log: Option<bool>,
}

impl Default for Dbeta {
    fn default() -> Self {
        Dbeta {
            x: None,
            shape1: None,
            shape2: None,
            ncp: None,
            log: Some(false),
        }
    }
}

pub struct Pbeta {
    pub q: Option<f64>,
    pub shape1: Option<f64>,
    pub shape2: Option<f64>,
    pub ncp: Option<f64>,
    pub lower_tail: Option<bool>,
    pub log_p: Option<bool>,
}

impl Default for Pbeta {
    fn default() -> Self {
        Pbeta {
            q: None,
            shape1: None,
            shape2: None,
            ncp: None,
            lower_tail: Some(true),
            log_p: Some(false),
        }
    }
}

#[allow(dead_code)]
pub fn safe_dbeta(args: Dbeta) -> Result<f64, &'static str> {
    match args.ncp.is_none() {
        true => dbeta(
            args.x.ok_or("argument \"x\" is missing, with no default")?,
            args.shape1
                .ok_or("argument \"shape1\" is missing, with no default")?,
            args.shape2
                .ok_or("argument \"shape2\" is missing, with no default")?,
            args.log.ok_or("Error with log")?,
        ),
        false => todo!(),
    }
}

#[allow(dead_code)]
pub fn safe_pbeta(args: Pbeta) -> Result<f64, &'static str> {
    match args.ncp.is_none() {
        true => pbeta(
            args.q.ok_or("argument \"q\" is missing, with no default")?,
            args.shape1
                .ok_or("argument \"shape1\" is missing, with no default")?,
            args.shape2
                .ok_or("argument \"shape2\" is missing, with no default")?,
            args.lower_tail.ok_or("Error with lower_tail")?,
            args.log_p.ok_or("Error with log_p")?,
        ),
        false => todo!(),
    }
}
