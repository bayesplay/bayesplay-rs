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
        (true, true) => Ok(dist.cdf(q).ln()),
        (true, false) => Ok(dist.cdf(q)),
        (false, true) => Ok((1.0 - dist.cdf(q)).ln()),
        (false, false) => Ok(1.0 - dist.cdf(q)),
    }
}

pub fn dnorm(x: f64, mean: f64, sd: f64, log: bool) -> Result<f64, &'static str> {
    let dist = statrs::distribution::Normal::new(mean, sd)
        .map_err(|_| "Error creating Normal distribution")?;

    match log {
        true => Ok(dist.pdf(x).ln()),
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
