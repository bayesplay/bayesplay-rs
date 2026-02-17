use statrs::distribution::{Cauchy, Continuous, ContinuousCDF};

pub struct Dcauchy {
    pub x: Option<f64>,
    pub location: Option<f64>,
    pub scale: Option<f64>,
    pub log: Option<bool>,
}

impl Default for Dcauchy {
    fn default() -> Self {
        crate::distributions::cauchy::Dcauchy {
            x: None,
            location: Some(0.0),
            scale: Some(1.0),
            log: Some(false),
        }
    }
}

pub struct Pcauchy {
    pub q: Option<f64>,
    pub location: Option<f64>,
    pub scale: Option<f64>,
    pub lower_tail: Option<bool>,
    pub log_p: Option<bool>,
}

impl Default for Pcauchy {
    fn default() -> Self {
        Pcauchy {
            q: None,
            location: Some(0.0),
            scale: Some(1.0),
            lower_tail: Some(true),
            log_p: Some(false),
        }
    }
}

#[allow(dead_code)]
pub fn safe_dcauchy(args: Dcauchy) -> Result<f64, &'static str> {
    dcauchy(
        args.x.ok_or("argument \"x\" is missing, with no default")?,
        args.location
            .ok_or("argument \"location\" is missing, with no default")?,
        args.scale
            .ok_or("argument \"scale\" is missing, with no default")?,
        args.log
            .ok_or("argument \"log\" is missing, with no default")?,
    )
}

fn dcauchy(x: f64, location: f64, scale: f64, log: bool) -> Result<f64, &'static str> {
    let dist = Cauchy::new(location, scale).map_err(|_| "Error creating Cauchy distribution")?;

    match log {
        true => Ok(dist.pdf(x).ln()),
        false => Ok(dist.pdf(x)),
    }
}

pub fn pcauchy(
    q: f64,
    location: f64,
    scale: f64,
    lower_tail: bool,
    log_p: bool,
) -> Result<f64, &'static str> {
    let dist = Cauchy::new(location, scale).map_err(|_| "Error creating Cauchy distribution")?;
    match (lower_tail, log_p) {
        (true, true) => Ok(dist.cdf(q).ln()),
        (true, false) => Ok(dist.cdf(q)),
        (false, true) => Ok((1.0 - dist.cdf(q)).ln()),
        (false, false) => Ok(1.0 - dist.cdf(q)),
    }
}

#[allow(dead_code)]
pub fn safe_pcauchy(args: Pcauchy) -> Result<f64, &'static str> {
    pcauchy(
        args.q.ok_or("argument \"q\" is missing, with no default")?,
        args.location
            .ok_or("argument \"location\" is missing, with no default")?,
        args.scale
            .ok_or("argument \"scale\" is missing, with no default")?,
        args.lower_tail
            .ok_or("argument \"lower_tail\" is missing, with no default")?,
        args.log_p
            .ok_or("argument \"log_p\" is missing, with no default")?,
    )
}
