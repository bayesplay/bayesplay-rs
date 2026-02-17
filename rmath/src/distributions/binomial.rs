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

pub fn dbinom(x: f64, size: f64, prob: f64, log: bool) -> Result<f64, &'static str> {
    let dist = statrs::distribution::Binomial::new(prob, size as u64)
        .map_err(|_| "Error creating Binomial distribution")?;

    match log {
        true => Ok(dist.ln_pmf(x as u64)),
        false => Ok(dist.pmf(x as u64)),
    }
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
    let dist = statrs::distribution::Binomial::new(prob, size as u64)
        .map_err(|_| "Error creating Binomial distribution")?;

    match (lower_tail, log_p) {
        (true, true) => Ok(dist.cdf(x as u64).ln()),
        (true, false) => Ok(dist.cdf(x as u64)),
        (false, true) => Ok((1.0 - dist.cdf(x as u64)).ln()),
        (false, false) => Ok(1.0 - dist.cdf(x as u64)),
    }
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
