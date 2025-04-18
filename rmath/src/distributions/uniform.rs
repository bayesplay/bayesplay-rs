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

pub fn punif(x: f64, min: f64, max: f64, lower_tail: bool, log_p: bool) -> f64 {
    let dist = statrs::distribution::Uniform::new(min, max).unwrap();

    match (lower_tail, log_p) {
        (true, true) => (dist.cdf(x)).ln(),
        (true, false) => dist.cdf(x),
        (false, true) => (1.0 - dist.cdf(x)).ln(),
        (false, false) => 1.0 - dist.cdf(x),
    }
}

pub fn dunif(x: f64, mean: f64, sd: f64, log: bool) -> f64 {
    let dist = statrs::distribution::Uniform::new(mean, sd).unwrap();

    match log {
        true => dist.pdf(x).ln(),
        false => dist.pdf(x),
    }
}

#[allow(dead_code)]
pub fn safe_dunif(args: Dunif) -> f64 {
    dunif(
        args.x.expect("argument \"x\" is missing, with no default"),
        args.min
            .expect("argument \"min\" is missing, with no default"),
        args.max
            .expect("argument \"max\" is missing, with no default"),
        args.log
            .expect("argument \"log\" is missing, with no default"),
    )
}

#[allow(dead_code)]
pub fn safe_punif(args: Punif) -> f64 {
    punif(
        args.q.expect("argument \"q\" is missing, with no default"),
        args.min
            .expect("argument \"min\" is missing, with no default"),
        args.max
            .expect("argument \"max\" is missing, with no default"),
        args.lower_tail
            .expect("argument \"lower_tail\" is missing, with no default"),
        args.log_p
            .expect("argument \"give_log\" is missing, with no default"),
    )
}

