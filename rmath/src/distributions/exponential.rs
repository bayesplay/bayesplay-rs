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
    if q < 0.0 {
        return match (lower_tail, log_p) {
            (true, true) => Ok(f64::NEG_INFINITY),
            (true, false) => Ok(0.0),
            (false, true) => Ok(0.0),
            (false, false) => Ok(1.0),
        };
    }

    // Use -expm1(-rate*q) instead of 1 - exp(-rate*q) to avoid catastrophic
    // cancellation when rate*q is very small (CDF ≈ 0) or very large (CDF ≈ 1).
    // -expm1(-x) = -(exp(-x) - 1) = 1 - exp(-x), computed accurately by expm1.
    let neg_rate_q = -(rate * q);
    let cdf = -neg_rate_q.exp_m1(); // = 1 - exp(-rate*q), accurate for all q
    let sf = neg_rate_q.exp(); // = exp(-rate*q) = 1 - CDF

    match (lower_tail, log_p) {
        // log(CDF) = log(1 - exp(-rate*q)) = log(-expm1(-rate*q))
        (true, true) => Ok(cdf.ln()),
        (true, false) => Ok(cdf),
        // log(SF) = log(exp(-rate*q)) = -rate*q
        (false, true) => Ok(sf.ln()),
        (false, false) => Ok(sf),
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
