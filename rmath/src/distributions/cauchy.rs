use statrs::distribution::{Cauchy, Continuous};

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
        true => Ok(dist.ln_pdf(x)),
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
    Cauchy::new(location, scale).map_err(|_| "Error creating Cauchy distribution")?;
    let x = q - location;
    // atan(scale / |x|) computes the small tail without subtracting from 1/2.
    let small_tail = scale.atan2(x.abs()) / core::f64::consts::PI;
    match (lower_tail == (x < 0.0), log_p) {
        (true, true) => Ok(small_tail.ln()),
        (true, false) => Ok(small_tail),
        (false, true) => Ok((-small_tail).ln_1p()),
        (false, false) => Ok(1.0 - small_tail),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extreme_tails_and_truncation_mass() {
        let small = pcauchy(1e20, 0.0, 2.0, false, false).unwrap();
        let want = 2e-20 / core::f64::consts::PI;
        assert!((small / want - 1.0).abs() < 1e-14);
        assert_eq!(pcauchy(-1e20, 0.0, 2.0, true, false).unwrap(), small);
        assert_eq!(pcauchy(-1e20, 0.0, 2.0, true, true).unwrap(), small.ln());
        assert_eq!(pcauchy(1e20, 0.0, 2.0, false, true).unwrap(), small.ln());
        assert_eq!(
            pcauchy(1e20, 0.0, 2.0, true, true).unwrap(),
            (-small).ln_1p()
        );
        assert_eq!(
            pcauchy(-1e20, 0.0, 2.0, false, true).unwrap(),
            (-small).ln_1p()
        );
        let mass = small - pcauchy(2e20, 0.0, 2.0, false, false).unwrap();
        assert!((mass / (small / 2.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn center_and_endpoints() {
        for lower_tail in [false, true] {
            assert_eq!(pcauchy(3.0, 3.0, 2.0, lower_tail, false).unwrap(), 0.5);
            for q in [f64::NEG_INFINITY, f64::INFINITY] {
                let want: f64 = if lower_tail == (q > 0.0) { 1.0 } else { 0.0 };
                assert_eq!(pcauchy(q, 3.0, 2.0, lower_tail, false).unwrap(), want);
                assert_eq!(pcauchy(q, 3.0, 2.0, lower_tail, true).unwrap(), want.ln());
            }
        }
    }
}
