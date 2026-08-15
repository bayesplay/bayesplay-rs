use crate::distributions::normal::dnorm;

use statrs::distribution::{Beta, Normal, StudentsT};
use statrs::distribution::{Continuous, ContinuousCDF};

use statrs::function::gamma::ln_gamma;

const M_LN_SQRT_PI: f64 = 0.572_364_942_924_700_1;
const DBL_MIN_EXP: f64 = -1021.0;
const LN_2: f64 = core::f64::consts::LN_2;
const M_SQRT_2D_PI: f64 = 0.797_884_560_802_865_4;
const DBL_EPSILON: f64 = 2.7755575615628914e-17;

pub fn dnt(x: f64, df: f64, ncp: f64, give_log: bool) -> Result<f64, &'static str> {
    if x.is_nan() || df.is_nan() || ncp.is_nan() {
        return Ok(x + df + ncp);
    }

    if df <= 0.0 {
        return Ok(f64::NAN);
    };

    if ncp == 0.0 {
        return dt(x, df, give_log);
    }

    if !x.is_finite() {
        return Ok(r_d_0(give_log));
    };

    if !df.is_finite() || df > 1e8 {
        return dnorm(x, ncp, 1., give_log);
    }

    let u = match x.abs() > (df * DBL_EPSILON).sqrt() {
        true => {
            df.ln() - (x.abs()).ln()
                + ((pnt(x * ((df + 2.) / df).sqrt(), df + 2., ncp, true, false)?
                    - pnt(x, df, ncp, true, false)?)
                .abs())
                .ln()
        }
        false => {
            ln_gamma((df + 1.) / 2.)
                - ln_gamma(df / 2.)
                - (M_LN_SQRT_PI + 0.5 * (df.ln() + ncp * ncp))
        }
    };

    if give_log {
        Ok(u)
    } else {
        Ok(u.exp())
    }
}

pub fn pnt(q: f64, df: f64, ncp: f64, lower_tail: bool, log_p: bool) -> Result<f64, &'static str> {
    let itrmax = 1000;
    let errmax = 1e-12;

    let mut tnc = 0.0;

    if df <= 0.0 {
        return Ok(f64::NAN);
    }

    if ncp == 0.0 {
        return pt(q, df, lower_tail, log_p);
    };

    if q.is_infinite() {
        return Ok(f64::NAN);
    }

    let negdel = q.is_sign_negative();
    let tt = q.abs();

    let del = match q.is_sign_positive() {
        true => ncp,
        false => -ncp,
    };

    if q < 0.0 && ncp > 40.0 && (!log_p || !lower_tail) {
        return Ok(r_dt_0(lower_tail, log_p));
    }

    if df > 4e5 || del * del > 2.0 * LN_2 * -1.0 * DBL_MIN_EXP {
        let _s = 1.0 / (4.0 * df);

        let pnorm = Normal::new(del, (1.0 + tt * tt * 2.0 * _s).sqrt())
            .map_err(|_| "Error creating Normal distribution")?;

        match log_p {
            true => return Ok(pnorm.cdf(tt * (1.0 - _s)).ln()),
            false => return Ok(pnorm.cdf(tt * (1.0 - _s))),
        }
    }

    let mut x = q * q;

    let mut rxb = df / (x + df);
    x = x / (x + df);

    if x > 0.0 {
        let lambda = del * del;
        let mut p = 0.5 * (-0.5 * lambda).exp();

        if p == 0.0 {
            return Ok(r_dt_0(lower_tail, log_p));
        }

        let mut q = M_SQRT_2D_PI * p * del;
        let mut s = 0.5 - p;
        if s < 1e-7 {
            s = -0.5 * (-0.5 * lambda).exp_m1()
        }

        let mut a = 0.5;
        let b = 0.5 * df;
        rxb = rxb.powf(b);

        let albeta = M_LN_SQRT_PI + ln_gamma(b) - ln_gamma(0.5 + b);

        let mut xodd = Beta::new(a, b)
            .map_err(|_| "Error creating Beta distribution")?
            .cdf(x);

        let mut godd = 2. * rxb * (a * x.ln() - albeta).exp();

        tnc = b * x;

        let mut xeven = if tnc < DBL_EPSILON { tnc } else { 1.0 - rxb };

        let mut geven = tnc * rxb;
        tnc = p * xodd + q * xeven;

        let mut goto_finish = false;

        for it in 1..=itrmax {
            a += 1.0;
            xodd -= godd;
            xeven -= geven;
            godd *= x * (a + b - 1.0) / a;
            geven *= x * (a + b - 0.5) / (a + 0.5);
            p *= lambda / (2 * it) as f64;
            q *= lambda / (2 * it + 1) as f64;
            tnc += p * xodd + q * xeven;
            s -= p;

            if s < -1e-10 {
                goto_finish = true;
                break;
            }

            if s <= 0.0 {
                goto_finish = true;
                break;
            }

            let errbd = 2.0 * s * (xodd - godd);

            if errbd.abs() < errmax {
                goto_finish = true;
                break;
            }

            goto_finish = false
        }

        if !goto_finish {
            return Ok(f64::NAN);
        }
    }

    let pnorm = Normal::new(0.0, 1.0).map_err(|_| "Error creating Normal distribution")?;

    tnc += pnorm.cdf(-del);

    let lower_tail = lower_tail != negdel;

    let x = tnc.min(1.0);
    match lower_tail {
        true => Ok(r_d_val(log_p, x)),
        false => Ok(r_d_clog(log_p, x)),
    }
}

fn dt(x: f64, df: f64, log_p: bool) -> Result<f64, &'static str> {
    let dist = StudentsT::new(0.0, 1.0, df).map_err(|_| "Error creating Student t distribution")?;
    match log_p {
        true => Ok(dist.ln_pdf(x)),
        false => Ok(dist.pdf(x)),
    }
}

fn pt(q: f64, df: f64, lower_tail: bool, log_p: bool) -> Result<f64, &'static str> {
    let dist = StudentsT::new(0.0, 1.0, df).map_err(|_| "Error creating Student t distribution")?;
    let sf = dist.sf(q);
    let cdf = dist.cdf(q);
    match (lower_tail, log_p) {
        // Use ln_1p(-sf) instead of cdf.ln() to avoid catastrophic cancellation
        // when cdf ≈ 1 (large positive q), where cdf = 1 - sf and sf is tiny.
        (true, true) => Ok((-sf).ln_1p()),
        (true, false) => Ok(cdf),
        // Use sf() (survival function) directly instead of 1 - cdf() to avoid
        // catastrophic cancellation when cdf(q) is very close to 1 (large q).
        (false, true) => Ok(sf.ln()),
        (false, false) => Ok(sf),
    }
}

fn r_dt_0(lower_tail: bool, log_p: bool) -> f64 {
    match lower_tail {
        true => r_d_0(log_p),
        false => r_d_1(log_p),
    }
}

fn r_d_0(log_p: bool) -> f64 {
    match log_p {
        true => f64::NEG_INFINITY,
        false => 0.0,
    }
}

fn r_d_1(log_p: bool) -> f64 {
    match log_p {
        true => 0.0,
        false => 1.0,
    }
}

fn r_d_val(log_p: bool, x: f64) -> f64 {
    match log_p {
        true => x.ln(),
        false => x,
    }
}

fn r_d_clog(log_p: bool, p: f64) -> f64 {
    match log_p {
        true => (-p).ln_1p(),
        false => 0.5 - p + 0.5,
    }
}

pub struct Dt {
    pub x: Option<f64>,
    pub df: Option<f64>,
    pub ncp: Option<f64>,
    pub log: Option<bool>,
}

impl Default for Dt {
    fn default() -> Self {
        Dt {
            x: None,
            df: None,
            ncp: None,
            log: Some(false),
        }
    }
}

pub struct Pt {
    pub q: Option<f64>,
    pub df: Option<f64>,
    pub ncp: Option<f64>,
    pub lower_tail: Option<bool>,
    pub log_p: Option<bool>,
}

impl Default for Pt {
    fn default() -> Self {
        Pt {
            q: None,
            df: None,
            ncp: None,
            lower_tail: Some(true),
            log_p: Some(false),
        }
    }
}

pub struct DtScaled {
    pub x: Option<f64>,
    pub df: Option<f64>,
    pub mean: Option<f64>,
    pub sd: Option<f64>,
    pub ncp: Option<f64>,
    pub log: Option<bool>,
}

impl Default for DtScaled {
    fn default() -> Self {
        DtScaled {
            x: None,
            df: None,
            mean: Some(0.0),
            sd: Some(1.0),
            ncp: None,
            log: Some(false),
        }
    }
}

pub struct PtScaled {
    pub q: Option<f64>,
    pub df: Option<f64>,
    pub mean: Option<f64>,
    pub sd: Option<f64>,
    pub ncp: Option<f64>,
    pub lower_tail: Option<bool>,
    pub log_p: Option<bool>,
}

impl Default for PtScaled {
    fn default() -> Self {
        PtScaled {
            q: None,
            df: None,
            mean: Some(0.0),
            sd: Some(1.0),
            ncp: None,
            lower_tail: Some(true),
            log_p: Some(false),
        }
    }
}

pub fn pt_scaled(args: PtScaled) -> Result<f64, &'static str> {
    let mean = args
        .mean
        .ok_or("argument \"mean\" is missing, with no default")?;
    let sd = args
        .sd
        .ok_or("argument \"sd\" is missing, with no default")?;
    let df = args
        .df
        .ok_or("argument \"df\" is missing, with no default")?;
    let q = args.q.expect("argument \"q\" is missing, with no default");
    let q = (q - mean) / sd;
    let lower_tail = args
        .lower_tail
        .ok_or("argument \"lower_tail\" is missing, with no default")?;
    let log_p = args
        .log_p
        .ok_or("argument \"log_p\" is missing, with no default")?;
    pt(q, df, lower_tail, log_p)
}

pub fn dt_scaled(args: DtScaled) -> Result<f64, &'static str> {
    let mean = args
        .mean
        .ok_or("argument \"mean\" is missing, with no default")?;
    let sd = args
        .sd
        .ok_or("argument \"sd\" is missing, with no default")?;
    let df = args
        .df
        .ok_or("argument \"df\" is missing, with no default")?;
    let x = args.x.ok_or("argument \"x\" is missing, with no default")?;
    let x = (x - mean) / sd;
    let log = args
        .log
        .ok_or("argument \"log\" is missing, with no default")?;
    Ok(dt(x, df, log)? / sd)
}

pub fn safe_dt(args: Dt) -> Result<f64, &'static str> {
    match args.ncp {
        None => dt(
            args.x.ok_or("argument \"x\" is missing, with no default")?,
            args.df
                .ok_or("argument \"df\" is missing, with no default")?,
            args.log
                .ok_or("argument \"log\" is missing, with no default")?,
        ),

        Some(ncp) => dnt(
            args.x.expect("argument \"x\" is missing, with no default"),
            args.df
                .expect("argument \"df\" is missing, with no default"),
            ncp,
            args.log
                .expect("argument \"log\" is missing, with no default"),
        ),
    }
}

pub fn safe_pt(args: Pt) -> Result<f64, &'static str> {
    match args.ncp {
        None => pt(
            args.q.ok_or("argument \"q\" is missing, with no default")?,
            args.df
                .ok_or("argument \"df\" is missing, with no default")?,
            args.lower_tail
                .ok_or("argument \"lower_tail\" is missing, with no default")?,
            args.log_p
                .ok_or("argument \"log_p\" is missing, with no default")?,
        ),

        Some(ncp) => pnt(
            args.q.ok_or("argument \"q\" is missing, with no default")?,
            args.df
                .ok_or("argument \"df\" is missing, with no default")?,
            ncp,
            args.lower_tail
                .ok_or("argument \"lower_tail\" is missing, with no default")?,
            args.log_p
                .ok_or("argument \"log_p\" is missing, with no default")?,
        ),
    }
}

// write some tests
#[cfg(test)]
mod tests {

    use super::*;
    use approx_eq::assert_approx_eq;
    #[test]

    fn test_dt() {
        let x = 1.0;
        let df = 2.0;
        let want = dt(x, df, false).unwrap();
        let got = statrs::distribution::StudentsT::new(0.0, 1.0, df)
            .unwrap()
            .pdf(x);
        assert_approx_eq!(want, got);
    }

    #[test]
    fn test_pt() {
        let x = 1.0;
        let df = 2.0;
        let lower_tail = true;
        let want = pt(x, df, lower_tail, false).unwrap();
        let got = statrs::distribution::StudentsT::new(0.0, 1.0, df)
            .unwrap()
            .cdf(x);
        assert_approx_eq!(want, got);
    }

    #[test]
    fn test_pt2() {
        let x = 1.0;
        let df = 5.0;
        let lower_tail = false;
        let want = pt(x, df, lower_tail, false).unwrap();
        let got = 1.0
            - statrs::distribution::StudentsT::new(0.0, 1.0, df)
                .unwrap()
                .cdf(x);
        println!("want: {}, got: {}", want, got);
        assert_approx_eq!(want, got);
    }
}
