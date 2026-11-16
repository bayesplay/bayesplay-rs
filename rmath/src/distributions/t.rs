use crate::distributions::normal::{dnorm, pnorm};

use statrs::distribution::{Beta, Normal, StudentsT};
use statrs::distribution::{Continuous, ContinuousCDF};

use statrs::function::gamma::ln_gamma;

const M_LN_SQRT_PI: f64 = 0.572_364_942_924_700_1;
const DBL_MIN_EXP: f64 = -1021.0;
const LN_2: f64 = core::f64::consts::LN_2;
const M_SQRT_2D_PI: f64 = 0.797_884_560_802_865_4;
const DBL_EPSILON: f64 = f64::EPSILON; // 2.22e-16, as in R's C headers

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

    // Where pnt() falls back to its normal approximation, the pnt() difference
    // below is not a density at all (e.g. log dnt(30, 19, 45) is off by 0.27).
    if pnt_uses_normal_approximation(df, ncp)
        && let Some(accurate) = dnt_by_integration(x, df, ncp)
    {
        return Ok(if give_log { accurate } else { accurate.exp() });
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

    // The pnt() difference above cancels catastrophically when the density is tiny
    // (e.g. x and ncp of opposite sign, far tails): pnt() carries ~1e-13 absolute
    // error, which R reports as "full precision may not have been achieved".
    // Below the threshold use a positive-integrand representation instead.
    if !(u.is_finite() && u > DNT_CANCELLATION_LOG_THRESHOLD) {
        if let Some(accurate) = dnt_by_integration(x, df, ncp) {
            return Ok(if give_log { accurate } else { accurate.exp() });
        }
    }

    if give_log { Ok(u) } else { Ok(u.exp()) }
}

/// ln(1e-3): densities below this are recomputed by integration. pnt() is good to
/// ~1e-13 absolute, so above 1e-3 the relative error is <= 1e-10, below the
/// quadrature tolerances used downstream (a seam at a lower threshold shows up
/// as a roundoff error in the integrator).
const DNT_CANCELLATION_LOG_THRESHOLD: f64 = -6.907_755_278_982_137;

/// log of the noncentral-t density via
///   f(x) = integral_0^inf  s * phi(x s - ncp) * h(s) ds,
/// where s = sqrt(V/df), V ~ chi^2_df has density h(s) (so T = Z'/s, Z' ~ N(ncp, 1)).
/// Every term is positive, so there is no cancellation. The integrand is scaled
/// by its (closed-form) mode so deep tails do not underflow.
fn dnt_by_integration(x: f64, df: f64, ncp: f64) -> Option<f64> {
    use crate::integration::{Integrate, integratewrapper};

    let half_df = df / 2.0;
    let ln_h_const = LN_2 + half_df * half_df.ln() - ln_gamma(half_df);
    let ln_norm = 0.5 * (2.0 * core::f64::consts::PI).ln();
    let log_integrand = move |s: f64| -> f64 {
        let z = x * s - ncp;
        s.ln() - 0.5 * z * z - ln_norm + ln_h_const + (df - 1.0) * s.ln() - df * s * s / 2.0
    };

    // d/ds log integrand = 0  <=>  (df + x^2) s^2 - x*ncp*s - df = 0
    let a = df + x * x;
    let b = x * ncp;
    let mode = (b + (b * b + 4.0 * df * a).sqrt()) / (2.0 * a);
    if !(mode.is_finite() && mode > 0.0) {
        return None;
    }
    let peak = log_integrand(mode);
    let width = 1.0 / (a + df).sqrt(); // scale of the peak (curvature >= a + df)
    let upper = mode + 40.0 * width;
    let result = integratewrapper(Integrate {
        f: Some(move |s: f64| if s <= 0.0 { 0.0 } else { (log_integrand(s) - peak).exp() }),
        lower: Some((mode - 40.0 * width).max(0.0)),
        upper: Some(upper),
        points: Some(vec![mode]),
        // integrand is scaled to a peak of 1, so abs_tol is effectively relative too;
        // a purely relative tolerance trips the roundoff detector on the ~0 tails
        abs_tol: Some(1e-13),
        rel_tol: Some(1e-10),
        ..Default::default()
    })
    .ok()?;
    (result.value > 0.0 && result.value.is_finite()).then(|| peak + result.value.ln())
}

/// pnt() abandons its series for a normal approximation when df is huge or the
/// Poisson weights exp(-ncp^2/2) would underflow (|ncp| > ~37.6).
fn pnt_uses_normal_approximation(df: f64, ncp: f64) -> bool {
    df > 4e5 || ncp * ncp > -2.0 * LN_2 * DBL_MIN_EXP
}

pub fn pnt(q: f64, df: f64, ncp: f64, lower_tail: bool, log_p: bool) -> Result<f64, &'static str> {
    let itrmax = 1000;
    let errmax = 1e-12;

    let mut tnc = 0.0;

    if q.is_nan() || df.is_nan() || ncp.is_nan() {
        return Ok(f64::NAN);
    }

    if df <= 0.0 {
        return Ok(f64::NAN);
    }

    if ncp == 0.0 {
        return pt(q, df, lower_tail, log_p);
    };

    if q.is_infinite() {
        return Ok(r_dt_0(lower_tail == q.is_sign_negative(), log_p));
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

    if pnt_uses_normal_approximation(df, del) {
        let s = 1.0 / (4.0 * df);
        return pnorm(
            tt * (1.0 - s),
            del,
            1.0_f64.hypot(tt * (2.0 * s).sqrt()),
            lower_tail != negdel,
            log_p,
        );
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
        (true, true) => Ok(if q > 0.0 { (-sf).ln_1p() } else { cdf.ln() }),
        (true, false) => Ok(cdf),
        // Use sf() (survival function) directly instead of 1 - cdf() to avoid
        // catastrophic cancellation when cdf(q) is very close to 1 (large q).
        (false, true) => Ok(if q < 0.0 { (-cdf).ln_1p() } else { sf.ln() }),
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
    match args.ncp {
        Some(ncp) => pnt(q, df, ncp, lower_tail, log_p),
        None => pt(q, df, lower_tail, log_p),
    }
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
    let density = match args.ncp {
        Some(ncp) => dnt(x, df, ncp, log)?,
        None => dt(x, df, log)?,
    };
    Ok(if log { density - sd.ln() } else { density / sd })
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
    fn noncentral_density_far_tail_is_smooth_and_matches_integral_reference() {
        // Reference values from R: integrating the positive mixture representation
        // at high precision (R's own dt(ncp=) returns cancellation noise here:
        // 5.5e-13 and 9.0e-14 where the true values are 3.269e-13 and 2.067e-31).
        let df = 79.0;
        for (ncp_over_sqrt_n, expected) in [
            (0.2, 3.021158e-6),
            (0.5, 3.269087e-13),
            (1.0, 2.067437e-31),
            (2.0, 1.15111e-92),
        ] {
            let got = dnt(-3.2, df, ncp_over_sqrt_n * 80f64.sqrt(), false).unwrap();
            assert!((got / expected - 1.0).abs() < 1e-5, "ncp={ncp_over_sqrt_n}: {got:e} vs {expected:e}");
        }
        // log density decreases monotonically through the former noise floor
        let logs: Vec<f64> = (0..300)
            .map(|i| dnt(-3.2, df, i as f64 * 0.01 * 80f64.sqrt(), true).unwrap())
            .collect();
        assert!(logs.windows(2).all(|w| w[1] <= w[0] + 1e-9));
    }

    #[test]
    fn noncentral_density_is_accurate_across_pnt_normal_approximation_seam() {
        // log dt(30, 19, ncp) from the positive mixture representation in R
        // (integrate(), rel.tol = 1e-13). pnt() switches to a normal
        // approximation above ncp ~ 37.6, which the density must not inherit.
        for (ncp, expected) in [
            (5.0, -2.322_098_698_483e1),
            (30.0, -2.530_805_012_800),
            (35.0, -3.011_343_477_453),
            (38.0, -3.733_746_820_726),
            (40.0, -4.383_316_622_106),
            (45.0, -6.561_258_410_337),
        ] {
            let got = dnt(30.0, 19.0, ncp, true).unwrap();
            assert!((got - expected).abs() < 1e-9, "ncp={ncp}: {got} vs {expected}");
        }
    }

    #[test]
    fn scaled_helpers_preserve_noncentrality_and_log_jacobian() {
        for ncp in [None, Some(0.0), Some(2.0)] {
            let expected = dnt(0.5, 7.0, ncp.unwrap_or(0.0), false).unwrap() / 3.0;
            for log in [false, true] {
                let got = dt_scaled(DtScaled {
                    x: Some(5.5),
                    df: Some(7.0),
                    mean: Some(4.0),
                    sd: Some(3.0),
                    ncp,
                    log: Some(log),
                })
                .unwrap();
                let want = if log { expected.ln() } else { expected };
                assert!((got - want).abs() < 1e-12);
            }
            for lower_tail in [false, true] {
                for log_p in [false, true] {
                    let got = pt_scaled(PtScaled {
                        q: Some(5.5),
                        df: Some(7.0),
                        mean: Some(4.0),
                        sd: Some(3.0),
                        ncp,
                        lower_tail: Some(lower_tail),
                        log_p: Some(log_p),
                    })
                    .unwrap();
                    let want = pnt(0.5, 7.0, ncp.unwrap_or(0.0), lower_tail, log_p).unwrap();
                    assert_eq!(got, want);
                }
            }
        }
    }

    #[test]
    fn noncentral_approximation_respects_reflection_and_tail() {
        for (df, ncp, q) in [(1e6, 2.0, 1.0), (10.0, 40.0, 39.0)] {
            for sign in [-1.0, 1.0] {
                for lower_tail in [false, true] {
                    for log_p in [false, true] {
                        let s = 1.0 / (4.0 * df);
                        let want = pnorm(
                            sign * q * (1.0 - s),
                            sign * ncp,
                            (1.0 + q * q * 2.0 * s).sqrt(),
                            lower_tail,
                            log_p,
                        )
                        .unwrap();
                        let got = pnt(sign * q, df, sign * ncp, lower_tail, log_p).unwrap();
                        assert!((got - want).abs() < 1e-12, "{got} != {want}");
                    }
                }
            }
        }
    }

    #[test]
    fn noncentral_infinite_quantiles_and_nan() {
        for df in [5.0, 1e6, f64::INFINITY] {
            for ncp in [-2.0, 0.0, 2.0] {
                for q in [f64::NEG_INFINITY, f64::INFINITY] {
                    for lower_tail in [false, true] {
                        let want: f64 = if lower_tail == (q > 0.0) { 1.0 } else { 0.0 };
                        assert_eq!(pnt(q, df, ncp, lower_tail, false).unwrap(), want);
                        assert_eq!(pnt(q, df, ncp, lower_tail, true).unwrap(), want.ln());
                    }
                }
            }
        }
        assert!(pnt(f64::NAN, 5.0, 2.0, true, false).unwrap().is_nan());
    }

    #[test]
    fn central_log_tails_preserve_small_probabilities() {
        let small = pt(-1e4, 5.0, true, false).unwrap();
        assert!(small > 0.0 && small < f64::EPSILON);
        assert_eq!(pt(-1e4, 5.0, true, true).unwrap(), small.ln());
        assert_eq!(pt(1e4, 5.0, false, true).unwrap(), small.ln());
        assert_eq!(pt(1e4, 5.0, true, true).unwrap(), (-small).ln_1p());
        assert_eq!(pt(-1e4, 5.0, false, true).unwrap(), (-small).ln_1p());
    }

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
