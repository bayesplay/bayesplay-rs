use statrs::distribution::{ChiSquared, Continuous};

pub struct Dchisq {
    pub x: Option<f64>,
    pub df: Option<f64>,
    pub log: Option<bool>,
}

impl Default for Dchisq {
    fn default() -> Self {
        Self {
            x: None,
            df: None,
            log: Some(false),
        }
    }
}

pub fn dchisq(x: f64, df: f64, log: bool) -> Result<f64, &'static str> {
    let dist = ChiSquared::new(df).map_err(|_| "Error creating Chi-squared distribution")?;
    let log_density = dist.ln_pdf(x);
    Ok(if log { log_density } else { log_density.exp() })
}

pub fn safe_dchisq(args: Dchisq) -> Result<f64, &'static str> {
    dchisq(
        args.x.ok_or("argument \"x\" is missing, with no default")?,
        args.df
            .ok_or("argument \"df\" is missing, with no default")?,
        args.log
            .ok_or("argument \"log\" is missing, with no default")?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_matches_known_value() {
        assert!((dchisq(4.0, 3.0, false).unwrap() - 0.107_981_933_026_376_14).abs() < 1e-15);
        assert_eq!(dchisq(-1.0, 3.0, false).unwrap(), 0.0);
        assert!((dchisq(300.0, 300.0, false).unwrap() - 0.016_277_704_728_418_278).abs() < 1e-14);
        assert!((dchisq(5.0, 2.0, true).unwrap() + 3.193_147_180_559_945_4).abs() < 1e-15);
    }
}
