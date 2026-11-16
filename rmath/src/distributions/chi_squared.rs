use crate::error::{Distribution, Error, Result, require_positive};
use statrs::distribution::{ChiSquared, Continuous};

pub fn dchisq(x: f64, df: f64, log: bool) -> Result<f64> {
    require_positive(Distribution::ChiSquared, "df", df)?;
    let dist = ChiSquared::new(df).map_err(|_| Error::InvalidParameter {
        distribution: Distribution::ChiSquared,
        parameter: "df".into(),
        value: df,
        requirement: "rejected by statrs".into(),
    })?;
    let log_density = dist.ln_pdf(x);
    Ok(if log { log_density } else { log_density.exp() })
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
