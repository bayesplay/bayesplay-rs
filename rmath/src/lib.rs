pub mod distributions;
pub mod integration;

pub use distributions::beta::lbeta;

#[macro_export]
macro_rules! dbeta {
    ($($a:tt = $c:expr ),*) => {
   $crate::distributions::beta::safe_dbeta(
       #[allow(clippy::needless_update)]

       $crate::distributions::beta::Dbeta {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! pbeta {
    ($($a:tt = $c:expr ),*) => {
   $crate::distributions::beta::safe_pbeta(
       #[allow(clippy::needless_update)]

       $crate::distributions::beta::Pbeta {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! dchisq {
    ($($a:tt = $c:expr ),*) => {
    $crate::distributions::chi_squared::safe_dchisq(
       #[allow(clippy::needless_update)]

       $crate::distributions::chi_squared::Dchisq {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! dt {
    ($($a:tt = $c:expr ),*) => {
   $crate::distributions::t::safe_dt(
       #[allow(clippy::needless_update)]

       $crate::distributions::t::Dt {
           $($a: Some($c),)*
        ..Default::default()
       })
    }

}

#[macro_export]
macro_rules! pt {
    ($($a:tt = $c:expr ),*) => {
    $crate::distributions::t::safe_pt(
       #[allow(clippy::needless_update)]

       $crate::distributions::t::Pt {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! dt_scaled {
    ($($a:tt = $c:expr ),*) => {
    $crate::distributions::t::dt_scaled(
       #[allow(clippy::needless_update)]

       $crate::distributions::t::DtScaled {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! pt_scaled {
    ($($a:tt = $c:expr ),*) => {
    $crate::distributions::t::pt_scaled(
       #[allow(clippy::needless_update)]

       $crate::distributions::t::PtScaled {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! dbinom {
    ($($a:tt = $c:expr ),*) => {
   $crate::distributions::binomial::safe_dbinom(
        #[allow(clippy::needless_update)]

        $crate::distributions::binomial::Dbinom {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! pbinom {
    ($($a:tt = $c:expr ),*) => {
   $crate::distributions::binomial::safe_pbinom(
        #[allow(clippy::needless_update)]

        $crate::distributions::binomial::Pbinom {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! dnorm {
    ($($a:tt = $c:expr ),*) => {
    $crate::distributions::normal::safe_dnorm(
       #[allow(clippy::needless_update)]

       $crate::distributions::normal::Dnorm {
        $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! pnorm {
    ($($a:tt = $c:expr ),*) => {
    $crate::distributions::normal::safe_pnorm(
       #[allow(clippy::needless_update)]

       $crate::distributions::normal::Pnorm {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! dcauchy {
    ($($a:tt = $c:expr ),*) => {
    $crate::distributions::cauchy::safe_dcauchy(
       #[allow(clippy::needless_update)]

       $crate::distributions::cauchy::Dcauchy {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! pcauchy {
    ($($a:tt = $c:expr ),*) => {
    $crate::distributions::cauchy::safe_pcauchy(
       #[allow(clippy::needless_update)]

       $crate::distributions::cauchy::Pcauchy {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! dunif {
    ($($a:tt = $c:expr ),*) => {
    $crate::distributions::uniform::safe_dunif(
       #[allow(clippy::needless_update)]

       $crate::distributions::uniform::Dunif {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! punif {
    ($($a:tt = $c:expr ),*) => {
    $crate::distributions::uniform::safe_punif(
       #[allow(clippy::needless_update)]

       $crate::distributions::uniform::Punif {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! dexp {
    ($($a:tt = $c:expr ),*) => {
    $crate::distributions::exponential::safe_dexp(
       #[allow(clippy::needless_update)]

       $crate::distributions::exponential::Dexp {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! pexp {
    ($($a:tt = $c:expr ),*) => {
    $crate::distributions::exponential::safe_pexp(
       #[allow(clippy::needless_update)]

       $crate::distributions::exponential::Pexp {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! integrate {
    ($($a:tt = $c:expr ),*) => {
    $crate::integration::integratewrapper(
       #[allow(clippy::needless_update)]

       $crate::integration::Integrate {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}
