pub mod integration;
pub mod distributions;


#[macro_export]
macro_rules! dbeta {
    ($($a:tt = $c:expr ),*) => {
   $crate::distributions::beta::safe_dbeta(
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
       $crate::distributions::beta::Pbeta {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}

#[macro_export]
macro_rules! dt {
    ($($a:tt = $c:expr ),*) => {
   $crate::distributions::t::safe_dt(
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
       $crate::distributions::uniform::Punif {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}


#[macro_export]
macro_rules! integrate {
    ($($a:tt = $c:expr ),*) => {
    $crate::integration::integratewrapper(
       $crate::integration::Integrate {
           $($a: Some($c),)*
        ..Default::default()
       })
    }
}
