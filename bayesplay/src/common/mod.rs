pub trait Validate<E> {
    fn validate(&self) -> Result<(), E>;
}

pub trait Function<T, S, E> {
    fn function(&self, x: T) -> Result<S, E>;
}

pub trait Range {
    fn range(&self) -> (Option<f64>, Option<f64>);
    fn default_range(&self) -> (f64, f64);

    fn range_or_default(&self) -> (f64, f64) {
        let (ll, ul) = self.range();
        let (default_ll, default_ul) = self.default_range();

        (ll.unwrap_or(default_ll), ul.unwrap_or(default_ul))
    }
    fn has_default_range(&self) -> bool {
        let (ll, ul) = self.range_or_default();
        let (default_ll, default_ul) = self.default_range();
        ll == default_ll && ul == default_ul
    }
}

pub trait Integrate<E, O>: Function<f64, f64, O> + Range {
    fn integral(&self) -> Result<f64, E>;
    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> Result<f64, E>;
}
