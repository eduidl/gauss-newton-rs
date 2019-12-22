use ndarray::{Array1, Array2};
use ndarray_linalg::{Norm, Scalar};

pub trait Problem
where
    Array1<Self::T>: Norm<Output = Self::T>,
{
    type T: Scalar;

    fn params(&self) -> &Array1<Self::T>;
    fn update_params(&mut self, delta_param: &Array1<Self::T>);
    fn error_vector(&self) -> Array1<Self::T>;
    fn squared_error(&self) -> Self::T {
        self.error_vector().norm_l2()
    }
    fn jacobian(&self) -> Array2<Self::T>;
}
