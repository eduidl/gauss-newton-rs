use nalgebra as na;

pub trait Problem {
    type T: na::RealField;

    fn params(&self) -> &na::DVector<Self::T>;
    fn update_params(&mut self, delta_params: &na::DVector<Self::T>);
    fn error_vector(&self) -> na::DVector<Self::T>;
    fn squared_error(&self) -> Self::T {
        self.error_vector().norm()
    }
    fn jacobian(&self) -> na::DMatrix<Self::T>;
}
