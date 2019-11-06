use approx::relative_eq;
use nalgebra as na;

use gauss_newton_rs::nalgebra_ver::{GaussNewton, Problem};

pub struct SampleProblem {
    x: na::DVector<f32>,
    s: na::DVector<f32>,
    v: na::DVector<f32>,
}

impl SampleProblem {
    fn new() -> Self {
        let x = na::DVector::from_vec(vec![1.5, 1.5]);
        let s = na::DVector::from_vec(vec![0.038, 0.194, 0.425, 0.626, 1.253, 2.500, 3.740]);
        let v = na::DVector::from_vec(vec![0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317]);
        Self { x, s, v }
    }
}

impl Problem for SampleProblem {
    type T = f32;

    fn parameters(&self) -> &na::DVector<Self::T> {
        &self.x
    }

    fn update_parameters(&mut self, update_value: &na::DVector<Self::T>) {
        self.x += update_value;
    }

    fn error_vector(&self) -> na::DVector<Self::T> {
        &self.v - self.s.map(|ss| self.x[0] * ss / (self.x[1] + ss))
    }

    fn jacobian(&self) -> na::DMatrix<Self::T> {
        let c1 = self.s.map(|ss| -ss / (self.x[1] + ss));
        let c2 = self.s.map(|ss| self.x[0] * ss / (self.x[1] + ss).powi(2));
        let jacobian = na::DMatrix::<Self::T>::from_columns(&[c1, c2]);
        jacobian
    }
}

#[test]
fn test_gauss_newton_nalgebra() {
    let problem = SampleProblem::new();
    let begin_squared_error = problem.squared_error();

    let mut gauss_newton = GaussNewton::new(problem);
    gauss_newton.solve_until_converged();

    let end_squared_error = gauss_newton.problem.squared_error();

    println!("{} iter(s)", gauss_newton.iters());
    assert!(begin_squared_error > end_squared_error);
    assert!(0.1 > end_squared_error);
    assert!(relative_eq!(
        &gauss_newton.problem.x,
        &na::DVector::from_vec(vec![0.362, 0.556]),
        epsilon = 1e-3
    ));
}
