use ndarray::{arr1, stack, Array1, Array2, Axis};

use gauss_newton_rs::ndarray_ver::{GaussNewton, Problem};

struct SampleProblem {
    x: Array1<f32>,
    s: Array1<f32>,
    v: Array1<f32>,
}

impl SampleProblem {
    fn new() -> Self {
        let x = arr1(&[1.5, 1.5]);
        let s = arr1(&[0.038, 0.194, 0.425, 0.626, 1.253, 2.500, 3.740]);
        let v = arr1(&[0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317]);
        Self { x, s, v }
    }
}

impl Problem for SampleProblem {
    type T = f32;

    fn params(&self) -> &Array1<Self::T> {
        &self.x
    }

    fn update_params(&mut self, delta_params: &Array1<Self::T>) {
        self.x += delta_params;
    }

    fn error_vector(&self) -> Array1<Self::T> {
        let predicted = &self.s * self.x[0] / (&self.s + self.x[1]);
        &self.v - &predicted
    }

    fn jacobian(&self) -> Array2<Self::T> {
        let c1 = -&self.s / (&self.s + self.x[1]);
        let c2 = &self.s * self.x[0] / (&self.s + self.x[1]).mapv(|a| a.powi(2));
        stack![Axis(1), c1, c2]
    }
}

#[test]
fn test_gauss_newton_ndarray() {
    let problem = SampleProblem::new();
    let begin_squared_error = problem.squared_error();

    let mut gauss_newton = GaussNewton::new(problem);
    gauss_newton.solve_until_converged();

    let end_squared_error = gauss_newton.problem.squared_error();

    println!("{} iter(s)", gauss_newton.iters());
    assert!(begin_squared_error > end_squared_error);
    assert!(0.1 > end_squared_error);
    assert!(gauss_newton
        .problem
        .x
        .abs_diff_eq(&arr1(&[0.362, 0.556]), 1e-3));
}
