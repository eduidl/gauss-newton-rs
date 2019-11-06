use std::convert::From;

use ndarray::{Array1, Array2};
use ndarray_linalg::{Norm, SolveC};
use num_traits::Float;

use super::problem::Problem;

pub struct GaussNewton<P, T> {
    pub problem: P,
    converged: bool,
    iters: usize,
    eps: T,
}

#[allow(non_snake_case)]
impl<P: Problem> GaussNewton<P, P::T>
where
    P::T: Float + From<f32>,
    Array1<P::T>: Norm<Output = P::T>,
    Array2<P::T>: Norm<Output = P::T> + SolveC<P::T>,
{
    pub fn new(problem: P) -> Self {
        Self {
            problem,
            converged: false,
            iters: 0,
            eps: 1e-6.into(),
        }
    }

    pub fn step(&mut self) {
        assert!(!self.converged);
        self.iters += 1;

        let prev_squared_error = self.problem.squared_error();

        let J = self.problem.jacobian();
        let A = J.t().dot(&J);
        let mut a = -J.t().dot(&self.problem.error_vector());
        // TODO: error handling
        let _ = A.solvec_inplace(&mut a).unwrap();
        self.problem.update_parameters(&a);

        let squared_error = self.problem.squared_error();
        let delta_squared_error = (squared_error - prev_squared_error).abs();
        if delta_squared_error / squared_error < self.eps {
            self.converged = true;
            return;
        }
        let delta_x_norm = a.norm_l2();
        let x_norm = self.problem.parameters().norm_l2();
        if delta_x_norm / x_norm < self.eps {
            self.converged = true;
        }
    }

    pub fn solve_until_converged(&mut self) {
        while !self.converged() {
            self.step()
        }
    }

    pub fn iters(&self) -> usize {
        self.iters
    }

    pub fn converged(&self) -> bool {
        self.converged
    }
}
