use num_traits::Signed;

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
    P::T: From<f32> + Signed,
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
        let A = &J.transpose() * &J;
        let cholesky = A.cholesky().unwrap();
        let mut a = -&J.transpose() * self.problem.error_vector();
        cholesky.solve_mut(&mut a);
        self.problem.update_params(&a);

        let squared_error = self.problem.squared_error();
        let delta_squared_error = (squared_error - prev_squared_error).abs();
        if delta_squared_error / squared_error < self.eps {
            self.converged = true;
            return;
        }
        let delta_x_norm = a.norm();
        let x_norm = self.problem.params().norm();
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
