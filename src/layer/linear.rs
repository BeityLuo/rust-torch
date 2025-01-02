use ndarray::{Array, Axis};
use ndarray::{ Array1, Array2, ArrayView2 };
use rand::prelude::*;
use rand_distr::StandardNormal;

use crate::layer::Propagable;

pub struct Linear {
    w: Array2<f64>,
    b: Array1<f64>,

    input: Option<Array2<f64>>,
    w_grad: Option<Array2<f64>>,
    b_grad: Option<Array1<f64>>,
}

impl Propagable for Linear {
    fn init(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = thread_rng();
        let v = (0..in_dim * out_dim)
                          .map(|_| rng.sample(StandardNormal))
                          .collect::<Vec<f64>>();
        let weights = Array::from_shape_vec((in_dim, out_dim), v).unwrap();
        let weights = weights / (in_dim as f64).sqrt();

        let v = (0..out_dim)
                          .map(|_| rng.sample(StandardNormal))
                          .collect::<Vec<f64>>();
        let bias = Array::from_vec(v);


        return Linear { w: weights, b: bias, input: None, 
                        w_grad: None, b_grad: None };
    }
    fn forward(&mut self, x: &ArrayView2<f64>) -> Array2<f64>{
        assert_eq!(x.shape()[1], self.w.shape()[0]);
        self.input = Some(x.to_owned());
        return x.dot(&self.w) + &self.b;
    }

    fn backward(&mut self, output_grad: &ArrayView2<f64>) -> Array2<f64>{
        let batch_size = output_grad.shape()[0];
        
        self.w_grad = Some(self.input.as_ref().unwrap().t()
                      .dot(output_grad) / batch_size as f64);
        self.b_grad = Some(output_grad.sum_axis(Axis(0)));
        return output_grad.dot(&self.w.t()) / self.w.shape()[1] as f64;
    }

    fn step(&mut self, lr: f64) {
        self.w = &self.w - self.w_grad.as_ref().unwrap() * lr;
        self.b = &self.b - self.b_grad.as_ref().unwrap() * lr;
    }   
}