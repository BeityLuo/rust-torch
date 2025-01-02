use ndarray::{ Array2, ArrayView2 };


use crate::layer::Propagable;

pub struct Sigmoid {
    input: Option<Array2<f64>>,
}
impl Propagable for Sigmoid {
    fn init(_in_dim: usize, _out_dim: usize) -> Self {
        assert_eq!(_in_dim, _out_dim);
        return Sigmoid { input: None };
    }

    fn forward(&mut self, x: &ArrayView2<f64>) -> Array2<f64> {
        self.input = Some(x.to_owned());
        return x.map(|e| 1.0 / (1.0 + (-1.0 * e).exp()));
    }

    fn backward(&mut self, output_grad: &ArrayView2<f64>) -> Array2<f64> {
        let e: Array2<f64> = self.input.as_ref().unwrap()
                             .map(|x| (-x)).exp();
        return output_grad * e.map(|e| e / ((1.0 + e) * (1.0 + e)));
    }

    fn step(&mut self, _lr: f64) { }
}
