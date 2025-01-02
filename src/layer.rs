mod linear;
mod activation;
pub use linear::Linear;
pub use activation::Sigmoid;


use ndarray::{ Array2, ArrayView2 };
pub trait Propagable {
    fn init(in_dim: usize, out_dim: usize) -> Self where Self: Sized;
    fn forward(&mut self, x: &ArrayView2<f64>) -> Array2<f64>;
    fn backward(&mut self, output_grad: &ArrayView2<f64>) -> Array2<f64>;
    fn step(&mut self, lr: f64);
}