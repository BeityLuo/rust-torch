use ndarray::{ Array, ArrayView, Dimension };

pub struct Sigmoid<D> where D: Dimension {
    input: Option<Array<f64, D>>,
}
impl<D> Sigmoid<D> where D: Dimension {
    pub fn init() -> Self {
        return Sigmoid { input: None };
    }

    pub fn forward(&mut self, x: &ArrayView<f64, D>) -> Array<f64, D> {
        self.input = Some(x.to_owned());
        return x.map(|e| 1.0 / (1.0 + (-1.0 * e).exp()));
    }

    pub fn backward(&self, grad: &ArrayView<f64, D>) -> Array<f64, D> {
        let e: Array<f64, D> = self.input.as_ref().unwrap().map(|x| (-x)).exp();
        return grad * e.map(|e| e / ((1.0 + e) * (1.0 + e)));
    }
}
