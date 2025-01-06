use ndarray::{ Array, Array2, ArrayView, ArrayView2, Axis, Dimension, Shape };

pub struct Sigmoid<D> where D: Dimension {
    input: Option<Array<f64, D>>,
}
impl<D> Sigmoid<D> where D: Dimension {
    pub fn init() -> Self {
        return Sigmoid { input: None };
    }

    pub fn forward(&mut self, x: ArrayView<f64, D>) -> Array<f64, D> {
        self.input = Some(x.to_owned());
        return x.map(|e| 1.0 / (1.0 + (-1.0 * e).exp()));
    }

    pub fn backward(&self, grad: ArrayView<f64, D>) -> Array<f64, D> {
        let e: Array<f64, D> = self.input.as_ref().unwrap().map(|x| (-x)).exp();
        return &grad * e.map(|e| e / ((1.0 + e) * (1.0 + e)));
    }
}

pub struct LayerNorm2{
    a: Array2<f64>,
    b: Array2<f64>,
    eps: f64,
}

impl LayerNorm2{
    pub fn init(shape: (usize, usize), eps: Option<f64>) -> Self {
        let array2: Array2<f64> = Array2::ones((3, 4));
        
        return LayerNorm2 {
            a      : Array2::<f64>::ones(shape.clone()),
            b      : Array2::<f64>::zeros(shape.clone()),
            eps: eps.unwrap_or(1e-6),
        }
    }

    pub fn forward(&mut self, x: ArrayView2<f64>) -> Array2<f64> {
        let mean = x.mean_axis(Axis(0)).unwrap();
        let std_dev = x.std_axis(Axis(0), 1.0);
        return &self.a * (&x - mean) / (std_dev + self.eps) + &self.b;
    }
}
