use ndarray::{ Array, Array2, Array3, ArrayView, ArrayView2, ArrayView3, Axis, Dimension };

pub struct Sigmoid<D> where D: Dimension {
    input: Option<Array<f64, D>>,
}
impl<D> Sigmoid<D> where D: Dimension {
    pub fn init() -> Self {
        return Sigmoid { input: None };
    }

    pub fn forward(&mut self, x: ArrayView<f64, D>) -> Array<f64, D> {
        self.input = Some(x.to_owned());
        return x.mapv(|e| 1.0 / (1.0 + (-1.0 * e).exp()));
    }

    pub fn backward(&self, grad: ArrayView<f64, D>) -> Array<f64, D> {
        let e: Array<f64, D> = self.input.as_ref().unwrap().map(|x| (-x)).exp();
        return &grad * e.mapv(|e| e / ((1.0 + e) * (1.0 + e)));
    }
}

pub struct ReLU<D> where D: Dimension {
    input: Option<Array<f64, D>>,
}
impl<D> ReLU<D> where D: Dimension {
    pub fn init() -> Self {
        return ReLU { input: None };
    }

    pub fn forward(&mut self, x: ArrayView<f64, D>) -> Array<f64, D> {
        self.input = Some(x.to_owned());
        return x.mapv(|e| if e > 0.0 { e } else { 0.0 });
    }

    pub fn backward(&self, grad: ArrayView<f64, D>) -> Array<f64, D> {
        let e: Array<f64, D> = self.input.as_ref().unwrap().map(|x| (-x)).exp();
        return &grad * e.mapv(|e| if e > 0.0 { 1.0 } else { 0.0 });
    }
}

pub struct LayerNorm2{
    a: Array2<f64>,
    b: Array2<f64>,
    eps: f64,

    input: Option<Array3<f64>>,
    a_grad: Option<Array2<f64>>,
    b_grad: Option<Array2<f64>>,
}

impl LayerNorm2{
    pub fn init(d1: usize, d2: usize, eps: Option<f64>) -> Self {        
        return LayerNorm2 {
            a     : Array2::<f64>::ones((d1, d2)),
            b     : Array2::<f64>::zeros((d1, d2)),
            eps   : eps.unwrap_or(1e-6),
            input : None,
            a_grad: None,
            b_grad: None,
        }
    }

    pub fn forward(&mut self, x: ArrayView3<f64>) -> Array3<f64> {
        let mean = x.mean_axis(Axis(0)).unwrap();
        let std_dev = x.std_axis(Axis(0), 1.0);
        self.input = Some(x.to_owned());
        return (&x - mean) / (std_dev + self.eps) * &self.a + &self.b;
    }

    pub fn backward(&mut self, grad: ArrayView3<f64>) -> Array3<f64> {
        let x = self.input.as_ref().unwrap();
        let mean = x.mean_axis(Axis(0)).unwrap();
        let std_dev = x.std_axis(Axis(0), 1.0);

        self.a_grad = Some(((x - mean) / (&std_dev + self.eps))
                            .mean_axis(Axis(0))
                            .unwrap());
        self.b_grad = Some(grad.mean_axis(Axis(0)).unwrap());
        return &grad * &self.a / (std_dev + self.eps);
    }
}
