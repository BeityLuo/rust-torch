use ndarray::{s, Array, Axis, Dimension, Ix3};
use ndarray::{ Array1, Array2, ArrayView2, ArrayView3, Array3, ArrayView };
use rand::prelude::*;
use rand_distr::num_traits::zero;
use rand_distr::StandardNormal;

use super::Sigmoid;

pub struct MatMul3 {
    a: Option<Array3<f64>>,
    b: Option<Array3<f64>>,
}

impl MatMul3 {
    pub fn init() -> Self { MatMul3 { a: None, b: None } }

    pub fn forward(&mut self, a: &ArrayView3<f64>, b: &ArrayView3<f64>)
        -> Array3<f64> {
        // a: (batch, d1, d2), b: (batch, d3, d2)
        assert_eq!(a.shape()[0], b.shape()[0]);
        assert_eq!(a.shape()[1], b.shape()[2]);

        self.a = Some(a.to_owned());
        self.b = Some(b.to_owned());

        let mut ret = Array3::zeros((a.shape()[0], a.shape()[1], b.shape()[1]));
        for i in 0..a.shape()[0] {
            ret.slice_mut(s![i,..,..])
               .assign(&a.slice(s![i,..,..])
                         .dot(&b.slice(s![i,..,..]).t()));
        }
    
        return ret;
    }

    pub fn backward(&self, grad: &ArrayView3<f64>)
        -> (Array3<f64>, Array3<f64>) {
        // grad: (batch, d1, d3)
        let a = self.a.as_ref().unwrap();
        let b = self.b.as_ref().unwrap();
        assert_eq!(a.shape()[0], grad.shape()[0]);
        assert_eq!(a.shape()[1], grad.shape()[1]);
        assert_eq!(b.shape()[1], grad.shape()[2]);

        let mut a_ret = Array3::zeros((a.shape()[0], a.shape()[1], a.shape()[2]));
        let mut b_ret = Array3::zeros((b.shape()[0], b.shape()[1], b.shape()[2]));
        for i in 0..grad.shape()[0] {
            a_ret.slice_mut(s![i,..,..])
                 .assign(&grad.slice(s![i,..,..])
                              .dot(&b.slice(s![i,..,..])));
            b_ret.slice_mut(s![i,..,..])
                 .assign(&grad.slice(s![i,..,..])
                              .t()
                              .dot(&a.slice(s![i,..,..])))
        }

        return (a_ret, b_ret);
    }
}

pub struct Linear2 {
    w: Array2<f64>,
    b: Array1<f64>,

    input: Option<Array2<f64>>,
    w_grad: Option<Array2<f64>>,
    b_grad: Option<Array1<f64>>,
}

impl Linear2 {
    pub fn init(in_dim: usize, out_dim: usize) -> Self {
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


        return Linear2 { w: weights, b: bias, input: None, 
                        w_grad: None, b_grad: None };
    }
    pub fn forward(&mut self, x: &ArrayView2<f64>) -> Array2<f64>{
        assert_eq!(x.shape()[1], self.w.shape()[0]);
        self.input = Some(x.to_owned());
        return x.dot(&self.w) + &self.b;
    }

    pub fn backward(&mut self, grad: &ArrayView2<f64>) -> Array2<f64>{
        let batch_size = grad.shape()[0];
        
        self.w_grad = Some(self.input.as_ref().unwrap().t()
                      .dot(grad) / batch_size as f64);
        self.b_grad = Some(grad.sum_axis(Axis(0)) / batch_size as f64);
        return grad.dot(&self.w.t()) / self.w.shape()[1] as f64;
    }

    pub fn step(&mut self, lr: f64) {
        self.w = &self.w - self.w_grad.as_ref().unwrap() * lr;
        self.b = &self.b - self.b_grad.as_ref().unwrap() * lr;
    }   
}

pub struct Linear3 {
    w: Array2<f64>,
    b: Array1<f64>,

    matmul3: MatMul3,

    input: Option<Array3<f64>>,
    w_grad: Option<Array2<f64>>,
    b_grad: Option<Array1<f64>>,
}

impl Linear3 {
    pub fn init(in_dim: usize, out_dim: usize) -> Self {
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


        return Linear3 { w: weights, b: bias, matmul3: MatMul3::init(), 
                        input: None, w_grad: None, b_grad: None };
    }
    pub fn forward(&mut self, x: &ArrayView3<f64>) -> Array3<f64>{
        assert_eq!(x.shape()[2], self.w.shape()[0]);
        self.input = Some(x.to_owned());
        let mut ret = 
            Array3::zeros((x.shape()[0], x.shape()[1], self.w.shape()[1]));
        for i in 0..x.shape()[0] {
            ret.slice_mut(s![i,..,..])
               .assign(&(x.slice(s![i,..,..]).dot(&self.w) + &self.b));
        }
        return ret;
    }

    pub fn backward(&mut self, grad: &ArrayView3<f64>) -> Array3<f64>{
        let x = self.input.as_ref().unwrap();
        assert_eq!(x.shape()[0], grad.shape()[0]);
        assert_eq!(x.shape()[1], grad.shape()[1]);
        assert_eq!(self.w.shape()[1], grad.shape()[2]);

        let mut w_grad = Array2::<f64>::zeros((self.w.shape()[0], self.w.shape()[1]));
        let mut b_grad = Array1::<f64>::zeros(self.b.shape()[0]);
        let mut ret = 
            Array3::<f64>::zeros((x.shape()[0], x.shape()[1], x.shape()[2]));

        for i in 0..x.shape()[0] {
            w_grad = w_grad + x.slice(s![i,..,..]).t().dot(&grad.slice(s![i,..,..]));
            b_grad = b_grad + grad.slice(s![i,..,..]).sum_axis(Axis(0));
            ret.slice_mut(s![i,..,..])
               .assign(&(grad.slice(s![i,..,..]).dot(&self.w.t())));
        }

        self.w_grad = Some(w_grad / grad.shape()[0] as f64);
        self.b_grad = Some(b_grad / grad.shape()[0] as f64);
        ret /= self.w.shape()[1] as f64;

        return ret;
    }

    pub fn step(&mut self, lr: f64) {
        self.w = &self.w - self.w_grad.as_ref().unwrap() * lr;
        self.b = &self.b - self.b_grad.as_ref().unwrap() * lr;
    }   
}

pub struct Scale {
    s: f64
}
impl Scale {
    pub fn init(s: f64) -> Self { 
        assert_ne!(s, 0.0);
        return Scale{ s }
    }
    pub fn forward<D>(&self, a: &ArrayView<f64, D>) -> Array<f64, D>
        where D: Dimension {
        return a * self.s;
    }

    pub fn backward<D>(&self, a: &ArrayView<f64, D>) -> Array<f64, D>
        where D: Dimension {
        return a / self.s;
    }
}



pub struct ScaledDotProductAttention {
    
    matmul3_1: MatMul3,
    scale: Scale,
    soft: Sigmoid<Ix3>,
    matmul3_2: MatMul3,
   
}

impl ScaledDotProductAttention {
    /// q: (batch, d1, d2), k: (batch, d1, d2), v: (batch, d1, d2)
    fn init(d1: usize) -> Self where Self: Sized {
        return ScaledDotProductAttention {
            matmul3_1: MatMul3::init(),
            soft: Sigmoid::<Ix3>::init(),
            scale: Scale::init((d1 as f64).sqrt()),
            matmul3_2: MatMul3::init(),
        }
    }


    fn forward(&mut self, q: &ArrayView3<f64>, k: &ArrayView3<f64>,
               v: &ArrayView3<f64>) -> Array3<f64> {
        assert_eq!(q.shape(), k.shape());
        assert_eq!(q.shape(), v.shape());
        
        let mut temp = self.matmul3_1.forward(&q.view(), &k.view());
        temp = self.scale.forward(&temp.view());
        temp = self.soft.forward(&temp.view());
        temp = self.matmul3_2.forward(&temp.view(), v);
        return temp;
    }

    fn backward(&self, grad: &ArrayView3<f64>)
        -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let (mut temp, v_grad) = self.matmul3_2.backward(grad);
        temp = self.soft.backward(&temp.view());
        temp = self.scale.backward(&temp.view());
        let (q_grad, k_grad) = self.matmul3_1.backward(&temp.view());
        return (q_grad, k_grad, v_grad);
    }
}


pub struct SingleHeadAttention {
    q_linear: Linear3,
    k_linear: Linear3,
    v_linear: Linear3,
    linear: Linear3,
    
    sdpa: ScaledDotProductAttention,
}

impl SingleHeadAttention {
    pub fn init(in_dim: usize, out_dim: usize) -> Self {
        return SingleHeadAttention {
            q_linear: Linear3::init(in_dim, out_dim),
            k_linear: Linear3::init(in_dim, out_dim),
            v_linear: Linear3::init(in_dim, out_dim),
            linear: Linear3::init(out_dim, out_dim),

            sdpa: ScaledDotProductAttention::init(out_dim),
        }
    }

    pub fn forward(&mut self, q: &ArrayView3<f64>, k: &ArrayView3<f64>, v: &ArrayView3<f64>)
        -> Array3<f64> {
        let q = self.q_linear.forward(q);
        let k = self.k_linear.forward(k);
        let v = self.v_linear.forward(v);
        let temp = self.sdpa.forward(&q.view(), &k.view(), &v.view());
        return self.linear.forward(&temp.view());
    }

    pub fn backward(&mut self, grad: &ArrayView3<f64>)
        -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let temp = self.linear.backward(grad);
        let (q_grad, k_grad, v_grad) = self.sdpa.backward(&temp.view());
        return (self.q_linear.backward(&q_grad.view()),
                self.k_linear.backward(&k_grad.view()),
                self.v_linear.backward(&v_grad.view()));
    }
}