use ndarray::{s, Array, Axis, Dimension, Ix3};
use ndarray::{ Array1, Array2, ArrayView2, ArrayView3, Array3, ArrayView };
use rand::prelude::*;
use rand_distr::StandardNormal;

use super::Sigmoid;

/// perform a[i].dot(b[i].t()) for all slice.
pub fn matmul3(a: &ArrayView3<f64>, b: &ArrayView3<f64>) -> Array3<f64> {
    assert_eq!(a.shape()[0], b.shape()[0]);
    assert_eq!(a.shape()[1], b.shape()[2]);
    let mut ret = Array3::zeros((a.shape()[0], a.shape()[1], b.shape()[1]));
    for i in 0..a.shape()[0] {
        ret.slice_mut(s![i,..,..])
           .assign(&a.slice(s![i,..,..])
                     .dot(&b.slice(s![i,..,..]).t()));
    }

    return ret;

}

pub struct Linear {
    w: Array2<f64>,
    b: Array1<f64>,

    input: Option<Array2<f64>>,
    w_grad: Option<Array2<f64>>,
    b_grad: Option<Array1<f64>>,
}

impl Linear {
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


        return Linear { w: weights, b: bias, input: None, 
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
        self.b_grad = Some(grad.sum_axis(Axis(0)));
        return grad.dot(&self.w.t()) / self.w.shape()[1] as f64;
    }

    pub fn step(&mut self, lr: f64) {
        self.w = &self.w - self.w_grad.as_ref().unwrap() * lr;
        self.b = &self.b - self.b_grad.as_ref().unwrap() * lr;
    }   
}

pub struct Scale;
impl Scale {
    fn forward<D>(a: &ArrayView<f64, D>, s: f64) -> Array<f64, D>
        where D: Dimension {
        return a * s;
    }

    fn backward<D>(a: &ArrayView<f64, D>, s: f64) -> Array<f64, D>
        where D: Dimension {
        return a / s;
    }
}

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

    pub fn backward(&mut self, grad: &ArrayView3<f64>)
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

pub struct ScaledDotProductAttention {
    soft: Sigmoid<Ix3>,
    matmul3_1: MatMul3,
    matmul3_2: MatMul3,
}

impl ScaledDotProductAttention {
    fn init(in_dim: usize, out_dim: usize) -> Self where Self: Sized {
        return ScaledDotProductAttention { 
            soft: Sigmoid::<Ix3>::init(in_dim, out_dim),
            matmul3_1: MatMul3::init(),
            matmul3_2: MatMul3::init(),
        }
    }


    fn forward(&mut self, q: &ArrayView3<f64>, k: &ArrayView3<f64>,
               v: &ArrayView3<f64>) -> Array3<f64> {
        let d = q.shape()[1];
        assert_eq!(q.shape(), k.shape());
        assert_eq!(q.shape(), v.shape());

        let mut temp = self.matmul3_1.forward(&q.view(), &k.view());
        temp = Scale::forward(&temp.view(), (d as f64).sqrt());
        temp = self.soft.forward(&temp.view());
        temp = self.matmul3_2.forward(&temp.view(), v);
        return temp;
    }

    fn backward(&mut self, grad: &ArrayView2<f64>) -> Array2<f64> {
        todo!()
    }

    fn step(&mut self, lr: f64) {
        todo!()
    }
}


pub struct SingleHeadAttention {
    q_linear: Linear,
    k_linear: Linear,
    v_linear: Linear,
    
    sdpa: ScaledDotProductAttention,
}