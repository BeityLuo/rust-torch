use std::ops::Sub;

use super::linear::{ Add, Linear3, MatMul3, Scale };
use super::non_linear::{ReLU, Sigmoid} ;
use super::LayerNorm2;
use ndarray::{ ArrayView3, Array3, Ix3 };

pub struct ScaledDotProductAttention {
    
    matmul3_1: MatMul3,
    scale: Scale,
    softmax: Sigmoid<Ix3>,
    matmul3_2: MatMul3,
   
}

impl ScaledDotProductAttention {
    /// q: (batch, d1, d2), k: (batch, d1, d2), v: (batch, d1, d2)
    /// (batch, d1, d2) -> (batch, d1, d2)
    fn init(d1: usize) -> Self where Self: Sized {
        return ScaledDotProductAttention {
            matmul3_1: MatMul3::init(),
            softmax: Sigmoid::<Ix3>::init(),
            scale: Scale::init((d1 as f64).sqrt()),
            matmul3_2: MatMul3::init(),
        }
    }


    fn forward(&mut self, q: ArrayView3<f64>, k: ArrayView3<f64>,
               v: ArrayView3<f64>) -> Array3<f64> {
        assert_eq!(q.shape(), k.shape());
        assert_eq!(q.shape(), v.shape());
        
        let mut temp = self.matmul3_1.forward(q, k);
        temp = self.scale.forward(temp.view());
        temp = self.softmax.forward(temp.view());
        temp = self.matmul3_2.forward(temp.view(), v);
        return temp;
    }

    fn backward(&self, grad: ArrayView3<f64>)
        -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let (mut temp, v_grad) = self.matmul3_2.backward(grad);
        temp = self.softmax.backward(temp.view());
        temp = self.scale.backward(temp.view());
        let (q_grad, k_grad) = self.matmul3_1.backward(temp.view());
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
    /// (batch, seq_len, in_dim) -> (batch, seq_len, out_dim)
    pub fn init(in_dim: usize, out_dim: usize) -> Self {
        return SingleHeadAttention {
            q_linear: Linear3::init(in_dim, out_dim),
            k_linear: Linear3::init(in_dim, out_dim),
            v_linear: Linear3::init(in_dim, out_dim),
            linear: Linear3::init(out_dim, out_dim),

            sdpa: ScaledDotProductAttention::init(out_dim),
        }
    }

    pub fn forward(&mut self, q: ArrayView3<f64>, k: ArrayView3<f64>,
                   v: ArrayView3<f64>) -> Array3<f64> {
        let q = self.q_linear.forward(q);
        let k = self.k_linear.forward(k);
        let v = self.v_linear.forward(v);
        let temp = self.sdpa.forward(q.view(), k.view(), v.view());
        return self.linear.forward(temp.view());
    }

    pub fn backward(&mut self, grad: ArrayView3<f64>)
        -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let temp = self.linear.backward(grad);
        let (q_grad, k_grad, v_grad) = self.sdpa.backward(temp.view());
        return (self.q_linear.backward(q_grad.view()),
                self.k_linear.backward(k_grad.view()),
                self.v_linear.backward(v_grad.view()));
    }
}

pub struct Add_Norm {
    norm: LayerNorm2,
}

impl Add_Norm {
    /// (batch, d1, d2) -> (batch, d1, d2)
    pub fn init(d1: usize, d2: usize) -> Self {
        return Add_Norm {
            norm: LayerNorm2::init(d1, d2, None),
        }
    }

    pub fn forward(&mut self, a: ArrayView3<f64>, b: ArrayView3<f64>)
        -> Array3<f64>{
        return Add::forward(a, self.norm.forward(b).view());
    }

    pub fn backward(&mut self, grad: ArrayView3<f64>)
        -> (Array3<f64>, Array3<f64>) {
        let (a_grad, b_grad) = Add::backward(grad);
        let b_grad = self.norm.backward(b_grad.view());
        return (a_grad, b_grad);
    }
}

struct FeedForward {
    l1: Linear3,
    l2: Linear3,
    relu: ReLU<Ix3>,
}

impl FeedForward {
    pub fn init(in_dim: usize, out_dim: usize) -> Self {
        const d_ff: usize = 256;
        return FeedForward {
            l1: Linear3::init(in_dim, d_ff),
            l2: Linear3::init(d_ff, out_dim),
            relu: ReLU::init(),
        }
    }

    pub fn forward(&mut self, x: ArrayView3<f64>) -> Array3<f64> {
        return self.l2.forward(
               self.relu.forward(
               self.l1.forward(x).view()).view());
    }

    pub fn backward(&mut self, grad: ArrayView3<f64>) -> Array3<f64> {
        return self.l1.backward(
               self.relu.backward(
               self.l2.backward(grad).view()).view());
    }
}

struct Encoder {
    attention: SingleHeadAttention,
    add_norm1: Add_Norm,
    feed_forward: FeedForward,
    add_norm2: Add_Norm,
}

impl Encoder {
    fn init(seq_len: usize, embed_len: usize, hidden_len: usize) -> Self {
        return Encoder {
            attention: SingleHeadAttention::init(embed_len, hidden_len),
            add_norm1: Add_Norm::init(seq_len, hidden_len),
            feed_forward: FeedForward::init(hidden_len, embed_len),
            add_norm2: Add_Norm::init(seq_len, embed_len),
        }
    }

    fn forward(&mut self, x: ArrayView3<f64>) -> Array3<f64> {
        let x1 = self.attention.forward(x.clone(), x.clone(), x.clone());
        let x1 = self.add_norm1.forward(x, x1.view());
        let x2 = self.feed_forward.forward(x1.view());
        let x2 = self.add_norm2.forward(x1.view(), x2.view());
        return x2;
    }
}