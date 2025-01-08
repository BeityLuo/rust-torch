use core::num;
use std::char::decode_utf16;
use std::ops::Sub;

use super::linear::{ Add, Linear3, MatMul3, Scale };
use super::non_linear::{ReLU, Sigmoid} ;
use super::LayerNorm2;
use ndarray::{ ArrayView3, Array3, Ix3 };
use rand_distr::num_traits::zero;

pub struct ScaledDotProductAttention {
    
    matmul3_1: MatMul3,
    scale: Scale,
    softmax: Sigmoid<Ix3>,
    matmul3_2: MatMul3,
    masked: bool,
}

impl ScaledDotProductAttention {
    /// q: (batch, d1, d2), k: (batch, d1, d2), v: (batch, d1, d2)
    /// (batch, d1, d2) -> (batch, d1, d2)
    fn init(d1: usize, masked: bool) -> Self where Self: Sized {
        return ScaledDotProductAttention {
            matmul3_1: MatMul3::init(),
            softmax: Sigmoid::<Ix3>::init(),
            scale: Scale::init((d1 as f64).sqrt()),
            matmul3_2: MatMul3::init(),
            masked: masked,
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
    /// (batch, seq_len, in_size) -> (batch, seq_len, out_size)
    pub fn init(in_size: usize, out_size: usize, masked: bool) -> Self {
        return SingleHeadAttention {
            q_linear: Linear3::init(in_size, out_size),
            k_linear: Linear3::init(in_size, out_size),
            v_linear: Linear3::init(in_size, out_size),
            linear: Linear3::init(out_size, out_size),
            
            sdpa: ScaledDotProductAttention::init(out_size, masked),
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
    pub fn init(in_size: usize, out_size: usize) -> Self {
        const d_ff: usize = 256;
        return FeedForward {
            l1: Linear3::init(in_size, d_ff),
            l2: Linear3::init(d_ff, out_size),
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
    pub fn init(seq_len: usize, embed_size: usize, hidden_size: usize) -> Self {
        return Encoder {
            attention: SingleHeadAttention::init(embed_size, hidden_size, false),
            add_norm1: Add_Norm::init(seq_len, hidden_size),
            feed_forward: FeedForward::init(hidden_size, embed_size),
            add_norm2: Add_Norm::init(seq_len, embed_size),
        }
    }

    pub fn forward(&mut self, enc_input: ArrayView3<f64>) -> Array3<f64> {
        let x1 = self.attention
            .forward(enc_input.clone(), enc_input.clone(), enc_input.clone());
        let x1 = self.add_norm1.forward(enc_input, x1.view());
        let x2 = self.feed_forward.forward(x1.view());
        let x2 = self.add_norm2.forward(x1.view(), x2.view());
        return x2;
    }

    pub fn backward(&mut self, grad: ArrayView3<f64>) -> Array3<f64> {
        let (mut add_norm_grad, feed_grad) = self.add_norm2.backward(grad);
        add_norm_grad = 
            add_norm_grad + self.feed_forward.backward(feed_grad.view());

        let (mut input_grad, att_grad) = 
            self.add_norm1.backward(add_norm_grad.view());
        let (q_grad, k_grad, v_grad) = self.attention.backward(att_grad.view());
        input_grad = (input_grad + q_grad + k_grad + v_grad) / 4.0;

        return input_grad;
    }
}

struct Decoder {
    attention1: SingleHeadAttention,
    add_norm1: Add_Norm,
    attention2: SingleHeadAttention,
    add_norm2: Add_Norm,
    feed_forward: FeedForward,
    add_norm3: Add_Norm,
}

impl Decoder {
    fn init(seq_len: usize, embed_size: usize, hidden_size: usize) -> Self {
        return Decoder {
            attention1: SingleHeadAttention::init(embed_size, hidden_size, false),
            add_norm1: Add_Norm::init(seq_len, hidden_size),
            attention2: SingleHeadAttention::init(hidden_size, hidden_size, false),
            add_norm2: Add_Norm::init(seq_len, hidden_size),
            feed_forward: FeedForward::init(hidden_size, embed_size),
            add_norm3: Add_Norm::init(seq_len, embed_size),
        }
    }

    pub fn forward(&mut self, dec_input: ArrayView3<f64>, enc_input: ArrayView3<f64>)
        -> Array3<f64> {
        let x1 = self.attention1
            .forward(dec_input.clone(), dec_input.clone(), dec_input.clone());
        let x1 = self.add_norm1.forward(dec_input, x1.view());

        let x2 = self.attention2.forward(enc_input.clone(), enc_input, x1.view());
        let x2 = self.add_norm2.forward(x2.view(), x1.view());

        let x3 = self.feed_forward.forward(x2.view());
        let x3 = self.add_norm2.forward(x3.view(), x2.view());
        return x3;
    }

    pub fn backward(&mut self, grad: ArrayView3<f64>)
        -> (Array3<f64>, Array3<f64>) {
        let (mut add_norm_grad2, feed_grad) = self.add_norm3.backward(grad);
        add_norm_grad2 =
            add_norm_grad2 + self.feed_forward.backward(feed_grad.view());

        let (mut add_norm_grad1, att_grad2) =
            self.add_norm2.backward(add_norm_grad2.view());
        let (q_grad, k_grad, v_grad) = self.attention2.backward(att_grad2.view());
        add_norm_grad1 = (add_norm_grad1 + v_grad) / 2.0;
        let enc_input_grad = (q_grad + k_grad) / 2.0;

        let (mut dec_input_grad, att_grad1) =
            self.add_norm1.backward(add_norm_grad1.view());
        let (q_grad, k_grad, v_grad) = self.attention1.backward(att_grad1.view());
        dec_input_grad = (dec_input_grad + q_grad + k_grad + v_grad) / 4.0;

        return (dec_input_grad, enc_input_grad);
    }
}

struct Transformer {
    encoders: Vec<Encoder>,
    decoders: Vec<Decoder>,

    linear: Linear3,
    softmax: Sigmoid<Ix3>,
}

impl Transformer {
    fn init(num_enc: usize, num_dec: usize,
            seq_len: usize, embed_size: usize, tgt_vocab_size: usize,
            enc_hidden_size: usize, dec_hidden_size: usize) -> Self{
        assert_ne!(num_enc, 0);
        assert_ne!(num_dec, 0);
        let mut encoders: Vec<Encoder> = Vec::new();
        for i in 0..num_enc {
            if i == 0 {
                encoders.push(Encoder::init(seq_len, embed_size, enc_hidden_size));
            } else {
                encoders.push(Encoder::init(seq_len, enc_hidden_size, enc_hidden_size));
            }
        }

        let mut decoders: Vec<Decoder> = Vec::new();
        for i in 0..num_dec {
            if i == 0 {
                decoders.push(Decoder::init(seq_len, embed_size, dec_hidden_size));
            } else {
                decoders.push(Decoder::init(seq_len, dec_hidden_size, dec_hidden_size));
            }
        }

        return Transformer {
            encoders: encoders,
            decoders: decoders,
            linear: Linear3::init(dec_hidden_size, tgt_vocab_size),
            softmax: Sigmoid::init(),
        }
    }

    fn forward(&mut self, input: ArrayView3<f64>) -> Array3<f64> {
        let mut enc_output: Array3<f64>;
        let mut enc_output_view = input.view();
        for encoder in &mut self.encoders {
            enc_output = encoder.forward(enc_output_view);
            enc_output_view = enc_output.view();
        }

        let mut dec_output: Array3<f64>;
        let mut dec_output_view = input.view();
        for decoder in &mut self.decoders {
            dec_output = decoder.forward(dec_output_view, enc_output_view);
            dec_output_view = dec_output.view();
        }

        let output = self.softmax.forward(self.linear.forward(dec_output_view).view());
        return output;
    }

    fn backward(&mut self, grad: ArrayView3<f64>) -> Array3<f64> {
        let mut dec_grad = self.softmax.backward(grad);
        let mut dec_grad_view = dec_grad.view();
        let mut enc_grad = None;
        
        let mut temp;
        for decoder in self.decoders.iter_mut().rev() {
            (dec_grad, temp) = decoder.backward(dec_grad_view);
            dec_grad_view = dec_grad.view();

            match enc_grad {
                Some(grad) => enc_grad = Some(grad + temp),
                None => enc_grad = Some(temp),
            }
        }

        
        let mut enc_grad_view = enc_grad.as_ref().unwrap().view();
        let mut enc_grad: Array3<f64> = Default::default();
        for encoder in self.encoders.iter_mut().rev() {
            enc_grad = encoder.backward(enc_grad_view);
            enc_grad_view = enc_grad.view();
        }

        return enc_grad;
    }
}