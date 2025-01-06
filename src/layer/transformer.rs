use super::linear::{ MatMul3, Scale, Linear3 };
use super::non_linear::Sigmoid ;
use ndarray::{ ArrayView3, Array3, Ix3 };

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


    fn forward(&mut self, q: ArrayView3<f64>, k: ArrayView3<f64>,
               v: ArrayView3<f64>) -> Array3<f64> {
        assert_eq!(q.shape(), k.shape());
        assert_eq!(q.shape(), v.shape());
        
        let mut temp = self.matmul3_1.forward(q, k);
        temp = self.scale.forward(temp.view());
        temp = self.soft.forward(temp.view());
        temp = self.matmul3_2.forward(temp.view(), v);
        return temp;
    }

    fn backward(&self, grad: ArrayView3<f64>)
        -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let (mut temp, v_grad) = self.matmul3_2.backward(grad);
        temp = self.soft.backward(temp.view());
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
    pub fn init(in_dim: usize, out_dim: usize) -> Self {
        return SingleHeadAttention {
            q_linear: Linear3::init(in_dim, out_dim),
            k_linear: Linear3::init(in_dim, out_dim),
            v_linear: Linear3::init(in_dim, out_dim),
            linear: Linear3::init(out_dim, out_dim),

            sdpa: ScaledDotProductAttention::init(out_dim),
        }
    }

    pub fn forward(&mut self, q: ArrayView3<f64>, k: ArrayView3<f64>, v: ArrayView3<f64>)
        -> Array3<f64> {
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

