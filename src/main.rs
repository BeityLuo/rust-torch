mod mnist;
mod layer;
mod dataloader;
use std::{array, result, vec};
use dataloader::DataLoader;
use ndarray::{Array1, Array2, ArrayView2};
use ndarray::prelude::*;
use layer::{Linear2, Sigmoid};
struct DNN {
    l1: Linear2,
    a1: Sigmoid<Ix2>,
    l2: Linear2,
    a2: Sigmoid<Ix2>,
    l3: Linear2,
    a3: Sigmoid<Ix2>,
}

impl DNN {
    pub fn init(in_dim: usize, out_dim: usize) -> Self {
        const HIDDEN1: usize = 64;
        const HIDDEN2: usize = 64;
        return DNN { 
            l1: Linear2::init(in_dim, HIDDEN1),
            a1: Sigmoid::init(),
            l2: Linear2::init(HIDDEN1, HIDDEN2),
            a2: Sigmoid::init(),
            l3: Linear2::init(HIDDEN2, out_dim),
            a3: Sigmoid::init(),
        }
    }

    pub fn forward(&mut self, x: &ArrayView2<f64>) -> Array2<f64> {
        let mut temp = x.to_owned();
        temp = self.l1.forward(temp.view());
        temp = self.a1.forward(temp.view());
        temp = self.l2.forward(temp.view());
        temp = self.a2.forward(temp.view());
        temp = self.l3.forward(temp.view());
        temp = self.a3.forward(temp.view());
        return temp;
    }

    pub fn backward(&mut self, output_grad: &ArrayView2<f64>) -> Array2<f64> {
        let mut temp = output_grad.to_owned();
        temp = self.a3.backward(temp.view());
        temp = self.l3.backward(temp.view());
        temp = self.a2.backward(temp.view());
        temp = self.l2.backward(temp.view());
        temp = self.a1.backward(temp.view());
        temp = self.l1.backward(temp.view());
        return temp;
    }

    pub fn step(&mut self, lr: f64) {
        self.l3.step(lr);
        self.l2.step(lr);
        self.l1.step(lr);
    }
}

fn loss(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> f64 {
    let diff = a - b;
    return diff.mapv_into(|v| v * v).sum();
}

fn label_to_onehot(labels: &Array1<u8>) -> Array2<u8> {
    let min = *labels.iter().min().unwrap();
    let max = *labels.iter().max().unwrap();
    let mut onehot = Array2::zeros([labels.shape()[0],
                                max as usize - min as usize + 1]);

    for i in 0..labels.len() {
        onehot[[i, labels[i] as usize]] = 1;
    }
    return onehot;
}

/// Returns the indices of the maximum values along an axis.
fn argmax(a: &ArrayView2<f64>) -> Array1<usize> {
    let mut ret = Array1::zeros(a.shape()[0]);
    for (i, s) in a.rows().into_iter().enumerate() {
        (ret[i], _) = s.iter()
                       .enumerate()
                       .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                       .unwrap();
    }
    return ret;
}

/// Compare a probability distribution result and label, return correct num.
// fn compare_results(result: &ArrayView2<f64>, label: &ArrayView1<u8>) -> usize {
//     assert_eq!(result.shape()[0], label.shape()[0]);
//     let max_index = argmax(result);
//     let max_index = max_index.mapv(|x| x as u8);
    
//     // compare every element and count for 'true'.
//     // Not for sure this is faster than just for-loop, just looks very cool!
//     return label.iter()
//                 .zip(max_index.iter())
//                 .map(|(&x, &y)| x == y)
//                 .collect::<Array1<bool>>()
//                 .iter()
//                 .filter(|&&x| x)
//                 .count()
// }

/// Compare a probability distribution result and label, return correct num.
fn compare_results(result: &ArrayView2<f64>, label: &ArrayView2<f64>) -> usize {
    assert_eq!(result.shape()[0], label.shape()[0]);
    let max_index = argmax(result);
    let label_index = argmax(label);
    
    // compare every element and count for 'true'.
    // Not for sure this is faster than just for-loop, just looks very cool!
    return label_index.iter()
                      .zip(max_index.iter())
                      .map(|(&x, &y)| x == y)
                      .collect::<Array1<bool>>()
                      .iter()
                      .filter(|&&x| x)
                      .count()
}

fn test_func() {
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    let b = array![[1.0, 2.0]];
    // let b = norm.forward(&a.view());
    println!("{:?}", a.dim().into_shape_with_order());
}


fn main() {
    test_func();
    return;

    
    let (imgs, labels) = mnist::get_train_set();
    let num = imgs.shape()[0];
    let imgs = imgs.map(|v| *v as f64)
                   .into_shape_with_order((num, 28*28)).unwrap();
    let labels = label_to_onehot(&labels)
                 .map(|v| *v as f64);
    let mut loader = DataLoader::init((imgs, labels), 100, false).unwrap();
    
    let mut module = DNN::init(28*28, 10);

    for i in 0..100 {
        let mut i = 0;
        let mut correct_cnt = 0;
        let mut total_loss = 0.0;
        while let Some(data) = loader.next() {
            let (img, label) = data;
            // println!("img.shape = {:?}, label.shape() = {:?}", img.shape(), label.shape());
            let result = module.forward(&img.view());
            // println!("label  = {}", &label);
            // println!("result = {}", &result);
            let loss = loss(&result.view(), &label);
            let grad = &result - &label;
            // println!("grad   = {}", &grad);
            let grad = module.backward(&grad.view());
            // println!("grad   = {}", &grad);
            module.step(0.1);
            
            correct_cnt += compare_results(&result.view(), &label);
            total_loss += loss;
            i += 1;
            if i % 200 == 0 {
                println!("i = {}", i);
            }
        }
        loader.reset();

        println!("epoch {}: total loss is {}, train accuracy is {}%",
                 i, total_loss, correct_cnt as f64 / num as f64);
    }
    
}