mod mnist;
mod layers;
mod dataloader;
use std::{array, result, vec};
use dataloader::DataLoader;
use layers::Propagable;
use ndarray::{Array1, Array2, ArrayView2, Ix1, Ix2, Ix3};
use ndarray::prelude::*;
struct DNN {
    layers: Vec<Box<dyn layers::Propagable>>,
}

impl layers::Propagable for DNN {
    fn init(in_dim: usize, out_dim: usize) -> Self {
        const HIDDEN1: usize = 64;
        const HIDDEN2: usize = 64;
        return DNN { layers: vec![
            Box::new(layers::Linear::init(in_dim, HIDDEN1)),
            Box::new(layers::Sigmoid::init(HIDDEN1, HIDDEN1)),
            Box::new(layers::Linear::init(HIDDEN1, HIDDEN2)),
            Box::new(layers::Sigmoid::init(HIDDEN2, HIDDEN2)),
            Box::new(layers::Linear::init(HIDDEN2, out_dim)),
            Box::new(layers::Sigmoid::init(out_dim, out_dim)),
        ] }
    }

    fn forward(&mut self, x: &ArrayView2<f64>) -> Array2<f64> {
        let mut temp = x.to_owned();
        for l in &mut self.layers {
            temp = l.forward(&temp.view());
        }
        return temp;
    }

    fn backward(&mut self, output_grad: &ArrayView2<f64>) -> Array2<f64> {
        let mut temp = output_grad.to_owned();
        for l in &mut self.layers.iter_mut().rev(){
            temp = l.backward(&temp.view());
            // println!("temp = {}", &temp);
        }
        return temp;
    }

    fn step(&mut self, lr: f64) {
        for l in &mut self.layers {
            l.step(lr);
        }
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

fn test_func() {
    let mut sig = layers::Sigmoid::init(1, 1);
    let result = sig.forward(&array![[1.0]].view());
    println!("result = {}", result);
    let result = sig.backward(&array![[1.0]].view());
    println!("result = {}", result);

    let mut module = layers::Linear::init(10, 10);
    println!("w = {}", module.w);

    // let mut linear = layers::Linear { 
    //     w: array![[1.0f64, 1.0], [2.0, 2.0]],
    //     b: array![1.0f64, 2.0],
    //     input: None,
    //     w_grad: None,
    //     b_grad: None };
    // let result = linear.forward(&array![[1.0, 1.0]].view());
    // println!("result = {}", result);
    // let result = linear.backward(&array![[1.0, 1.0]].view());
    // println!("result = {}", result);
}


fn main() {
    // test_func();
    // return;
    let (imgs, labels) = mnist::get_train_set();
    let num = imgs.shape()[0];
    let imgs = imgs.map(|v| *v as f64)
                   .into_shape_with_order((num, 28*28)).unwrap();
    let labels = label_to_onehot(&labels)
                 .map(|v| *v as f64);
    let mut loader = DataLoader::init((imgs, labels), 100, false).unwrap();
    
    let mut module = DNN::init(28*28, 10);

    for i in 0..100 {
        let mut cnt = 0;
        let mut total_loss = 0.0;
        while let Some(data) = loader.next() {
            let (img, label) = data;
            // println!("img.shape = {:?}, label.shape() = {:?}", img.shape(), label.shape());
            let result = module.forward(&img.view());
            // println!("label  = {}", &label);
            // println!("result = {}", &result);
            let loss = loss(&result.view(), &label);
            let grad = (result - label);
            // println!("grad   = {}", &grad);
            let grad = module.backward(&grad.view());
            // println!("grad   = {}", &grad);
            module.step(0.1);
    
            total_loss += loss;
            cnt += 1;
            if cnt % 200 == 0 {
                println!("cnt = {}", cnt);
            }
        }
        loader.reset();
        println!("epoch {}: total loss is {}", i, total_loss);
    }
    
}