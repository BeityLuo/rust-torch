use ndarray::{self, Array, Array1, Array3};
use std::fs;
fn get_i32(bytes: &[u8]) -> i32 {
    assert!(bytes.len() >= 4);
    return i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
}

fn analysis_idx1(bytes: &Vec<u8>) -> Array1<u8>{
    assert!(bytes.len() >= 8);
    assert_eq!(bytes[0], 0u8);
    assert_eq!(bytes[1], 0u8);
    let data_type = i32::from(bytes[2]);
    let type_size: i32 = match data_type {
        0x8 | 0x9 => 1,
        0xB       => 2,
        0xC | 0xD => 4,
        0xE       => 8,
        _         => panic!("type error: {}", data_type)
    };
    assert_eq!(type_size, 1);
    let dim = bytes[3];
    let item_num = get_i32(&bytes[4..8]);

    assert_eq!(dim, 1);
    assert_eq!(bytes.len() as i32, 8 + type_size * item_num);
    
    println!("type = {}, dim = {}, item_num = {}", data_type, dim, item_num);

    let v = Array1::from_vec(Vec::from(&bytes[8..]));
    // let v = v.mapv(|x| i32::from(x));
    return v;
}

fn analysis_idx3(bytes: &Vec<u8>) -> Array3<u8>{
    assert!(bytes.len() >= 8);
    assert_eq!(bytes[0], 0u8);
    assert_eq!(bytes[1], 0u8);
    let data_type = i32::from(bytes[2]);
    let type_size: i32 = match data_type {
        0x8 | 0x9 => 1,
        0xB       => 2,
        0xC | 0xD => 4,
        0xE       => 8,
        _         => panic!("type error: {}", data_type)
    };
    assert_eq!(type_size, 1);
    let dim = bytes[3];
    assert_eq!(dim, 3);

    let item_num: usize = get_i32(&bytes[4..8]) as usize;
    let row: usize = get_i32(&bytes[8..12]) as usize;
    let column: usize = get_i32(&bytes[12..16]) as usize;


    assert_eq!(bytes.len(), 16 + type_size as usize * item_num * row * column);
    
    println!("type = {}, dim = {}, item_num = {}, cow = {}, column = {}",
             data_type, dim, item_num, row, column);
    

    let v =  Array::from_shape_vec((item_num, row, column), Vec::from(&bytes[16..])).unwrap();
    // let v = v.mapv(|x| i32::from(x));

    return v;
}

pub fn get_train_set() -> (Array3<u8>, Array1<u8>){
    let bytes = fs::read("./datasets/mnist/train-images-idx3-ubyte").unwrap();
    let images: Array3<u8> = analysis_idx3(&bytes);
    let bytes = fs::read("./datasets/mnist/train-labels-idx1-ubyte").unwrap();
    let labels: Array1<u8> = analysis_idx1(&bytes);
    
    return (images, labels);
}

pub fn get_test_set() -> (Array3<u8>, Array1<u8>){
    let bytes = fs::read("./datasets/mnist/t10k-images-idx3-ubyte").unwrap();
    let images: Array3<u8> = analysis_idx3(&bytes);
    let bytes = fs::read("./datasets/mnist/t10k-labels-idx1-ubyte").unwrap();
    let labels: Array1<u8> = analysis_idx1(&bytes);
    
    return (images, labels);
}
