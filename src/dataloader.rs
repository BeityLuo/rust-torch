use ndarray::Array;
use ndarray::ArrayView;
use ndarray::Axis;
use ndarray::Dimension;
use ndarray::Slice;
pub struct DataLoader<T1, U1, T2, U2>
where
    U1: Dimension,
    U2: Dimension,
{
    dataset: (Array<T1, U1>, Array<T2, U2>),
    batch_size: usize,
    shuffle: bool,

    size: usize,
    iter_idx: usize,
}

impl<T1, U1, T2, U2> DataLoader<T1, U1, T2, U2>
where
    U1: Dimension,
    U2: Dimension,
{
    pub fn init(
        dataset: (Array<T1, U1>, Array<T2, U2>),
        batch_size: usize,
        shuffle: bool,
    ) -> Option<Self> {
        let size: usize = dataset.0.shape()[0];
        if batch_size > size {
            return None;
        }
        return Some(DataLoader {
            dataset,
            batch_size,
            shuffle,
            size,
            iter_idx: 0,
        });
    }

    pub fn reset(&mut self) {
        self.iter_idx = 0;
    }

    pub fn next(&mut self) -> Option<(ArrayView<T1, U1>, ArrayView<T2, U2>)> {
        let step: usize;
        if self.iter_idx <= self.size - self.batch_size {
            step = self.batch_size;
        } else if self.iter_idx < self.size {
            step = self.size - self.iter_idx;
        } else {
            return None;
        }
        let imgs = self.dataset.0.slice_axis(
            Axis(0),
            Slice::new(
                self.iter_idx.try_into().unwrap(),
                Some((self.iter_idx + step).try_into().unwrap()),
                1,
            ),
        );

        let labels = self.dataset.1.slice_axis(
            Axis(0),
            Slice::new(
                self.iter_idx.try_into().unwrap(),
                Some((self.iter_idx + step).try_into().unwrap()),
                1,
            ),
        );
        self.iter_idx += step;
        return Some((imgs, labels));
    }

    pub fn shuffle(&mut self) {
        todo!();
    }
}
