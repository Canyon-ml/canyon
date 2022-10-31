
use core::gt::Tensor;
use std::result;

#[derive(Clone)]
pub struct Batch {
    pub data: Vec<(Tensor, Tensor)>,
    pub batch_size: usize,
    pub input: (usize, usize, usize, usize),
    pub target: (usize, usize),
}

impl Batch {
    pub fn new3d (batch_size: usize, input: (usize, usize, usize), target: usize) -> Self {
        Self {
            data: Vec::new(),
            batch_size,
            input: (input.0, input.1, input.2, batch_size),
            target: (batch_size, target)
        }
    }

    pub fn new1d (batch_size: usize, input: usize, target: usize) -> Self {
        Self {
            data: Vec::new(),
            batch_size,
            input: (batch_size, input, 1, 1),
            target: (batch_size, target)
        }
    }

    pub fn load (&mut self, mut batch: Vec<(Vec<f32>, Vec<f32>)>) {

        let mut input = Vec::new();
        let mut target = Vec::new();

        for (i, t) in batch.iter_mut() {
            input.append(i); target.append(t);
        }

        self.data.push((
            Tensor::from_vec(self.input.0, self.input.1, self.input.2, self.input.3, input),
            Tensor::from_vec(self.target.0, self.target.1, 1, 1, target)
        ))
    }

    pub fn load_csv (&mut self, path: &str) {
        let mut rdr = csv::Reader::from_path(path).unwrap();

        for result in rdr.records() {
            let record = result.unwrap();

            
        }
    }
}



