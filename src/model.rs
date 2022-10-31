
use core::modules::*;
use core::gt::Tensor;
use core::Loss;

use soul_core::optomizers::Optim;

use crate::Batch;
use crate::modules::Mod;
use crate::optimizers::Opt;

pub struct Model {
    /// The Modules that make up the network (or layers)
    modules: Vec<Box<dyn Module>>,

    /// The Data we will be operating on
    batch: Vec<(Tensor, Tensor)>,

    /// The Loss Function
    loss: Loss,

    // The Size of each Batch
    batch_size: usize,

    /// The Size of the Input Tensor\
    /// rows, cols, channels, duration
    input_size: (usize, usize, usize, usize),

    /// The Size of the Output / target
    target_size: (usize, usize),

    /// Size of last added Module\
    /// rows, cols, channels, duration
    prev_size: (usize, usize, usize, usize),

    /// The Optomizer for this Network
    opt: Opt,
}

impl Model {
    pub fn new (batch: Batch, opt: Opt, loss: Loss) -> Self {
        Self {
            modules: Vec::new(),
            batch: batch.data,
            loss,
            batch_size: batch.batch_size,
            input_size: batch.input,
            target_size: batch.target,
            prev_size: batch.input,
            opt
        }
    }

    pub fn add (&mut self, module: &[Mod]) {
        for m in module {
            let a = m.create(&mut self.prev_size, self.opt.clone());
            self.modules.push(a);
        }
    }

    /// Backpropogate the Neural Network using the Dataset provided.
    /// - epochs: the amount of times the dataset is iterated. 
    /// - stop_at: the once the avg loss is less than this number, training terminates. 
    pub fn train (&mut self, epochs: usize, stop_at: f32) {

        let mut delta = Tensor::new(self.batch_size, self.batch[0].1.cols, 1, 1);

        for _e in 0..epochs {
            let mut avg_loss = 0.0;
            for (input, target) in self.batch.iter() {

                // The Forward Pass
                let mut next: &Tensor = input;
                {
                    for m in self.modules.iter_mut() {
                        next = m.forward(next);
                    }
                }

                avg_loss += self.loss.compute(next, target, &mut delta);

                println!("Output: {:?}", next.data);

                // The Backward Pass
                let mut next: &Tensor = &delta;
                {
                    for m in self.modules.iter_mut().rev() {
                        next = m.backward(next);
                    }
                }
            }
            avg_loss = self.batch.len() as f32 / avg_loss;
            println!("Avg Loss: {}", avg_loss);

            if avg_loss < stop_at {
                println!("Reached Target Average Loss!");
                return
            }

        }
    }

    /// Measures the Accuracy of the network against the entire dataset.
    /// The number returned is the average loss over each dataset entry. 
    pub fn validate (&mut self) -> f32 {
        let mut loss: f32 = 0.0;
        for (input, target) in self.batch.iter() {
            // The Forward Pass
            let mut next: &Tensor = input;
            {
                for m in self.modules.iter_mut() {
                    next = m.forward(next); 
                }
            }

            let mut delta = Tensor::new(target.rows, target.cols, 1, 1);
            loss += self.loss.compute(next, target, &mut delta);
        }

        loss / self.batch.len() as f32
    }

    /// Supply the network with an input and target. Run the network to produce an output. 
    /// Input and target should be the same size as wht is specified in your dataset - result is the
    /// output of the network and is the same size as target. 
    pub fn run (&mut self, input: &Tensor, target: &Tensor, result: &mut Tensor) -> f32 {

        // The Forward Pass
        let mut next: &Tensor = input;
        {  
            for m in self.modules.iter_mut() { 
                next = m.forward(next);
            }
        }

        Tensor::copy(next, result);

        let mut delta = Tensor::new(target.rows, target.cols, 1, 1);
        let avg_loss = self.loss.compute(next, target, &mut delta);

        avg_loss
    }
}