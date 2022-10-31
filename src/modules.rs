
use core::modules::*;

use crate::optimizers::Opt;

type Size = usize;
type KernelSize = usize;
type NumFilters = usize;
type Stride = usize;
type Padding = usize;

pub enum Mod {
    Dense (Size),
    Conv (KernelSize, NumFilters, Stride, Padding),
    Flatten,
    ELU,
    LeakyReLU,
    MaxPool (KernelSize, Stride),
    Softmax,
    Sigmoid,
    Tanh,
    ReLU,
}

impl Mod {

    pub fn create (&self, prev: &mut (usize, usize, usize, usize), opt: Opt) -> Box<dyn Module> {
        let result: Box<dyn Module>;
        match self {
            Mod::Dense(size) => {
                result = Box::new(Dense::new((prev.0, prev.1), *size, opt.create()));
                *prev = (prev.0, *size, 1, 1);
            },

            Mod::ReLU => {
                result = Box::new(ReLU::new(*prev));
            }

            Mod::Softmax => {
                result = Box::new(Softmax::new(*prev));
            }

            Mod::Sigmoid => {
                result = Box::new(Sigmoid::new(*prev));
            }

            _ => { todo!() }

        };

        result
    }
}