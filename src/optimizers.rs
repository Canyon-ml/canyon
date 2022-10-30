
use core::optomizers::*;

type LearningRate = f32;
type Beta = f32;
type Beta1 = f32;
type Beta2 = f32;

#[derive(Clone)]
pub enum Opt {
    SGD(LearningRate),
    Adagrad(LearningRate),
    Adadelta(LearningRate),
    Adam(LearningRate, Beta1, Beta2),
    ConjGrad(LearningRate),
    Momentum(LearningRate, Beta),
    RMSPROP(LearningRate, Beta),
}

impl Opt {
    pub fn create (&self) -> Optim {
        match self {
            Opt::SGD(lr) => {
                Box::new(SGD::new(*lr))
            },
            _ => { todo!() }
        }
    }
}