
extern crate soul_core as core;

pub(crate) mod model;
pub(crate) mod batch;
pub(crate) mod modules;
pub(crate) mod optimizers;

pub use batch::Batch;

pub mod prelude {
    pub use crate::batch::Batch;
    pub use crate::model::Model;
    pub use crate::modules::*;
    pub use crate::optimizers::*;
    pub use core::Loss;
    pub use core::gt;
}
