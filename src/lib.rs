mod decoder;
mod engine;
mod image_input;
mod model;

use pyo3::prelude::*;

pub use engine::Rustcha;

#[pymodule]
fn _rustcha(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Rustcha>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
