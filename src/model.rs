use std::{collections::HashSet, fs, path::PathBuf};

use ort::{
    session::Session,
    value::{TensorElementType, ValueType},
};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use serde::Deserialize;

const MODEL_PATH: &str = "models/ocr/common.onnx";
const CHARACTERS_PATH: &str = "models/ocr/characters.json";
const DETECTOR_MODEL_PATH: &str = "models/detection/common_det.onnx";
pub struct ModelResources {
    pub session: Session,
    pub characters: Vec<String>,
    pub input: InputLayout,
}

pub struct InputLayout {
    pub height: usize,
    pub width: Option<usize>,
    pub channels: usize,
    pub channels_first: bool,
}

#[derive(Deserialize)]
struct CharacterMap {
    version: u32,
    blank_index: usize,
    characters: Vec<String>,
}

fn package_path(py: Python<'_>, relative_path: &str) -> PyResult<PathBuf> {
    let package = py.import("rustcha")?;
    let package_file = package.getattr("__file__")?.extract::<PathBuf>()?;
    let package_directory = package_file
        .parent()
        .ok_or_else(|| PyRuntimeError::new_err("could not determine the package directory"))?
        .to_path_buf();

    Ok(package_directory.join(relative_path))
}

pub fn load_resources(py: Python<'_>) -> PyResult<ModelResources> {
    let model_path = package_path(py, MODEL_PATH)?;
    let characters_path = package_path(py, CHARACTERS_PATH)?;
    let characters_bytes = fs::read(&characters_path).map_err(|error| {
        PyRuntimeError::new_err(format!("could not read character map: {error}"))
    })?;
    let character_map: CharacterMap =
        serde_json::from_slice(&characters_bytes).map_err(|error| {
            PyRuntimeError::new_err(format!("could not parse character map: {error}"))
        })?;
    if character_map.version != 1 || character_map.blank_index != 0 {
        return Err(PyRuntimeError::new_err("unsupported character map format"));
    }
    validate_character_map(&character_map.characters)?;

    let mut builder = Session::builder().map_err(runtime_error)?;
    let session = builder
        .commit_from_file(&model_path)
        .map_err(runtime_error)?;
    let input = inspect_input(&session)?;

    Ok(ModelResources {
        session,
        characters: character_map.characters,
        input,
    })
}

pub fn load_detector(py: Python<'_>) -> PyResult<Session> {
    let model_path = package_path(py, DETECTOR_MODEL_PATH)?;
    let mut builder = Session::builder().map_err(runtime_error)?;
    builder.commit_from_file(&model_path).map_err(runtime_error)
}

fn inspect_input(session: &Session) -> PyResult<InputLayout> {
    let Some(ValueType::Tensor { shape, ty, .. }) =
        session.inputs().first().map(|input| input.dtype())
    else {
        return Err(PyRuntimeError::new_err(
            "the OCR model must expose a tensor input",
        ));
    };
    if *ty != TensorElementType::Float32 {
        return Err(PyRuntimeError::new_err(format!(
            "the OCR model input must contain float32 values, not {ty}"
        )));
    }
    if shape.len() != 4 {
        return Err(PyRuntimeError::new_err(format!(
            "the OCR model input must have four dimensions, not {shape}"
        )));
    }
    if !matches!(shape[0], -1 | 1) {
        return Err(PyRuntimeError::new_err(
            "the OCR model batch dimension must be one or dynamic",
        ));
    }

    let (height, width, channels, channels_first) = if matches!(shape[1], 1 | 3) {
        (shape[2], shape[3], shape[1], true)
    } else if matches!(shape[3], 1 | 3) {
        (shape[1], shape[2], shape[3], false)
    } else {
        return Err(PyRuntimeError::new_err(format!(
            "the OCR model must have one or three color channels, not {shape}"
        )));
    };
    let height = usize::try_from(height)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            PyRuntimeError::new_err("the OCR model height must be fixed and positive")
        })?;
    let width = usize::try_from(width).ok().filter(|value| *value > 0);

    Ok(InputLayout {
        height,
        width,
        channels: channels as usize,
        channels_first,
    })
}

fn validate_character_map(characters: &[String]) -> PyResult<()> {
    if characters.len() < 2 || !characters[0].is_empty() {
        return Err(PyRuntimeError::new_err(
            "the character map must start with one blank entry",
        ));
    }
    if characters[1..].iter().any(String::is_empty) {
        return Err(PyRuntimeError::new_err(
            "the character map contains an unexpected blank entry",
        ));
    }
    let unique = characters.iter().collect::<HashSet<_>>();
    if unique.len() != characters.len() {
        return Err(PyRuntimeError::new_err(
            "the character map contains duplicate entries",
        ));
    }
    Ok(())
}

fn runtime_error(error: impl ToString) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
