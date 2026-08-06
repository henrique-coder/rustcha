use image::DynamicImage;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{
    decoder::{self, DecodedText},
    image_input,
    model::{self, InputLayout},
};

pub type DetectionBox = (i64, i64, i64, i64, f32);
type RecognitionWithPositions = (String, Option<f32>, Vec<DetectionBox>);
type RawDetectionBox = (f32, f32, f32, f32, f32);

const DETECTOR_ROWS: usize = 3549;
const DETECTOR_COLUMNS: usize = 6;

#[pyclass]
pub struct Rustcha {
    session: ort::session::Session,
    detector: Option<ort::session::Session>,
    characters: Vec<String>,
    input: InputLayout,
}

#[pymethods]
impl Rustcha {
    #[new]
    pub fn new(py: Python<'_>) -> PyResult<Self> {
        let resources = model::load_resources(py)?;
        Ok(Self {
            session: resources.session,
            detector: None,
            characters: resources.characters,
            input: resources.input,
        })
    }

    #[pyo3(signature = (image, allowed_characters = None))]
    pub fn recognize_detailed(
        &mut self,
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        allowed_characters: Option<String>,
    ) -> PyResult<(String, Option<f32>)> {
        let bytes = image_input::read_image_bytes(image)?;
        py.detach(move || {
            let decoded_image = image_input::decode_image(&bytes)?;
            let decoded = self.recognize_image(&decoded_image, allowed_characters.as_deref())?;
            Ok((decoded.text, decoded.confidence))
        })
    }

    #[pyo3(signature = (images, allowed_characters = None))]
    pub fn batch_recognize_detailed(
        &mut self,
        py: Python<'_>,
        images: &Bound<'_, PyAny>,
        allowed_characters: Option<String>,
    ) -> PyResult<Vec<(String, Option<f32>)>> {
        let image_bytes = images
            .try_iter()?
            .map(|image| image.and_then(|image| image_input::read_image_bytes(&image)))
            .collect::<PyResult<Vec<_>>>()?;

        py.detach(move || {
            image_bytes
                .iter()
                .map(|bytes| {
                    let decoded_image = image_input::decode_image(bytes)?;
                    let decoded =
                        self.recognize_image(&decoded_image, allowed_characters.as_deref())?;
                    Ok((decoded.text, decoded.confidence))
                })
                .collect()
        })
    }

    pub fn load_detector(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.detector.is_none() {
            self.detector = Some(model::load_detector(py)?);
        }
        Ok(())
    }

    pub fn detect(
        &mut self,
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<DetectionBox>> {
        let bytes = image_input::read_image_bytes(image)?;
        py.detach(move || {
            let decoded_image = image_input::decode_image(&bytes)?;
            self.detect_image(&decoded_image)
        })
    }

    #[pyo3(signature = (image, allowed_characters = None))]
    pub fn recognize_with_positions(
        &mut self,
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        allowed_characters: Option<String>,
    ) -> PyResult<(String, Option<f32>, Vec<DetectionBox>)> {
        let bytes = image_input::read_image_bytes(image)?;
        py.detach(move || {
            let decoded_image = image_input::decode_image(&bytes)?;
            let decoded = self.recognize_image(&decoded_image, allowed_characters.as_deref())?;
            let boxes = self.detect_image(&decoded_image)?;
            Ok((decoded.text, decoded.confidence, boxes))
        })
    }

    #[pyo3(signature = (images, allowed_characters = None))]
    pub fn batch_recognize_with_positions(
        &mut self,
        py: Python<'_>,
        images: &Bound<'_, PyAny>,
        allowed_characters: Option<String>,
    ) -> PyResult<Vec<RecognitionWithPositions>> {
        let image_bytes = images
            .try_iter()?
            .map(|image| image.and_then(|image| image_input::read_image_bytes(&image)))
            .collect::<PyResult<Vec<_>>>()?;

        py.detach(move || {
            image_bytes
                .iter()
                .map(|bytes| {
                    let decoded_image = image_input::decode_image(bytes)?;
                    let decoded =
                        self.recognize_image(&decoded_image, allowed_characters.as_deref())?;
                    let boxes = self.detect_image(&decoded_image)?;
                    Ok((decoded.text, decoded.confidence, boxes))
                })
                .collect()
        })
    }
}

impl Rustcha {
    fn recognize_image(
        &mut self,
        image: &DynamicImage,
        allowed_characters: Option<&str>,
    ) -> PyResult<DecodedText> {
        let tensor = image_input::create_tensor(image, &self.input)?;
        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .map_err(|error| PyRuntimeError::new_err(format!("inference failed: {error}")))?;
        if outputs.len() == 0 {
            return Err(PyRuntimeError::new_err("the model returned no outputs"));
        }
        decoder::decode_text(&outputs[0], &self.characters, allowed_characters)
    }

    fn detect_image(&mut self, image: &DynamicImage) -> PyResult<Vec<DetectionBox>> {
        let detector = self
            .detector
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("the detection model has not been loaded"))?;
        let (tensor, image_width, image_height, ratio) =
            image_input::create_detection_tensor(image)?;
        let outputs = detector
            .run(ort::inputs![tensor])
            .map_err(|error| PyRuntimeError::new_err(format!("detection failed: {error}")))?;
        if outputs.len() == 0 {
            return Err(PyRuntimeError::new_err("the detector returned no outputs"));
        }

        let (_, values) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        decode_detection_output(values, image_width, image_height, ratio)
            .map_err(PyRuntimeError::new_err)
    }
}

fn decode_detection_output(
    values: &[f32],
    image_width: u32,
    image_height: u32,
    ratio: f32,
) -> Result<Vec<DetectionBox>, String> {
    if values.len() != DETECTOR_ROWS * DETECTOR_COLUMNS {
        return Err(format!(
            "the detector returned {} values; expected {}",
            values.len(),
            DETECTOR_ROWS * DETECTOR_COLUMNS
        ));
    }
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err("the detector received an invalid resize ratio".to_owned());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("the detector returned a non-finite value".to_owned());
    }

    let mut candidates = Vec::new();
    for (index, row) in values.chunks_exact(DETECTOR_COLUMNS).enumerate() {
        let (stride, grid_offset) = detector_grid(index)
            .expect("the detector output length guarantees a known grid position");
        let grid_width = 416 / stride as usize;
        let center_x = (row[0] + (grid_offset % grid_width) as f32) * stride;
        let center_y = (row[1] + (grid_offset / grid_width) as f32) * stride;
        let box_width = row[2].exp() * stride;
        let box_height = row[3].exp() * stride;
        let score = row[4] * row[5];
        if score > 0.1 {
            candidates.push((
                (center_x - box_width / 2.0) / ratio,
                (center_y - box_height / 2.0) / ratio,
                (center_x + box_width / 2.0) / ratio,
                (center_y + box_height / 2.0) / ratio,
                score,
            ));
        }
    }

    candidates.sort_by(|left, right| right.4.total_cmp(&left.4));
    let mut selected: Vec<RawDetectionBox> = Vec::new();
    for candidate in candidates {
        if selected
            .iter()
            .all(|previous| intersection_over_union(&candidate, previous) <= 0.45)
        {
            selected.push(candidate);
        }
    }

    Ok(selected
        .into_iter()
        .map(|candidate| {
            (
                candidate.0.clamp(0.0, image_width as f32) as i64,
                candidate.1.clamp(0.0, image_height as f32) as i64,
                candidate.2.clamp(0.0, image_width as f32) as i64,
                candidate.3.clamp(0.0, image_height as f32) as i64,
                candidate.4,
            )
        })
        .collect())
}

fn detector_grid(index: usize) -> Option<(f32, usize)> {
    match index {
        0..=2703 => Some((8.0, index)),
        2704..=3379 => Some((16.0, index - 2704)),
        3380..=3548 => Some((32.0, index - 3380)),
        _ => None,
    }
}

fn intersection_over_union(left: &RawDetectionBox, right: &RawDetectionBox) -> f32 {
    let x1 = left.0.max(right.0);
    let y1 = left.1.max(right.1);
    let x2 = left.2.min(right.2);
    let y2 = left.3.min(right.3);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let left_area = (left.2 - left.0).max(0.0) * (left.3 - left.1).max(0.0);
    let right_area = (right.2 - right.0).max(0.0) * (right.3 - right.1).max(0.0);
    intersection / (left_area + right_area - intersection).max(1e-6)
}

#[cfg(test)]
mod tests {
    use super::{DETECTOR_ROWS, detector_grid};

    #[test]
    fn detector_grid_boundaries_are_contiguous() {
        assert_eq!(detector_grid(0), Some((8.0, 0)));
        assert_eq!(detector_grid(2703), Some((8.0, 2703)));
        assert_eq!(detector_grid(2704), Some((16.0, 0)));
        assert_eq!(detector_grid(3379), Some((16.0, 675)));
        assert_eq!(detector_grid(3380), Some((32.0, 0)));
        assert_eq!(detector_grid(DETECTOR_ROWS - 1), Some((32.0, 168)));
        assert_eq!(detector_grid(DETECTOR_ROWS), None);
    }
}
