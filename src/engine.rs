use image::DynamicImage;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{
    decoder::{self, DecodedText},
    image_input, model,
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
    preprocessor: image_input::OcrPreprocessor,
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
            preprocessor: image_input::OcrPreprocessor::new(resources.input),
        })
    }

    #[pyo3(signature = (image, allowed_characters = None, calculate_confidence = false))]
    pub fn recognize_detailed(
        &mut self,
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        allowed_characters: Option<String>,
        calculate_confidence: bool,
    ) -> PyResult<(String, Option<f32>)> {
        let bytes = image_input::read_image_bytes(image)?;
        py.detach(move || {
            let decoded_image = image_input::decode_image(bytes.as_ref())?;
            let decoded = self.recognize_image(
                &decoded_image,
                allowed_characters.as_deref(),
                calculate_confidence,
            )?;
            Ok((decoded.text, decoded.confidence))
        })
    }

    #[pyo3(signature = (images, allowed_characters = None, calculate_confidence = false))]
    pub fn batch_recognize_detailed(
        &mut self,
        py: Python<'_>,
        images: &Bound<'_, PyAny>,
        allowed_characters: Option<String>,
        calculate_confidence: bool,
    ) -> PyResult<Vec<(String, Option<f32>)>> {
        let image_bytes = images
            .try_iter()?
            .map(|image| image.and_then(|image| image_input::read_image_bytes(&image)))
            .collect::<PyResult<Vec<_>>>()?;

        py.detach(move || {
            self.recognize_batch(
                &image_bytes,
                allowed_characters.as_deref(),
                calculate_confidence,
            )
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
            let decoded_image = image_input::decode_image(bytes.as_ref())?;
            self.detect_image(&decoded_image)
        })
    }

    #[pyo3(signature = (image, allowed_characters = None, calculate_confidence = false))]
    pub fn recognize_with_positions(
        &mut self,
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        allowed_characters: Option<String>,
        calculate_confidence: bool,
    ) -> PyResult<(String, Option<f32>, Vec<DetectionBox>)> {
        let bytes = image_input::read_image_bytes(image)?;
        py.detach(move || {
            let decoded_image = image_input::decode_image(bytes.as_ref())?;
            let decoded = self.recognize_image(
                &decoded_image,
                allowed_characters.as_deref(),
                calculate_confidence,
            )?;
            let boxes = self.detect_image(&decoded_image)?;
            Ok((decoded.text, decoded.confidence, boxes))
        })
    }

    #[pyo3(signature = (images, allowed_characters = None, calculate_confidence = false))]
    pub fn batch_recognize_with_positions(
        &mut self,
        py: Python<'_>,
        images: &Bound<'_, PyAny>,
        allowed_characters: Option<String>,
        calculate_confidence: bool,
    ) -> PyResult<Vec<RecognitionWithPositions>> {
        let image_bytes = images
            .try_iter()?
            .map(|image| image.and_then(|image| image_input::read_image_bytes(&image)))
            .collect::<PyResult<Vec<_>>>()?;

        py.detach(move || {
            self.recognize_batch_with_positions(
                &image_bytes,
                allowed_characters.as_deref(),
                calculate_confidence,
            )
        })
    }
}

impl Rustcha {
    fn recognize_batch(
        &mut self,
        image_bytes: &[image_input::ImageBytes],
        allowed_characters: Option<&str>,
        calculate_confidence: bool,
    ) -> PyResult<Vec<(String, Option<f32>)>> {
        if image_bytes.len() < 2 {
            return image_bytes
                .iter()
                .map(|bytes| {
                    let image = image_input::decode_image(bytes.as_ref())?;
                    let decoded =
                        self.recognize_image(&image, allowed_characters, calculate_confidence)?;
                    Ok((decoded.text, decoded.confidence))
                })
                .collect();
        }

        let input = self.preprocessor.layout();
        std::thread::scope(|scope| {
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            scope.spawn(move || {
                let mut preprocessor = image_input::OcrPreprocessor::new(input);
                for bytes in image_bytes {
                    let prepared = image_input::decode_image(bytes.as_ref())
                        .and_then(|image| preprocessor.create_tensor(&image));
                    if sender.send(prepared).is_err() {
                        break;
                    }
                }
            });

            let mut results = Vec::with_capacity(image_bytes.len());
            for tensor in receiver {
                let decoded =
                    self.recognize_tensor(tensor?, allowed_characters, calculate_confidence)?;
                results.push((decoded.text, decoded.confidence));
            }
            Ok(results)
        })
    }

    fn recognize_batch_with_positions(
        &mut self,
        image_bytes: &[image_input::ImageBytes],
        allowed_characters: Option<&str>,
        calculate_confidence: bool,
    ) -> PyResult<Vec<RecognitionWithPositions>> {
        let input = self.preprocessor.layout();
        std::thread::scope(|scope| {
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            scope.spawn(move || {
                let mut preprocessor = image_input::OcrPreprocessor::new(input);
                for bytes in image_bytes {
                    let prepared = image_input::decode_image(bytes.as_ref()).and_then(|image| {
                        preprocessor
                            .create_tensor(&image)
                            .map(|tensor| (image, tensor))
                    });
                    if sender.send(prepared).is_err() {
                        break;
                    }
                }
            });

            let mut results = Vec::with_capacity(image_bytes.len());
            for prepared in receiver {
                let (image, tensor) = prepared?;
                let decoded =
                    self.recognize_tensor(tensor, allowed_characters, calculate_confidence)?;
                let boxes = self.detect_image(&image)?;
                results.push((decoded.text, decoded.confidence, boxes));
            }
            Ok(results)
        })
    }

    fn recognize_image(
        &mut self,
        image: &DynamicImage,
        allowed_characters: Option<&str>,
        calculate_confidence: bool,
    ) -> PyResult<DecodedText> {
        let tensor = self.preprocessor.create_tensor(image)?;
        self.recognize_tensor(tensor, allowed_characters, calculate_confidence)
    }

    fn recognize_tensor(
        &mut self,
        tensor: ort::value::Tensor<f32>,
        allowed_characters: Option<&str>,
        calculate_confidence: bool,
    ) -> PyResult<DecodedText> {
        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .map_err(|error| PyRuntimeError::new_err(format!("inference failed: {error}")))?;
        if outputs.len() == 0 {
            return Err(PyRuntimeError::new_err("the model returned no outputs"));
        }
        decoder::decode_text(
            &outputs[0],
            &self.characters,
            allowed_characters,
            calculate_confidence,
        )
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
    let mut candidates = Vec::with_capacity(64);
    for (index, row) in values.chunks_exact(DETECTOR_COLUMNS).enumerate() {
        if row.iter().any(|value| !value.is_finite()) {
            return Err("the detector returned a non-finite value".to_owned());
        }
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
    let mut selected: Vec<RawDetectionBox> = Vec::with_capacity(candidates.len());
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
