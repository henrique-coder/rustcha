use std::{
    fs::File,
    io::{Cursor, Read},
    path::{Path, PathBuf},
};

use image::{DynamicImage, ImageReader, Limits, Rgb, RgbImage, imageops::FilterType};
use ort::value::Tensor;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

use crate::model::InputLayout;

const MAX_ENCODED_IMAGE_BYTES: usize = 32 * 1024 * 1024;
const MAX_IMAGE_DIMENSION: u32 = 8192;
const MAX_DECODE_ALLOCATION: u64 = 128 * 1024 * 1024;

pub fn read_image_bytes(source: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(bytes) = source.extract::<Vec<u8>>() {
        return validate_encoded_size(bytes);
    }

    let path = if let Ok(path) = source.extract::<String>() {
        PathBuf::from(path)
    } else if source.hasattr("__fspath__")? {
        PathBuf::from(source.call_method0("__fspath__")?.extract::<String>()?)
    } else {
        return Err(PyTypeError::new_err(
            "image input must be bytes, a string, or a path-like object",
        ));
    };

    read_limited_file(&path)
}

pub fn decode_image(bytes: &[u8]) -> PyResult<DynamicImage> {
    if bytes.len() > MAX_ENCODED_IMAGE_BYTES {
        return Err(image_too_large_error());
    }

    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| PyValueError::new_err(format!("could not identify image: {error}")))?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIMENSION);
    limits.max_image_height = Some(MAX_IMAGE_DIMENSION);
    limits.max_alloc = Some(MAX_DECODE_ALLOCATION);
    reader.limits(limits);
    reader
        .decode()
        .map_err(|error| PyValueError::new_err(format!("could not decode image: {error}")))
}

fn validate_encoded_size(bytes: Vec<u8>) -> PyResult<Vec<u8>> {
    if bytes.len() > MAX_ENCODED_IMAGE_BYTES {
        Err(image_too_large_error())
    } else {
        Ok(bytes)
    }
}

fn read_limited_file(path: &Path) -> PyResult<Vec<u8>> {
    let file = File::open(path)
        .map_err(|error| PyValueError::new_err(format!("could not open image: {error}")))?;
    let mut bytes = Vec::new();
    file.take((MAX_ENCODED_IMAGE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| PyValueError::new_err(format!("could not read image: {error}")))?;
    validate_encoded_size(bytes)
}

fn image_too_large_error() -> PyErr {
    PyValueError::new_err(format!(
        "encoded image exceeds the {} MiB limit",
        MAX_ENCODED_IMAGE_BYTES / (1024 * 1024)
    ))
}

pub fn create_tensor(image: &DynamicImage, layout: &InputLayout) -> PyResult<Tensor<f32>> {
    let width = layout.width.unwrap_or_else(|| {
        let ratio = image.width() as f64 / image.height().max(1) as f64;
        ((layout.height as f64 * ratio).round() as usize).clamp(1, 4096)
    });
    let image = image.resize_exact(width as u32, layout.height as u32, FilterType::Triangle);
    let pixels = pixel_values(&image, layout.channels, layout.channels_first);
    let shape = if layout.channels_first {
        vec![1, layout.channels, layout.height, width]
    } else {
        vec![1, layout.height, width, layout.channels]
    };

    Tensor::from_array((shape, pixels.into_boxed_slice()))
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

pub fn create_detection_tensor(image: &DynamicImage) -> PyResult<(Tensor<f32>, u32, u32, f32)> {
    let image = image.to_rgb8();
    let original_width = image.width();
    let original_height = image.height();
    let ratio = (416.0 / original_width.max(1) as f32).min(416.0 / original_height.max(1) as f32);
    let resized_width = (original_width as f32 * ratio) as u32;
    let resized_height = (original_height as f32 * ratio) as u32;
    let resized = image::imageops::resize(
        &image,
        resized_width.max(1),
        resized_height.max(1),
        FilterType::Triangle,
    );
    let mut canvas = RgbImage::from_pixel(416, 416, Rgb([114, 114, 114]));
    image::imageops::overlay(&mut canvas, &resized, 0, 0);
    let mut values = vec![0.0_f32; 3 * 416 * 416];
    for (index, pixel) in canvas.pixels().enumerate() {
        values[index] = f32::from(pixel[2]);
        values[416 * 416 + index] = f32::from(pixel[1]);
        values[2 * 416 * 416 + index] = f32::from(pixel[0]);
    }

    Tensor::from_array((vec![1, 3, 416, 416], values.into_boxed_slice()))
        .map(|tensor| (tensor, original_width, original_height, ratio))
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

fn pixel_values(image: &DynamicImage, channels: usize, channels_first: bool) -> Vec<f32> {
    if channels == 1 {
        return image
            .to_luma8()
            .pixels()
            .map(|pixel| f32::from(pixel[0]) / 255.0)
            .collect();
    }

    let image = image.to_rgb8();
    if channels_first {
        return (0..3)
            .flat_map(|channel| {
                image
                    .pixels()
                    .map(move |pixel| f32::from(pixel[channel]) / 255.0)
            })
            .collect();
    }

    image
        .pixels()
        .flat_map(|pixel| pixel.0.into_iter().map(|value| f32::from(value) / 255.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage};

    use super::pixel_values;

    #[test]
    fn rgb_values_follow_the_model_layout() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgb([255, 128, 0])
            } else {
                Rgb([0, 64, 255])
            }
        }));

        assert_eq!(
            pixel_values(&image, 3, false),
            [1.0, 128.0 / 255.0, 0.0, 0.0, 64.0 / 255.0, 1.0]
        );
        assert_eq!(
            pixel_values(&image, 3, true),
            [1.0, 0.0, 128.0 / 255.0, 64.0 / 255.0, 0.0, 1.0]
        );
    }
}
