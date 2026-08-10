use ort::value::DynValue;
use pyo3::exceptions::PyRuntimeError;

pub struct DecodedText {
    pub text: String,
    pub confidence: Option<f32>,
}

pub fn decode_text(
    output: &DynValue,
    characters: &[String],
    allowed_characters: Option<&str>,
    calculate_confidence: bool,
) -> Result<DecodedText, pyo3::PyErr> {
    if characters.is_empty() {
        return Err(PyRuntimeError::new_err("the character map is empty"));
    }

    let (shape, values) = output
        .try_extract_tensor::<f32>()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    if shape.last() != Some(&(characters.len() as i64)) {
        return Err(PyRuntimeError::new_err(format!(
            "the model returned {shape}, but its last dimension must match the {}-entry character map",
            characters.len()
        )));
    }

    decode_scores(values, characters, allowed_characters, calculate_confidence)
        .map_err(PyRuntimeError::new_err)
}

fn decode_scores(
    values: &[f32],
    characters: &[String],
    allowed_characters: Option<&str>,
    calculate_confidence: bool,
) -> Result<DecodedText, String> {
    if values.is_empty() {
        return Err("the model returned an empty output".to_owned());
    }
    if !values.len().is_multiple_of(characters.len()) {
        return Err("the model output does not align with the character map".to_owned());
    }
    let allowed_indices = allowed_characters.map(|allowed| {
        characters
            .iter()
            .enumerate()
            .filter_map(|(index, character)| {
                (index == 0 || character.chars().all(|value| allowed.contains(value)))
                    .then_some(index)
            })
            .collect::<Vec<_>>()
    });

    let mut text = String::new();
    let mut confidence_log_sum = 0.0_f32;
    let mut confidence_count = 0_u32;
    let mut previous_index = usize::MAX;
    for scores in values.chunks_exact(characters.len()) {
        let mut maximum_score = f32::NEG_INFINITY;
        let mut maximum_index = 0;
        for (index, score) in scores.iter().copied().enumerate() {
            if !score.is_finite() {
                return Err("the model returned a non-finite score".to_owned());
            }
            if score >= maximum_score {
                maximum_score = score;
                maximum_index = index;
            }
        }
        let index = allowed_indices.as_ref().map_or(maximum_index, |indices| {
            indices
                .iter()
                .copied()
                .max_by(|left, right| scores[*left].total_cmp(&scores[*right]))
                .expect("the blank character is always a candidate")
        });

        if index != 0
            && index != previous_index
            && let Some(character) = characters.get(index)
        {
            text.push_str(character);
            if calculate_confidence {
                confidence_log_sum +=
                    softmax_probability(scores, scores[index], maximum_score).ln();
                confidence_count += 1;
            }
        }
        previous_index = index;
    }

    let confidence = (calculate_confidence && confidence_count > 0)
        .then(|| (confidence_log_sum / confidence_count as f32).exp());

    Ok(DecodedText { text, confidence })
}

fn softmax_probability(scores: &[f32], selected_score: f32, maximum_score: f32) -> f32 {
    let denominator = scores
        .iter()
        .map(|score| (score - maximum_score).exp())
        .sum::<f32>();
    ((selected_score - maximum_score).exp() / denominator).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::decode_scores;

    fn characters() -> Vec<String> {
        ["", "A", "1"].into_iter().map(str::to_owned).collect()
    }

    #[test]
    fn allowed_characters_constrain_selection() {
        let decoded = decode_scores(&[0.0, 10.0, 9.0], &characters(), Some("1"), true).unwrap();

        assert_eq!(decoded.text, "1");
        assert!(decoded.confidence.is_some());
    }

    #[test]
    fn ctc_decoding_removes_repeats_and_blanks() {
        let scores = [
            0.0, 4.0, 1.0, // A
            0.0, 4.0, 1.0, // repeated A
            4.0, 0.0, 0.0, // blank
            0.0, 1.0, 4.0, // 1
        ];
        let decoded = decode_scores(&scores, &characters(), None, false).unwrap();

        assert_eq!(decoded.text, "A1");
        assert_eq!(decoded.confidence, None);
    }

    #[test]
    fn malformed_scores_are_rejected() {
        assert!(decode_scores(&[0.0, 1.0], &characters(), None, false).is_err());
        assert!(decode_scores(&[0.0, f32::NAN, 1.0], &characters(), None, false).is_err());
    }
}
