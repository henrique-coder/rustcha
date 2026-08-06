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

    decode_scores(values, characters, allowed_characters).map_err(PyRuntimeError::new_err)
}

fn decode_scores(
    values: &[f32],
    characters: &[String],
    allowed_characters: Option<&str>,
) -> Result<DecodedText, String> {
    if values.is_empty() {
        return Err("the model returned an empty output".to_owned());
    }
    if !values.len().is_multiple_of(characters.len()) {
        return Err("the model output does not align with the character map".to_owned());
    }
    if values.iter().any(|score| !score.is_finite()) {
        return Err("the model returned a non-finite score".to_owned());
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
    let mut confidences = Vec::new();
    let mut previous_index = usize::MAX;
    for scores in values.chunks_exact(characters.len()) {
        let index = match &allowed_indices {
            Some(indices) => indices
                .iter()
                .copied()
                .max_by(|left, right| scores[*left].total_cmp(&scores[*right])),
            None => {
                (0..characters.len()).max_by(|left, right| scores[*left].total_cmp(&scores[*right]))
            }
        }
        .expect("the blank character is always a candidate");

        if index != 0
            && index != previous_index
            && let Some(character) = characters.get(index)
        {
            text.push_str(character);
            confidences.push(softmax_probability(scores, scores[index]));
        }
        previous_index = index;
    }

    let confidence = (!confidences.is_empty()).then(|| {
        let mean_log_probability =
            confidences.iter().map(|value| value.ln()).sum::<f32>() / confidences.len() as f32;
        mean_log_probability.exp()
    });

    Ok(DecodedText { text, confidence })
}

fn softmax_probability(scores: &[f32], selected_score: f32) -> f32 {
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denominator = scores
        .iter()
        .map(|score| (score - maximum).exp())
        .sum::<f32>();
    ((selected_score - maximum).exp() / denominator).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::decode_scores;

    fn characters() -> Vec<String> {
        ["", "A", "1"].into_iter().map(str::to_owned).collect()
    }

    #[test]
    fn allowed_characters_constrain_selection() {
        let decoded = decode_scores(&[0.0, 10.0, 9.0], &characters(), Some("1")).unwrap();

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
        let decoded = decode_scores(&scores, &characters(), None).unwrap();

        assert_eq!(decoded.text, "A1");
    }

    #[test]
    fn malformed_scores_are_rejected() {
        assert!(decode_scores(&[0.0, 1.0], &characters(), None).is_err());
        assert!(decode_scores(&[0.0, f32::NAN, 1.0], &characters(), None).is_err());
    }
}
