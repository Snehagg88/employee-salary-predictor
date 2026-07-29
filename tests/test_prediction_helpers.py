import pandas as pd

from app import (
    build_prediction_explanation,
    validate_batch_dataframe,
)


def test_validate_batch_dataframe_detects_missing_and_invalid():
    df = pd.DataFrame(
        [
            {
                "age": 30,
                "workclass": "Private",
                "education": "Bachelors",
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "gender": "Male",
                "capital-gain": 1000,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            },
            {
                "age": 10,
                "workclass": "Private",
                "education": "Bachelors",
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "gender": "Male",
                "capital-gain": 1000,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            },
        ]
    )

    result = validate_batch_dataframe(df)
    assert result["missing_columns"] == []
    assert len(result["valid_rows"]) == 1
    assert result["invalid_rows"][0]["reason"] == "age_out_of_range"


def test_build_prediction_explanation_emphasizes_factors():
    factors = [
        ("education", 0.42),
        ("occupation", 0.31),
        ("hours-per-week", 0.17),
    ]
    explanation = build_prediction_explanation(" >50K", factors, 0.91)
    assert "higher-income" in explanation.lower()
    assert "education" in explanation.lower()
