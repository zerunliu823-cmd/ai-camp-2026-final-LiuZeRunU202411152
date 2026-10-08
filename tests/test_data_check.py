import tempfile
import unittest
from pathlib import Path

from main import (
    build_random_forest,
    load_arff_summary,
    load_dataset,
    majority_label,
    validate_feature_values,
)


TINY_ARFF = """@RELATION tiny_shapes
@ATTRIBUTE Length REAL
@ATTRIBUTE Width REAL
@ATTRIBUTE Class {ROUND,LONG}
@DATA
1.0,1.0,ROUND
2.0,1.0,LONG
"""


class LoadArffSummaryTests(unittest.TestCase):
    def test_counts_rows_columns_classes_and_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.arff"
            path.write_text(TINY_ARFF, encoding="utf-8")

            summary = load_arff_summary(path)

        self.assertEqual(summary.rows, 2)
        self.assertEqual(summary.columns, 3)
        self.assertEqual(summary.missing_values, 0)
        self.assertEqual(summary.class_counts, {"ROUND": 1, "LONG": 1})

    def test_rejects_a_row_with_the_wrong_number_of_values(self) -> None:
        broken = TINY_ARFF.replace("2.0,1.0,LONG", "2.0,LONG")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.arff"
            path.write_text(broken, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected 3"):
                load_arff_summary(path)

    def test_majority_label_uses_the_most_common_class(self) -> None:
        self.assertEqual(
            majority_label(["ROUND", "LONG", "ROUND"]),
            "ROUND",
        )

    def test_majority_label_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "没有类别标签"):
            majority_label([])

    def test_load_dataset_separates_numeric_inputs_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.arff"
            path.write_text(TINY_ARFF, encoding="utf-8")

            features, labels = load_dataset(path)

        self.assertEqual(features, [[1.0, 1.0], [2.0, 1.0]])
        self.assertEqual(labels, ["ROUND", "LONG"])

    def test_random_forest_configuration_is_fixed(self) -> None:
        model = build_random_forest()
        self.assertEqual(model.n_estimators, 300)
        self.assertEqual(model.random_state, 42)

    def test_prediction_requires_exactly_sixteen_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "需要 16 个特征值"):
            validate_feature_values([1.0] * 15)

    def test_prediction_rejects_a_non_finite_value(self) -> None:
        values = [1.0] * 16
        values[5] = float("nan")
        with self.assertRaisesRegex(ValueError, "Eccentricity"):
            validate_feature_values(values)


if __name__ == "__main__":
    unittest.main()



