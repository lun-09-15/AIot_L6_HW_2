"""Check the Kaggle archives shipped with the website, independently of network access."""

import csv
import io
from pathlib import Path
import unittest
import zipfile


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class PackagedDataTests(unittest.TestCase):
    def test_builtin_archives_are_readable_and_have_expected_targets(self):
        cases = [
            ("flood_prediction_dataset.zip", "train.csv", 22, "FloodProbability"),
            ("life_expectancy_who.zip", "Life Expectancy Data.csv", 22, "Life expectancy"),
            ("student_performance.zip", "student-por.csv", 33, "G3"),
        ]
        for archive_name, csv_name, expected_columns, target in cases:
            with self.subTest(archive=archive_name):
                with zipfile.ZipFile(DATA_DIR / archive_name) as archive:
                    self.assertIn(csv_name, archive.namelist())
                    self.assertIsNone(archive.testzip(), "ZIP integrity check failed")
                    with archive.open(csv_name) as raw:
                        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                        reader = csv.reader(text)
                        columns = [column.strip() for column in next(reader)]
                        first_row = next(reader)
                    self.assertEqual(len(columns), expected_columns)
                    self.assertEqual(len(first_row), expected_columns)
                    self.assertIn(target, columns)


if __name__ == "__main__":
    unittest.main()
