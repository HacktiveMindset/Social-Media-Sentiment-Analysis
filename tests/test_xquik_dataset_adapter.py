import csv
import json
import tempfile
import unittest
from pathlib import Path

from xquik_dataset_adapter import (
    DATASET_COLUMNS,
    convert_csv,
    normalize_xquik_row,
    normalize_xquik_rows,
    sentiment_label_from_compound,
)


class XquikDatasetAdapterTest(unittest.TestCase):
    def test_normalizes_xquik_export_headers_to_project_schema(self):
        row = normalize_xquik_row(
            {
                "Tweet ID": "1842",
                "Tweet Created At": "2026-07-05T12:00:00Z",
                "Tweet Text": "Xquik export works well for #analytics and #research",
                "Username": "analyst",
                "Likes": "1,205",
                "Reposts": "17",
                "Source": "X for iPhone",
            },
            lambda text: "Positive",
        )

        self.assertEqual(row["Id"], "1842")
        self.assertEqual(row["Timestamp"], "2026-07-05T12:00:00Z")
        self.assertEqual(
            row["Text"], "Xquik export works well for #analytics and #research"
        )
        self.assertEqual(row["Sentiment"], "Positive")
        self.assertEqual(row["User"], "analyst")
        self.assertEqual(row["Platform"], "X")
        self.assertEqual(row["Hashtags"], "#analytics #research")
        self.assertEqual(row["Country"], "Unknown")
        self.assertEqual(row["Likes"], 1205)
        self.assertEqual(row["Retweets"], 17)

    def test_preserves_enriched_sentiment_and_country(self):
        row = normalize_xquik_row(
            {
                "tweet_text": "Useful launch notes",
                "sentiment": "Positive",
                "country": "Canada",
                "hashtags": "#launch,#research",
            }
        )

        self.assertEqual(row["Sentiment"], "Positive")
        self.assertEqual(row["Country"], "Canada")
        self.assertEqual(row["Hashtags"], "#launch,#research")

    def test_maps_vader_thresholds_to_project_labels(self):
        self.assertEqual(sentiment_label_from_compound(0.05), "Positive")
        self.assertEqual(sentiment_label_from_compound(0.049), "Neutral")
        self.assertEqual(sentiment_label_from_compound(-0.049), "Neutral")
        self.assertEqual(sentiment_label_from_compound(-0.05), "Negative")

    def test_skips_rows_without_text(self):
        rows = normalize_xquik_rows(
            [
                {"tweet_id": "1", "full_text": ""},
                {"tweet_id": "2", "text": "Useful launch notes"},
            ],
            lambda text: "Positive",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Id"], "2")
        self.assertEqual(rows[0]["Sentiment"], "Positive")

    def test_rejects_invalid_and_negative_engagement_counts(self):
        row = normalize_xquik_row(
            {
                "tweet_text": "Metrics should remain valid",
                "likes": "Infinity",
                "reposts": "-3",
            },
            lambda text: "Neutral",
        )

        self.assertEqual(row["Likes"], 0)
        self.assertEqual(row["Retweets"], 0)

    def test_neutralizes_spreadsheet_formula_prefixes(self):
        for value in (
            '=HYPERLINK("https://example.invalid")',
            "+SUM(1+1)",
            "-2+3",
            "@SUM(1+1)",
            "  =SUM(1+1)",
            "\t=SUM(1+1)",
        ):
            with self.subTest(value=value):
                row = normalize_xquik_row({"tweet_text": value}, lambda text: "Neutral")
                self.assertEqual(row["Text"], f"'{value.strip()}")

    def test_converts_csv_with_stable_header_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "xquik.csv"
            target_path = Path(tmp_dir) / "output" / "sentimentdataset_xquik.csv"
            source_path.write_text(
                "\ufeffTweet ID,Tweet Created At,Tweet Text,Username,Likes,Reposts\n"
                "42,2026-07-05T12:00:00Z,Great sentiment signal,researcher,8,3\n",
                encoding="utf-8",
            )

            count = convert_csv(source_path, target_path, lambda text: "Positive")

            self.assertEqual(count, 1)
            self.assertTrue(target_path.exists())
            with target_path.open(newline="", encoding="utf-8") as target:
                reader = csv.DictReader(target)
                self.assertEqual(reader.fieldnames, list(DATASET_COLUMNS))
                rows = list(reader)

        self.assertEqual(rows[0]["Id"], "42")
        self.assertEqual(rows[0]["Platform"], "X")
        self.assertEqual(rows[0]["Sentiment"], "Positive")
        self.assertEqual(rows[0]["Timestamp"], "2026-07-05T12:00:00Z")
        self.assertEqual(rows[0]["Retweets"], "3")

    def test_rejects_wrong_schema_without_overwriting_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "wrong.csv"
            target_path = Path(tmp_dir) / "existing.csv"
            source_path.write_text("Name,Value\nexample,1\n", encoding="utf-8")
            target_path.write_text("keep this output\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing a tweet text column"):
                convert_csv(source_path, target_path)

            self.assertEqual(
                target_path.read_text(encoding="utf-8"), "keep this output\n"
            )

    def test_preserves_output_when_sentiment_scoring_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "xquik.csv"
            target_path = Path(tmp_dir) / "existing.csv"
            source_path.write_text(
                "Tweet ID,Tweet Text\n42,Unlabeled post\n",
                encoding="utf-8",
            )
            target_path.write_text("keep this output\n", encoding="utf-8")

            def fail_scoring(_text: str) -> str:
                raise RuntimeError("scoring failed")

            with self.assertRaisesRegex(RuntimeError, "scoring failed"):
                convert_csv(source_path, target_path, fail_scoring)

            self.assertEqual(
                target_path.read_text(encoding="utf-8"), "keep this output\n"
            )
            self.assertEqual(list(Path(tmp_dir).glob(".existing.csv.*.tmp")), [])

    def test_rejects_using_the_input_as_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "xquik.csv"
            source_path.write_text(
                "Tweet ID,Tweet Text\n42,Unlabeled post\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "paths must differ"):
                convert_csv(source_path, source_path, lambda text: "Neutral")

            self.assertIn("Unlabeled post", source_path.read_text(encoding="utf-8"))

    def test_notebook_loads_the_bundled_dataset(self):
        notebook_path = (
            Path(__file__).parents[1] / "social-media-sentiments-analysis.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        loading_cell = notebook["cells"][2]["source"]
        cleanup_cell = notebook["cells"][3]["source"]

        self.assertIn("pd.read_csv('sentimentdataset.csv')", loading_cell)
        self.assertNotIn("/kaggle/input/", loading_cell)
        self.assertIn("errors='ignore'", cleanup_cell)


if __name__ == "__main__":
    unittest.main()
