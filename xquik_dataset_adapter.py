"""Normalize Xquik social export CSVs for the project notebook dataset."""

from __future__ import annotations

import argparse
import csv
import importlib
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol, cast

SentimentScorer = Callable[[str], str]


class MissingTextError(ValueError):
    """Raised when an export record has no usable post text."""


class SentimentAnalyzer(Protocol):
    """Describe the VADER method used by the adapter."""

    def polarity_scores(self, text: str) -> Mapping[str, float]:
        """Return sentiment scores for text."""


class TextWriter(Protocol):
    """Describe the text writer required by csv.DictWriter."""

    def write(self, data: str, /) -> int:
        """Write text and return the character count."""


DATASET_COLUMNS = (
    "Id",
    "Timestamp",
    "Text",
    "Sentiment",
    "User",
    "Platform",
    "Hashtags",
    "Country",
    "Likes",
    "Retweets",
)

COLUMN_ALIASES = {
    "Id": ("id", "tweet_id", "post_id", "status_id"),
    "Timestamp": (
        "timestamp",
        "tweet_created_at",
        "created_at",
        "published_at",
        "date",
    ),
    "Text": ("text", "full_text", "tweet_text", "content", "body"),
    "Sentiment": ("sentiment", "label", "polarity"),
    "User": ("user", "username", "x_username", "screen_name", "author", "handle"),
    "Platform": ("platform", "network"),
    "Hashtags": ("hashtags", "tags", "hashtag"),
    "Country": ("country", "user_country"),
    "Likes": ("likes", "like_count", "favorite_count", "favorites"),
    "Retweets": (
        "retweets",
        "retweet_count",
        "reposts",
        "repost_count",
        "share_count",
    ),
}

CSV_FORMULA_PATTERN = re.compile(r"^[\t\r\n=+\-@]|^ +[=+\-@]")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#[^\W#]+")


def _normalize_key(value: str) -> str:
    return value.lstrip("\ufeff").strip().casefold().replace(" ", "_")


def _indexed_row(row: Mapping[str, object]) -> dict[str, object]:
    return {_normalize_key(str(key)): value for key, value in row.items()}


def _first_value(indexed_row: Mapping[str, object], aliases: Sequence[str]) -> str:
    for alias in aliases:
        value = indexed_row.get(alias)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized and normalized.casefold() != "nan":
            return normalized
    return ""


def _first_integer(indexed_row: Mapping[str, object], aliases: Sequence[str]) -> int:
    value = _first_value(indexed_row, aliases)
    if not value:
        return 0
    try:
        parsed = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return 0
    if not parsed.is_finite():
        return 0
    return max(0, int(parsed))


def _neutralize_csv_formula(value: str) -> str:
    if CSV_FORMULA_PATTERN.match(value):
        return f"'{value}"
    return value


def _extract_hashtags(text: str) -> str:
    return " ".join(HASHTAG_PATTERN.findall(text))


def sentiment_label_from_compound(compound: float) -> str:
    """Map a VADER compound score to its documented sentiment label."""
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def create_vader_scorer() -> SentimentScorer:
    """Create the social-text sentiment scorer used for unlabeled exports."""
    try:
        module = importlib.import_module("vaderSentiment.vaderSentiment")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "VADER is required for unlabeled exports. "
            "Install requirements-xquik.txt first."
        ) from error

    analyzer_type = vars(module)["SentimentIntensityAnalyzer"]
    analyzer = cast(SentimentAnalyzer, analyzer_type())

    def score(text: str) -> str:
        compound = analyzer.polarity_scores(text).get("compound", 0.0)
        return sentiment_label_from_compound(compound)

    return score


def normalize_xquik_row(
    row: Mapping[str, object],
    sentiment_scorer: SentimentScorer | None = None,
) -> dict[str, object]:
    """Return one row that matches the notebook's documented dataset columns."""
    indexed_row = _indexed_row(row)
    text = _first_value(indexed_row, COLUMN_ALIASES["Text"])
    if not text:
        raise MissingTextError("Xquik export row is missing tweet text")

    sentiment = _first_value(indexed_row, COLUMN_ALIASES["Sentiment"])
    if not sentiment:
        if sentiment_scorer is None:
            raise RuntimeError("Unlabeled export row requires a sentiment scorer")
        sentiment = sentiment_scorer(text)
    hashtags = _first_value(indexed_row, COLUMN_ALIASES["Hashtags"])
    normalized = {
        "Id": _first_value(indexed_row, COLUMN_ALIASES["Id"]),
        "Timestamp": _first_value(indexed_row, COLUMN_ALIASES["Timestamp"]),
        "Text": text,
        "Sentiment": sentiment,
        "User": _first_value(indexed_row, COLUMN_ALIASES["User"]) or "Unknown",
        "Platform": _first_value(indexed_row, COLUMN_ALIASES["Platform"]) or "X",
        "Hashtags": hashtags or _extract_hashtags(text) or "None",
        "Country": _first_value(indexed_row, COLUMN_ALIASES["Country"]) or "Unknown",
        "Likes": _first_integer(indexed_row, COLUMN_ALIASES["Likes"]),
        "Retweets": _first_integer(indexed_row, COLUMN_ALIASES["Retweets"]),
    }
    return {
        key: _neutralize_csv_formula(value) if isinstance(value, str) else value
        for key, value in normalized.items()
    }


def normalize_xquik_rows(
    rows: Iterable[Mapping[str, object]],
    sentiment_scorer: SentimentScorer | None = None,
) -> list[dict[str, object]]:
    """Normalize rows and skip empty text records from export files."""
    normalized_rows = []
    for row in rows:
        try:
            normalized_rows.append(normalize_xquik_row(row, sentiment_scorer))
        except MissingTextError:
            continue
    return normalized_rows


def _validate_headers(fieldnames: Sequence[str] | None) -> None:
    normalized_fields = {_normalize_key(field) for field in (fieldnames or ())}
    if normalized_fields.isdisjoint(COLUMN_ALIASES["Text"]):
        raise ValueError("Input CSV is missing a tweet text column")


def _lazy_sentiment_scorer(
    sentiment_scorer: SentimentScorer | None,
) -> SentimentScorer:
    scorer = sentiment_scorer

    def score(text: str) -> str:
        nonlocal scorer
        if scorer is None:
            scorer = create_vader_scorer()
        return scorer(text)

    return score


def _write_normalized_rows(
    rows: Iterable[Mapping[str, object]],
    target: TextWriter,
    sentiment_scorer: SentimentScorer,
) -> int:
    writer = csv.DictWriter(target, fieldnames=DATASET_COLUMNS)
    writer.writeheader()

    count = 0
    for row in rows:
        try:
            normalized = normalize_xquik_row(row, sentiment_scorer)
        except MissingTextError:
            continue
        writer.writerow(normalized)
        count += 1
    return count


def _write_atomically(
    rows: Iterable[Mapping[str, object]],
    output_path: Path,
    sentiment_scorer: SentimentScorer,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    committed = False
    try:
        with NamedTemporaryFile(
            "w",
            delete=False,
            dir=output_path.parent,
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        ) as target:
            temporary_path = Path(target.name)
            count = _write_normalized_rows(rows, target, sentiment_scorer)

        temporary_path.replace(output_path)
        committed = True
        return count
    finally:
        if not committed and temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def convert_csv(
    input_path: Path,
    output_path: Path,
    sentiment_scorer: SentimentScorer | None = None,
) -> int:
    """Convert an Xquik CSV to the notebook schema."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output CSV paths must differ")

    with input_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        _validate_headers(reader.fieldnames)
        return _write_atomically(
            reader,
            output_path,
            _lazy_sentiment_scorer(sentiment_scorer),
        )


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize an Xquik CSV for the sentiment analysis notebook."
    )
    parser.add_argument("input_csv", type=Path, help="Xquik CSV export")
    parser.add_argument("output_csv", type=Path, help="normalized output CSV")
    arguments = parser.parse_args(argv[1:])

    try:
        count = convert_csv(arguments.input_csv, arguments.output_csv)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Conversion failed. {error}", file=sys.stderr)
        return 1

    print(f"Wrote {count} normalized rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
