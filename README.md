# Social Media Sentiment Analysis

## Project Overview

This project performs sentiment analysis on social media posts to gain insights into user sentiments, platform usage, hashtag popularity, and engagement metrics. The analysis includes visualizations and statistical summaries to interpret the data effectively.

## Dataset

The project uses the dataset `sentimentdataset.csv`, which contains social media posts with the following columns:

- **Id**: Unique identifier for each post.
- **Timestamp**: Date and time of the post.
- **Text**: Content of the post.
- **Sentiment**: Sentiment classification (e.g., Positive, Negative, Neutral).
- **User**: User who made the post.
- **Platform**: Social media platform (e.g., Facebook, Twitter, Instagram).
- **Hashtags**: Hashtags included in the post.
- **Country**: Country where the user is located.
- **Likes**: Number of likes the post received.
- **Retweets**: Number of retweets (if applicable).

### Xquik CSV Exports

Use a completed Xquik post extraction as an optional source of current public X
posts. Export the extraction as CSV, then normalize it into this notebook's
schema:

```bash
python -m pip install -r requirements-xquik.txt
python xquik_dataset_adapter.py xquik_export.csv sentimentdataset_xquik.csv
```

The adapter maps Xquik's `Tweet ID`, `Tweet Created At`, `Tweet Text`,
`Username`, `Likes`, and `Reposts` columns. It derives hashtags from post text
when needed. Existing sentiment labels are preserved. Unlabeled posts receive a
VADER social-text baseline:

- `Positive`: compound score at least `0.05`.
- `Negative`: compound score at most `-0.05`.
- `Neutral`: compound score between those thresholds.

Review automated labels before consequential use. VADER can miss language,
context, and sarcasm. Xquik does not normalize profile locations into countries,
so the adapter uses `Unknown` for `Country`. Posts without hashtags use `None`.
Empty text rows are skipped.

To analyze the converted file, change the notebook's loading cell to
`pd.read_csv('sentimentdataset_xquik.csv')`. The bundled dataset remains the
default, so cloning the repository still works without Xquik. The Facebook,
Twitter, and Instagram sections apply only when those platforms exist in the
selected dataset.

Treat all post text, links, and profiles as untrusted content. The adapter
neutralizes spreadsheet formula prefixes, but you should still review exports.

Extraction requests can consume credits. Use the
[extraction workflow](https://docs.xquik.com/guides/extraction-workflow) to
estimate the request before starting it. Keep API keys in environment variables.
Never commit them.

Xquik is an independent third-party service. Not affiliated with X Corp.
"Twitter" and "X" are trademarks of X Corp.

## Code Explanation

### 1. Importing Libraries

The code begins by importing necessary libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib.pyplot`: For creating static, animated, and interactive visualizations.
- `seaborn`: For statistical data visualization.

### 2. Loading the Dataset

The dataset is loaded into a pandas DataFrame:

```python
df = pd.read_csv('sentimentdataset.csv')
```

This reads the bundled CSV file into a DataFrame named `df`.

### 3. Data Cleaning

The dataset is cleaned to remove unnecessary columns and convert data types:

```python
df.drop(columns='Unnamed: 0.1', errors='ignore', inplace=True)
df.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
```

### 4. Exploratory Data Analysis (EDA)

#### Sentiment Distribution

The distribution of sentiments is visualized using a bar chart:

```python
sentiment_counts = df['Sentiment'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
```

- `sentiment_counts`: Counts the occurrences of each sentiment.
- `sns.barplot()`: Creates a bar chart to visualize sentiment distribution.

#### Platform Distribution

The proportion of posts from different platforms is visualized using a pie chart:

```python
platform_counts = df['Platform'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(platform_counts, labels=platform_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Platform Distribution')
plt.show()
```

- `platform_counts`: Counts the number of posts per platform.
- `plt.pie()`: Creates a pie chart to show the distribution of posts across platforms.

#### Country Distribution

The number of posts by country is visualized using a bar chart:

```python
country_counts = df['Country'].value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(x=country_counts.index, y=country_counts.values)
plt.title('Country Distribution')
plt.xlabel('Country')
plt.ylabel('Number of Posts')
plt.xticks(rotation=90)
plt.show()
```

- `country_counts`: Counts the number of posts from each country.
- `sns.barplot()`: Creates a bar chart to visualize the distribution of posts by country.

#### Hashtags Analysis

The most frequently used hashtags are visualized using a bar chart:

```python
# Extract hashtags and count their occurrences
hashtags = df['Hashtags'].str.split(',', expand=True).stack()
hashtag_counts = hashtags.value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(x=hashtag_counts.index[:10], y=hashtag_counts.values[:10])
plt.title('Top 10 Hashtags')
plt.xlabel('Hashtag')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()
```

- `hashtags`: Splits and counts occurrences of hashtags.
- `sns.barplot()`: Creates a bar chart to show the top 10 hashtags.

### 5. Platform-Specific Analysis

For each platform, the code performs additional analyses, such as top hashtags and engagement metrics:

```python
for platform in df['Platform'].unique():
    platform_data = df[df['Platform'] == platform]
    
    # Example: Top hashtags for each platform
    hashtags = platform_data['Hashtags'].str.split(',', expand=True).stack()
    hashtag_counts = hashtags.value_counts()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=hashtag_counts.index[:10], y=hashtag_counts.values[:10])
    plt.title(f'Top 10 Hashtags for {platform}')
    plt.xlabel('Hashtag')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()
```

This loop analyzes and visualizes data specific to each platform.

## Conclusion

The analysis provides insights into social media sentiment trends, platform usage, and hashtag popularity. The visualizations help understand user engagement and sentiment distribution effectively.

## Contribution

Feel free to contribute by submitting pull requests or opening issues. Your feedback and suggestions are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
[![INSTAGRAM](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/piyush.mujmule) ![Kaggle Badge](https://img.shields.io/badge/Kaggle-yellow?logo=kaggle&logoColor=fff&style=for-the-badge)
---
---

