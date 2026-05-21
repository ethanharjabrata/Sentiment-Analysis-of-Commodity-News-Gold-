import pandas as pd


LABEL_MAP = {
    "negative": 0,
    "positive": 1,
    "neutral": 2,
    "none": 3,
}

CLASSES = ["positive", "negative", "neutral", "none"]

COLORS = {
    "positive": "#66bb6a",
    "negative": "#ef5350",
    "neutral": "#42a5f5",
    "none": "#bdbdbd",
}


def load_data(csv_path: str) -> pd.DataFrame:
    """Loads the gold sentiment dataset and adds a numeric label column.

    Args:
        - csv_path (str): Path to the raw CSV file.

    Returns:
        - data (pd.DataFrame): The dataset with an added integer 'label' column.
    """
    data = pd.read_csv(filepath_or_buffer=csv_path)
    data["label"] = data["Price Sentiment"].map(func=LABEL_MAP)
    return data


def summarise_classes(data: pd.DataFrame) -> None:
    """Prints counts and percentages for each sentiment class.

    Args:
        - data (pd.DataFrame): Dataset with a 'Price Sentiment' column.

    Returns:
        - None
    """
    counts = data["Price Sentiment"].value_counts()
    total = len(data)

    print("=== Class Distribution ===")

    for label, count in counts.items():
        pct = 100 * count / total
        print(f"  {label:<10} {count:>5}  ({pct:.1f}%)")

    print(f"  {'TOTAL':<10} {total:>5}\n")


def summarise_headline_lengths(data: pd.DataFrame) -> None:
    """Prints min, max, mean, and median headline word counts split by class.

    Args:
        - data (pd.DataFrame): Dataset with 'News' and 'Price Sentiment' columns.

    Returns:
        - None
    """
    data = data.copy()
    data["word_count"] = data["News"].str.split().str.len()

    print("=== Headline Word Count by Class ===")

    for cls in CLASSES:
        wc = data[data["Price Sentiment"] == cls]["word_count"]
        print(
            f"  {cls:<10}  min={wc.min()}, max={wc.max()}, "
            f"mean={wc.mean():.1f}, median={wc.median():.0f}"
        )

    print()


def summarise_past_future(data: pd.DataFrame) -> None:
    """Prints a Past/Future information breakdown for each sentiment class.

    The dataset flags each headline as past-reporting (Past Information=1) or
    forward-looking (Future Information=1). This summary reveals whether the
    training set mixes historical fact-reports with genuine predictions — a
    distinction that the model training code does not filter for.

    Args:
        - data (pd.DataFrame): Dataset with 'Price Sentiment', 'Past Information',
            and 'Future Information' columns.

    Returns:
        - None
    """
    print("=== Past vs Future Information Breakdown by Class ===")
    print("  Past=1 -> headline reports a historical price move.")
    print("  Future=1 -> headline makes a forward-looking prediction.\n")

    for cls in CLASSES:
        subset = data[data["Price Sentiment"] == cls]
        n = len(subset)
        n_past = int(subset["Past Information"].sum())
        n_future = int(subset["Future Information"].sum())
        n_both = int(
            (
                (subset["Past Information"] == 1) & (subset["Future Information"] == 1)
            ).sum()
        )

        n_neither = int(
            (
                (subset["Past Information"] == 0) & (subset["Future Information"] == 0)
            ).sum()
        )

        print(f"  {cls} (n={n}):")
        print(
            f"    Past only:   {n_past - n_both:>5}  ({100*(n_past - n_both)/n:.1f}%)"
        )
        print(
            f"    Future only: {n_future - n_both:>5}  ({100*(n_future - n_both)/n:.1f}%)"
        )
        print(f"    Both:        {n_both:>5}  ({100*n_both/n:.1f}%)")
        print(f"    Neither:     {n_neither:>5}  ({100*n_neither/n:.1f}%)")
        print()


def show_sample_headlines(data: pd.DataFrame, n: int) -> None:
    """Prints n sample headlines per class for a quick eyeball check.

    Args:
        - data (pd.DataFrame): Dataset with 'News' and 'Price Sentiment' columns.
        - n (int): Number of sample headlines to show per class.

    Returns:
        - None
    """
    print(f"=== {n} Sample Headlines per Class ===")

    for cls in CLASSES:
        print(f"\n  {cls.upper()}:")
        samples = (
            data[data["Price Sentiment"] == cls]["News"]
            .sample(n=n, random_state=42)
            .tolist()
        )

        for headline in samples:
            print(f"    - {headline}")

    print()
