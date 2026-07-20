import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create Images folder
os.makedirs("Images", exist_ok=True)

# ==========================================
# Spotify Dark Theme
# ==========================================

SPOTIFY_GREEN = "#1DB954"
SPOTIFY_BLACK = "#191414"
SPOTIFY_DARK = "#121212"
SPOTIFY_GRAY = "#B3B3B3"

plt.style.use("dark_background")

plt.rcParams["figure.facecolor"] = SPOTIFY_DARK
plt.rcParams["axes.facecolor"] = SPOTIFY_BLACK
plt.rcParams["axes.edgecolor"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["text.color"] = "white"
plt.rcParams["grid.color"] = "#444444"
plt.rcParams["grid.alpha"] = 0.3

# Load Dataset
df = pd.read_csv("Dataset/spotify.csv")

print("=" * 60)
print("SPOTIFY DATA ANALYSIS")
print("=" * 60)

print("\nFirst 5 Rows")
print(df.head())

print("\nShape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
# -----------------------------
# Data Cleaning
# -----------------------------
numeric_columns = [
    "streams",
    "bpm",
    "danceability_%",
    "valence_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df.fillna(df.median(numeric_only=True), inplace=True)

print("\nCleaning Completed Successfully!")

# ==========================================
# GRAPH 1 - Top 10 Artists
# ==========================================

top_artists = (
    df["artists"]
    .value_counts()
    .head(10)
    .sort_values()
)

plt.figure(figsize=(12, 6))

plt.barh(
    range(len(top_artists)),
    top_artists.values,
    color=SPOTIFY_GREEN
)

plt.yticks(range(len(top_artists)), [a[:25] + "..." if len(a) > 25 else a for a in top_artists.index])

plt.title("Top 10 Artists")
plt.xlabel("Number of Songs")

plt.savefig("Images/top_artists.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ==========================================
# GRAPH 2 - Top Genres
# ==========================================

plt.figure(figsize=(12,6), facecolor=SPOTIFY_BG)

df["track_genre"].value_counts().head(10).sort_values().plot(
    kind="barh",
    color=SPOTIFY_BLACK
)

plt.title("Top 10 Genres")
plt.xlabel("Number of Songs")
plt.tight_layout()
plt.savefig("Images/top_genres.png", dpi=300)
plt.show()

# ==========================================
# GRAPH 3 - Top 10 Popular Songs
# ==========================================

top_popular = df.sort_values(
    by="popularity",
    ascending=False
).head(10)

plt.figure(figsize=(12,6), facecolor=SPOTIFY_BG)

plt.barh(
    top_popular["track_name"],
    top_popular["popularity"],
    color=SPOTIFY_GREEN
)

plt.title("Top 10 Most Popular Songs")
plt.xlabel("Popularity")
plt.tight_layout()
plt.savefig("Images/top_popular_songs.png", dpi=300)
plt.show()

# ==========================================
# GRAPH 4 - Popularity Distribution
# ==========================================

plt.figure(figsize=(10,6), facecolor=SPOTIFY_BG)

plt.hist(
    df["popularity"],
    bins=20,
    color=SPOTIFY_GREEN,
    edgecolor="black"
)

plt.title("Popularity Distribution")
plt.xlabel("Popularity")
plt.ylabel("Songs")
plt.tight_layout()
plt.savefig("Images/popularity_distribution.png", dpi=300)
plt.show()

# ==========================================
# GRAPH 5 - Danceability Distribution
# ==========================================

plt.figure(figsize=(10,6), facecolor=SPOTIFY_BG)

plt.hist(
    df["danceability"],
    bins=20,
    color="#1ED760",
    edgecolor="black"
)

plt.title("Danceability Distribution")
plt.xlabel("Danceability")
plt.ylabel("Songs")
plt.tight_layout()
plt.savefig("Images/danceability_distribution.png", dpi=300)
plt.show()

# ==========================================
# GRAPH 6 - Energy Distribution
# ==========================================

plt.figure(figsize=(10,6), facecolor=SPOTIFY_BG)

plt.hist(
    df["energy"],
    bins=20,
    color=SPOTIFY_BLACK,
    edgecolor="white"
)

plt.title("Energy Distribution")
plt.xlabel("Energy")
plt.ylabel("Songs")
plt.tight_layout()
plt.savefig("Images/energy_distribution.png", dpi=300)
plt.show()

# ==========================================
# GRAPH 7 - Explicit Songs
# ==========================================

plt.figure(figsize=(6,6), facecolor=SPOTIFY_BG)

df["explicit"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    colors=[SPOTIFY_GREEN, SPOTIFY_BLACK]
)

plt.ylabel("")
plt.title("Explicit vs Non Explicit Songs")
plt.tight_layout()
plt.savefig("Images/explicit_songs.png", dpi=300)
plt.show()

# ==========================================
# GRAPH 8 - Danceability vs Energy
# ==========================================

plt.figure(figsize=(10,6), facecolor=SPOTIFY_BG)

plt.scatter(
    df["danceability"],
    df["energy"],
    color=SPOTIFY_GREEN,
    alpha=0.5
)

plt.title("Danceability vs Energy")
plt.xlabel("Danceability")
plt.ylabel("Energy")
plt.tight_layout()
plt.savefig("Images/danceability_vs_energy.png", dpi=300)
plt.show()

# ==========================================
# GRAPH 9 - Correlation Heatmap
# ==========================================

plt.figure(figsize=(10,8), facecolor=SPOTIFY_BG)

sns.heatmap(
    df[
        [
            "popularity",
            "danceability",
            "energy",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo"
        ]
    ].corr(),
    annot=True,
    cmap="Greens"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("Images/correlation_heatmap.png", dpi=300)
plt.show()

# ==========================================
# GRAPH 10 - Top Albums
# ==========================================

plt.figure(figsize=(12,6), facecolor=SPOTIFY_BG)

df["album_name"].value_counts().head(10).sort_values().plot(
    kind="barh",
    color="#169C46"
)

plt.title("Top 10 Albums")
plt.xlabel("Number of Songs")
plt.tight_layout()
plt.savefig("Images/top_albums.png", dpi=300)
plt.show()

print("\nAll graphs generated successfully!")
print("Graphs saved inside Images folder.")