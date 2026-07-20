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

plt.figure(figsize=(12,6), facecolor="#191414")
ax = plt.gca()
ax.set_facecolor("#191414")

top_artists = df["artists"].value_counts().head(10).sort_values()

plt.barh(
    [a[:25] + "..." if len(a) > 25 else a for a in top_artists.index],
    top_artists.values,
    color="#1DB954",
    edgecolor="white"
)

plt.title("Top 10 Artists", color="white", fontsize=16)
plt.xlabel("Number of Songs", color="white")

plt.xticks(color="white")
plt.yticks(color="white")

for spine in ax.spines.values():
    spine.set_color("white")

plt.grid(axis="x", color="white", alpha=0.2)

plt.tight_layout()
plt.savefig("Images/top_artists.png", dpi=300, facecolor="#191414")
plt.show()


# ==========================================
# GRAPH 2 - Top Genres
# ==========================================

plt.figure(figsize=(12,6), facecolor="#191414")
ax = plt.gca()
ax.set_facecolor("#191414")

genres = df["track_genre"].value_counts().head(10).sort_values()

plt.barh(
    genres.index,
    genres.values,
    color="#1DB954",
    edgecolor="white"
)

plt.title("Top 10 Genres", color="white")
plt.xlabel("Number of Songs", color="white")

plt.xticks(color="white")
plt.yticks(color="white")

for spine in ax.spines.values():
    spine.set_color("white")

plt.grid(axis="x", color="white", alpha=0.2)

plt.tight_layout()
plt.savefig("Images/top_genres.png", dpi=300, facecolor="#191414")
plt.show()


# ==========================================
# GRAPH 3 - Top Popular Songs
# ==========================================

plt.figure(figsize=(12,6), facecolor="#191414")
ax = plt.gca()
ax.set_facecolor("#191414")

top_popular = df.sort_values(by="popularity", ascending=False).head(10)

plt.barh(
    [x[:30] + "..." if len(x) > 30 else x for x in top_popular["track_name"]],
    top_popular["popularity"],
    color="#1DB954",
    edgecolor="white"
)

plt.title("Top 10 Most Popular Songs", color="white")
plt.xlabel("Popularity", color="white")

plt.xticks(color="white")
plt.yticks(color="white")

for spine in ax.spines.values():
    spine.set_color("white")

plt.grid(axis="x", color="white", alpha=0.2)

plt.tight_layout()
plt.savefig("Images/top_popular_songs.png", dpi=300, facecolor="#191414")
plt.show()


# ==========================================
# GRAPH 4 - Popularity Distribution
# ==========================================

plt.figure(figsize=(10,6), facecolor="#191414")
ax = plt.gca()
ax.set_facecolor("#191414")

plt.hist(
    df["popularity"],
    bins=20,
    color="#1DB954",
    edgecolor="white"
)

plt.title("Popularity Distribution", color="white")
plt.xlabel("Popularity", color="white")
plt.ylabel("Number of Songs", color="white")

plt.xticks(color="white")
plt.yticks(color="white")

for spine in ax.spines.values():
    spine.set_color("white")

plt.grid(color="white", alpha=0.2)

plt.tight_layout()
plt.savefig("Images/popularity_distribution.png", dpi=300, facecolor="#191414")
plt.show()


# ==========================================
# GRAPH 5 - Danceability Distribution
# ==========================================

plt.figure(figsize=(10,6), facecolor="#191414")
ax = plt.gca()
ax.set_facecolor("#191414")

plt.hist(
    df["danceability"],
    bins=20,
    color="#1DB954",
    edgecolor="white"
)

plt.title("Danceability Distribution", color="white")
plt.xlabel("Danceability", color="white")
plt.ylabel("Number of Songs", color="white")

plt.xticks(color="white")
plt.yticks(color="white")

for spine in ax.spines.values():
    spine.set_color("white")

plt.grid(color="white", alpha=0.2)

plt.tight_layout()
plt.savefig("Images/danceability_distribution.png", dpi=300, facecolor="#191414")
plt.show()


# ==========================================
# GRAPH 6 - Energy Distribution
# ==========================================

plt.figure(figsize=(10,6), facecolor="#191414")
ax = plt.gca()
ax.set_facecolor("#191414")

plt.hist(
    df["energy"],
    bins=20,
    color="#1DB954",
    edgecolor="white"
)

plt.title("Energy Distribution", color="white")
plt.xlabel("Energy", color="white")
plt.ylabel("Number of Songs", color="white")

plt.xticks(color="white")
plt.yticks(color="white")

for spine in ax.spines.values():
    spine.set_color("white")

plt.grid(color="white", alpha=0.2)

plt.tight_layout()
plt.savefig("Images/energy_distribution.png", dpi=300, facecolor="#191414")
plt.show()


# ==========================================
# GRAPH 7 - Explicit Songs
# ==========================================

plt.figure(figsize=(7,7), facecolor="#191414")
ax = plt.gca()
ax.set_facecolor("#191414")

plt.pie(
    df["explicit"].value_counts(),
    labels=["False", "True"],
    autopct="%1.1f%%",
    colors=["#1DB954", "#535353"],
    textprops={"color":"white"}
)

plt.title("Explicit vs Non-Explicit Songs", color="white")

plt.tight_layout()
plt.savefig("Images/explicit_songs.png", dpi=300, facecolor="#191414")
plt.show()


# ==========================================
# GRAPH 8 - Danceability vs Energy
# ==========================================

plt.figure(figsize=(10,6), facecolor="#191414")
ax = plt.gca()
ax.set_facecolor("#191414")

plt.scatter(
    df["danceability"],
    df["energy"],
    color="#1DB954",
    alpha=0.4
)

plt.title("Danceability vs Energy", color="white")
plt.xlabel("Danceability", color="white")
plt.ylabel("Energy", color="white")

plt.xticks(color="white")
plt.yticks(color="white")

for spine in ax.spines.values():
    spine.set_color("white")

plt.grid(color="white", alpha=0.2)

plt.tight_layout()
plt.savefig("Images/danceability_vs_energy.png", dpi=300, facecolor="#191414")
plt.show()


# ==========================================
# GRAPH 9 - Correlation Heatmap
# ==========================================

plt.figure(figsize=(10,8), facecolor="#191414")

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

plt.title("Correlation Heatmap", color="white")

plt.tight_layout()
plt.savefig("Images/correlation_heatmap.png", dpi=300, facecolor="#191414")
plt.show()


# ==========================================
# GRAPH 10 - Top Albums
# ==========================================

plt.figure(figsize=(12,6), facecolor="#191414")
ax = plt.gca()
ax.set_facecolor("#191414")

albums = df["album_name"].value_counts().head(10).sort_values()

plt.barh(
    albums.index,
    albums.values,
    color="#1DB954",
    edgecolor="white"
)

plt.title("Top 10 Albums", color="white")
plt.xlabel("Number of Songs", color="white")

plt.xticks(color="white")
plt.yticks(color="white")

for spine in ax.spines.values():
    spine.set_color("white")

plt.grid(axis="x", color="white", alpha=0.2)

plt.tight_layout()
plt.savefig("Images/top_albums.png", dpi=300, facecolor="#191414")
plt.show()

print("\nAll graphs generated successfully!")
print("Graphs saved inside Images folder.")