# scripts/explore_data.py
import json
import pandas as pd
from collections import Counter

print("=" * 60)
print("EXPLORING ARXIV DATASET")
print("=" * 60)

# Load first 1000 papers (to start small)
papers = []
with open('data/raw/arxiv-metadata-oai-snapshot.json', 'r') as f:
    for i, line in enumerate(f):
        if i >= 1000:  # Only load 1000 for now
            break
        papers.append(json.loads(line))
        
        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1} papers...")

print(f"\n Loaded {len(papers)} papers successfully!")

# Convert to DataFrame
df = pd.DataFrame(papers)

print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Number of papers: {len(df)}")
print(f"\nColumns available:")
for col in df.columns:
    print(f"  - {col}")

# Show a sample paper
print("\n" + "=" * 60)
print("SAMPLE PAPER")
print("=" * 60)
sample = papers[0]
print(f"ID: {sample['id']}")
print(f"Title: {sample['title']}")
print(f"Authors: {sample['authors']}")
print(f"Categories: {sample['categories']}")
print(f"Abstract: {sample['abstract'][:300]}...")

# Category distribution
print("\n" + "=" * 60)
print("TOP 10 CATEGORIES")
print("=" * 60)
all_categories = []
for cats in df['categories']:
    all_categories.extend(cats.split())

category_counts = Counter(all_categories).most_common(10)
for cat, count in category_counts:
    print(f"{cat:15} : {count:4} papers")

# Year distribution
print("\n" + "=" * 60)
print("PAPERS BY YEAR (Sample)")
print("=" * 60)
df['year'] = pd.to_datetime(df['update_date']).dt.year
year_counts = df['year'].value_counts().sort_index().tail(10)
for year, count in year_counts.items():
    print(f"{year}: {count} papers")

# Abstract statistics
print("\n" + "=" * 60)
print("ABSTRACT STATISTICS")
print("=" * 60)
df['abstract_length'] = df['abstract'].str.len()
print(f"Average length: {df['abstract_length'].mean():.0f} characters")
print(f"Shortest: {df['abstract_length'].min():.0f} characters")
print(f"Longest: {df['abstract_length'].max():.0f} characters")

print("\n" + "=" * 60)
print(" EXPLORATION COMPLETE!")
print("=" * 60)
print("\nNow we understand our data structure.")
print("Next step: We'll process the full dataset!\n")