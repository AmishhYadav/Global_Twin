---
wave: 1
depends_on: []
files_modified: ["requirements.txt", "src/data/ingest.py", "scripts/test_ingest.py"]
autonomous: true
---

# Plan 1: Data Ingestion Pipeline

## Objective
Establish parsing and normalization of historical CSV datasets, addressing requirements DATA-01, DATA-02.

## Tasks

<task>
<action>
Create `requirements.txt` containing `pandas==2.2.1`.
</action>
<read_first>
None
</read_first>
<acceptance_criteria>
`cat requirements.txt | grep pandas` returns a match.
</acceptance_criteria>
</task>

<task>
<action>
Create directory `src/data/` and create `src/data/ingest.py`. Implement `load_and_clean_data(filepath)` which uses `pandas.read_csv`, attempts date parsing, normalizes the index to daily frequency, uses forward-fill (`ffill()`) for missing values, and prints warnings to stdout if gaps exceed a threshold.
</action>
<read_first>
`src/data/ingest.py` (when created)
</read_first>
<acceptance_criteria>
`cat src/data/ingest.py | grep "def load_and_clean_data"` returns a match.
`cat src/data/ingest.py | grep ffill` returns a match.
</acceptance_criteria>
</task>

<task>
<action>
Create directory `scripts/` and create `scripts/test_ingest.py` that generates a small mock CSV file with missing data, then runs `load_and_clean_data()` on it and `print()`s the resulting DataFrame shape and first few rows to confirm ingestion works properly.
</action>
<read_first>
`src/data/ingest.py`, `scripts/test_ingest.py` (when created)
</read_first>
<acceptance_criteria>
Running `python scripts/test_ingest.py` exits with 0 and prints a dataframe shape.
</acceptance_criteria>
</task>

## Verification
- User can use a local script to load data (tested via `test_ingest.py`).
- Dates normalize correctly and missing data forwards-fills (validated by script output).
- Output is a unified Pandas DataFrame (validated by script output).

## Must Haves
- Python script available to ingest datasets.
- Forward fill handles missing values.
- Datetime indices mapped properly.
