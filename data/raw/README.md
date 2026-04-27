# Raw data

This folder contains the original (raw) data files used by this project.

## Source

The dataset used here was obtained from the following public repository:

[Mendeley Data — btm76zndnt (version 2)](https://data.mendeley.com/datasets/btm76zndnt/2)

Please visit the Mendeley Data page above for the original dataset description, authors, DOI (if present), license details, and any additional usage restrictions.

## Description

The files in this directory were copied from the dataset hosted on Mendeley Data. They contain the original exported records and example utterances used for the help desk / issue tracking dataset referenced in the project.

## Files included

The repository's `data/raw/` directory contains (or may contain) CSV files such as:

- `issues.csv` — primary issue records export
- `issues_snapshot.csv` — snapshot history export
- `issues_change_history.csv` — change history export
- `sample_utterances.csv` — example user utterances / text samples

Note: filenames and exact contents may vary depending on which files were downloaded from the Mendeley dataset. Always refer to the original Mendeley page for the canonical file list.

## Citation and license

If you use this dataset in research or a project, please cite the original dataset as provided on the Mendeley Data page. The canonical citation (authors, year, title, DOI) is available on the dataset page linked above.

If no license is stated on the dataset page, contact the dataset authors or assume default academic attribution requirements before redistribution. Do not redistribute the original raw files without checking the licensing terms on Mendeley Data.

## Usage notes

- These raw files are treated as immutable source-of-truth for the project. Processed/cleaned data is stored under `data/processed/` after running the data preparation scripts.
- If you regenerate the processed datasets, place the original downloaded files here (under `data/raw/`) and do not commit any sensitive or private data to the repository.
- Keep the original Mendeley Data record for provenance: note the URL and any DOI in project documentation and in publications.

## Questions or corrections

If you believe the data in this folder is incomplete or incorrect relative to the Mendeley Data source, please re-download the dataset from the link above and update this README with the exact filenames and the dataset DOI / citation string.
