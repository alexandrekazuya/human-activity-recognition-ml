# Human Activity Recognition with Machine Learning

University project for human activity recognition using sensor data from the FORTH TRACE dataset.

## Structure
- `data/` – raw and processed dataset files
- `docs/` – project reports
- `results/` – evaluation results and generated plots
- `scripts/` – runnable scripts for preprocessing and analysis
- `src/har/` – core code for classification, evaluation, embeddings, deployment, and SMOTE
- `misc/` – auxiliary and older development files

## Main files
- `scripts/main_activity.py` – main script
- `src/har/classifier.py` – model training
- `src/har/evaluation.py` – model evaluation
- `src/har/embeddings_extractor.py` – embedding extraction
- `src/har/hypothesis_testing.py` – statistical tests
- `src/har/smote.py` – class balancing

## Data
The dataset is stored in `data/FORTH_TRACE_DATASET-master/`.
Processed files can be placed in `data/processed/`.

## Outputs
Generated results are stored in `results/`, including plots and evaluation files.

## Authors
- Alexandre Pereira
- Duarte Silva
