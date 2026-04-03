# Data Processing for LLM Competition

## Overview
This repository contains scripts used to improve model performance through data design.

## Data Processing

This script was used to improve model performance through data design.

Key ideas:

- Normalize different datasets into consistent formats  
  (ALFWorld: action-only, DBBench: SQL-only)

- Identify weak task types (e.g., aggregation, counting)  
  and upweight them by duplication

- Combine and shuffle datasets to control distribution

These steps were based on the hypothesis that data quality and distribution
significantly affect model performance.

Output dataset is available on Hugging Face:
https://huggingface.co/datasets/kochan13/mixed-agent-dataset-merged-clean-dedup-dbweak_2x_2

### Pipeline

The processing pipeline follows this order:

1. Normalize datasets into consistent formats  
2. Clean and filter invalid samples  
3. Remove duplicates using hash-based keys  
4. Upweight weak task types AFTER deduplication  
5. Shuffle and construct final dataset  

This ordering is critical to ensure that distribution control is preserved.

## Code
See `data_processing.py`
