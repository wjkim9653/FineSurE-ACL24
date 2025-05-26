#!/bin/bash

# To Run -> ./run_all.sh

# Exit immediately if a command exits with a non-zero status
set -e

echo "Running Fact-Checking Evaluation..."
python finesure/fact-checking.py \
    dataset/frank/frank-data.json \
    result/fact-checking \
    200

echo "Running KeyFact Alignment (Human KeyFacts)..."
python finesure/keyfact-alignment.py \
    dataset/realsumm/realsumm-data.json \
    dataset/realsumm/keyfact/human-keyfact-list.json \
    result/keyfact-alignment \
    200

echo "Running KeyFact Alignment (Machine-Extracted KeyFacts)..."
python finesure/keyfact-alignment.py \
    dataset/realsumm/realsumm-data.json \
    dataset/realsumm/keyfact/ \
    result/keyfact-alignment \
    200

echo "Reproducing Main Results..."
python reproduce/reproduce-main-results.py \
    dataset/realsumm/keyfact/ \
    result/

echo "✅ All evaluations completed."