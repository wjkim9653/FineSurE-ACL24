#!/bin/bash

# To Run -> ./run_all.sh

# Variable: Number of Samples to Process
SAMPLE_COUNT=50

# Exit immediately if a command exits with a non-zero status
set -e

# Frank 데이터셋 기반으로 Fact-Checking 진행
echo "Running Fact-Checking Evaluation..."
python finesure/fact-checking.py \
    dataset/frank/frank-data.json \
    result/fact-checking \
    $SAMPLE_COUNT

# Realsumm 데이터셋 기반으로 Summary<->KeyFact Alignment 진행
echo "Running KeyFact Alignment (Human KeyFacts)..."
python finesure/keyfact-alignment.py \
    dataset/realsumm/realsumm-data.json \
    dataset/realsumm/keyfact/human-keyfact-list.json \
    result/keyfact-alignment \
    $SAMPLE_COUNT

# LLM-Judge가 동적으로 KeyFact List 추출까지 진행
    # ⚠️ -> KeyFact 추출, LLM-as-a-Judge의 평가까지는 정상동작하나, 인간작성 GT Annotation은 human-keyfact-list.json 기반이므로 
    # ⚠️ Pearson/Spearman/Rank Correlation 계산에서 비정상적 수치 계산되므로 신뢰하기 어려움
echo "Running KeyFact Alignment (Machine-Extracted KeyFacts)..."
python finesure/keyfact-alignment.py \
    dataset/realsumm/realsumm-data.json \
    dataset/realsumm/keyfact/ \
    result/keyfact-alignment \
    $SAMPLE_COUNT


# 논문에 있던 테이블 수치 재구현 (Human Annotator Ground-Truth vs LLM-as-a-Judge Predictions)
# FineSurE 프레임워크 진행 후 결과의 Faithfulness/Completeness/Conciseness 간 Correlation 계산
echo "Reproducing Main Results..."
python reproduce/reproduce-main-results.py \
    dataset/realsumm/keyfact/ \
    result/

echo "✅ All evaluations completed."