#!/bin/bash

# 로그 파일 경로
TOKENIZE_LOG="../logs/tokenize.log"
TRAIN_LOG="../logs/train.log"
PLOT_LOG="../logs/plot_loss.png"

# # 1. 토큰화
# echo "==== Starting tokenization ===="
# python tokenize_corpus.py > $TOKENIZE_LOG 2>&1
# if [ $? -ne 0 ]; then
#     echo "Tokenization failed. Check $TOKENIZE_LOG"
#     exit 1
# fi
# echo "==== Tokenization completed. Log: $TOKENIZE_LOG ===="

# 2. 학습
echo "==== Starting training ===="
python train.py >> $TRAIN_LOG 2>&1
if [ $? -ne 0 ]; then
    echo "Training failed. Check $TRAIN_LOG"
    exit 1
fi
echo "==== Training completed. Log: $TRAIN_LOG ===="

# 3. loss curve 시각화
echo "==== Generating loss plot ===="
python plot_train_log.py $TRAIN_LOG $PLOT_LOG
echo "==== Loss plot saved to $PLOT_LOG ===="