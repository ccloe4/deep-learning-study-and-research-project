# Day 5 - Training Loop

## 참고 자료

- [CS231n Lecture 7](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7)
- [Pytorch Tutorials](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## 오늘 이해한 핵심

## Epoch

- 전체 dataset을 한 번 모두 순회

- 데이터 수: N, batch size: B
- 1 epoch = N / B iteration

## Batch (Mini-batch)

- 한 번에 모델에 넣는 샘플 묶음
- GPU 병렬성 + gradient stability trade-off

## Iteration (Step)

- 한 번의 forward + backward + parameter update
- 즉, batch 1개 처리 = iteration 1

```
for epoch in range(E):
    for batch in dataloader:
        # 1. forward
        # 2. loss 계산
        # 3. backward
        # 4. optimizer step
```
