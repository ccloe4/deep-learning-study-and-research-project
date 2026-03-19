# Day 4 - Optimizer & Regularization

## 참고 자료

- [CS231n Lecture 6](https://www.youtube.com/watch?v=wEoyxE0GP2M&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=6)

## 오늘 이해한 핵심

### Optimizer

SGD가 mini-batch gradient를 써도 되는 이유:

```
1. 기대값이 진짜 gradient와 같기 때문 (unbiased)
2. 계산량이 훨씬 적음
3. 노이즈가 오히려 최적화에 도움됨
```

Adam

- RMSProp + Momentum
- 대부분의 모델 학습에 적합함

Second-Order Optimization

- Hessian takes O(N^2) -> 자원을 많이 잡아먹어서 딥러닝에는 적합하지 않음

### Reqularization

Dropout

- In each forward pass, randomly set neurons to zero
- 드랍하는 확률은 hyperparameter, 0.5 가 흔히 사용됨

왜 좋은가?

- Prevents co-adaptation of features
- parameter 를 공유하는 여러 모델의 ensmeble 을 학습하는 것

Batch Normalization

- commonly used
- gradient 가 덜 폭발함
- 더 큰 learning rate 사용이 가능해서 학습 속도가 빨라짐
- 초기값에 덜 민감
- 약간 노이즈 역할을 해서 overfitting 감소

Data Augmentation

- random transformations to the input data

### Transfer Learning

- pretrained model + fine tuning 이용하면 학습 시간을 크게 단축시킬 수 있음
- 공개된 pretrained model 활용해서 프로젝트 진행하기 수월함