# Day 3 - MLP

## 참고 자료

- [Dive into Deep Learning 5. MLP](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html)
- [Deep Learning Basics](https://rtavenar.github.io/deep_book/en/content/en/mlp.html)
- [Pytorch tutorials](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html)

## 오늘 이해한 핵심

- 모든 층에 비선형 함수를 추가해서 단일층으로 단순화할 수 없게 함
- 선형, 비선형 항목을 서로 교차시키면서 레이어 깊이가 깊어짐
- 이때 필요한 것이 activation function ( e.g. ReLU, Sigmoid )

## 중요

### MLP란?
>>
>> MLP은 입력과 출력층에 한 개 이상의 완전 연결 은닉층(hidden layer)를 추가하고, 각 은닉층(hidden layer)의 결과에 활성화 함수(activation function)를 적용함
