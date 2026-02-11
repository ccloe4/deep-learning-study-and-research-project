# Day 2 - Autograd

## 참고 자료
- [PyTorch autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial)
- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd)
- [CS231n Lecture 4](https://www.youtube.com/watch?v=d14TUNcbn1k&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=4)
- [PyTorch Explained: Computation Graph and Backpropagation, torch.autograd, backward, grad](https://www.youtube.com/watch?v=SvyGrdzQ9KI)

## 오늘 이해한 핵심

- requires_grad=True → gradient 추적 시작
- 연산할 때마다 computational graph가 동적으로 생성됨
- directed acyclic graph(DAG)
- backward()는 chain rule 기반으로 graph를 역순회
- gradient는 leaf tensor의 .grad에 저장됨

## 중요
### loss를 scalar로 정의하는 이유?
```
1. 우리가 최소화하려는 목적함수는 하나의 실수값이어야 함
2. gradient는 scalar → vector 미분이어야 SGD 업데이트가 가능
3. 벡터 loss는 Jacobian이 되어 방향 정의가 모호함
4. 결국 항상 scalar로 reduce 해서 사용함
```