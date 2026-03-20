# Day 1 - Convolution

## 참고 자료

- [CS231n Lecture 5](https://www.youtube.com/watch?v=bNb2fEVKeEo&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=5)
- [Pytorch Docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

## 오늘 이해한 핵심

## Convolution

- 입력 신호에 필터(kernel)를 적용해 특정 패턴을 추출하는 연산
- 딥러닝(CNN)에서는 실제로 convolution이 아니라 cross-correlation을 사용

- CNN: kernel을 뒤집지 않고 그대로 sliding

### Discrete Convolution

- CNN 에서 실제 사용
- local region 만 본다 (receptive field)
- 동일한 kernel이 전체 spatial에 반복 적용 -> parameter sharing

## Kernel(filter)

- 특정 패턴을 감지하는 역할
- e.g. Sobel -> edge detection, Gaussian -> smoothing

## Stride

- kernel이 이동하는 간격
- stride 가 클수록 downsampling

## Padding

- input 주변에 0 추가
- valid: no padding
- same: output size = input size
- boundary 정보 보존
- spatial size 유지

## Feature Map

- = kernel 적용한 결과
- 각 위치에서 "해당 패턴이 얼마나 존재하는지" 점수
