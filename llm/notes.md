# Project - Building a Korean-LLM

## 참고 자료

- [LLM 바닥부터 만들기](https://www.youtube.com/watch?v=osv2csoHVAo&list=WL&index=4&t=679s)
- [홍정모 연구소](https://github.com/HongLabInc/HongLabLLM)

## LLM

1. 사전훈련(pretraining)으로 일반적인 언어 능력을 가르친 후에
2. 미세조정(fine tuning) 단계에서 특정 업무에 적응 시킴

3. 데이터베이스(+ 인터넷) 검색 기능을 추가하면 지식의 범위와 정확성을 높일 수 있음
4. 내부적으로 질의를 반복하여 더 좋은 결론을 도출

### 기본 과정

1. 훈련 데이터 준비

- Wikipedia(kor)
- 나무위키

2. 데이터 로더 정의

3. 모델 정의 - self attention 을 구현하는 transformer 구조

4. 훈련

5. 결과 확인
