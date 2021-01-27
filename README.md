# 2021_Peak_Fitting2
- Follow-up research Peak_Fitting
- deep learning을 공부하기 시작하면서 병행한 peak fitting 첫 시도는 빈약한 부분이 많았다.
- 학기가 끝난후 개인적으로 더 공부하기 시작하면서 부족한 부분이나 맞지 않는 부분, 잘못 알고 있었던 부분, 그리고 방황하면서 architecture로 방향을 옮기게 됨.

## 데이터
- peak 1~5개를 무작위로 1000만개

## label 선정기준
- area, width, amp, center 기준 중 현재 area를 기준으로 하는 중
- (label을 무엇으로 잡느냐에 따라 보정작업이 달라짐)

## Conv1D layer
- 공부를 하며 이전 Peak_Fitting을 보니까 매우 잘못된 방향으로 가고 있음을 알게 됨.
- 특히 Fully-connected network를 건드는 것이 아니라 CNN을 신경쓰는 것이 매우 중요하다는것을 알게됨 (따라서 이번 후속작업은 CNN architecture로 방향을 잡음)

## Fully-connected network
- Dense층은 layer가 깊지 않아도 학습에 영향을 주지 않았음( layer를 100(node)-1(node)로 총 2개의 layer로 쌓음)

## 느낀점
- label의 기준을 center로 할때가 area로 할때보다 center의 예측값이 훨씬 줄어든다.(label을 그래프의 가장 오른쪽 peak의 
--(c,w,a)parameter로 잡는다)
--(그러나 보정작업도 달라진다)
