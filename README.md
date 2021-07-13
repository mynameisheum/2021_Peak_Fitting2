# 2021_Peak_Fitting2

## “Iterative peak-fitting of frequency-domain data via deep convolution neural networks”
### - Current Applied Physics(SCI)(submitted)
- (https://arxiv.org/abs/2107.04287)


## 2021_Peak_Fitting2
- Follow-up research Peak_Fitting
- deep learning을 공부하기 시작하면서 병행한 peak fitting 첫 시도는 빈약한 부분이 많았다.
- 학기가 끝난후 개인적으로 더 공부하기 시작하면서 부족한 부분이나 맞지 않는 부분, 잘못 알고 있었던 부분, 그리고 방황하면서 architecture로 방향을 옮기게 됨.
- 확실히 과거peak_Fitting(lenet)보다 현재 peak_fitting2 model(SE-Dense-Resnet)(21/3/2)에서 보다 오차를 많이 감소 시킬수 있었음
- ![SE-Dense-Resnet](https://github.com/mynameisheum/2021_Peak_Fitting2/blob/main/picture_storage/SE-Dense-Resnet.png?raw=true)

## 데이터
- peak 1~5개를 무작위로 1000만개

## label 선정기준
- area, center, width, amplitude 기준 중 현재 가장 큰 area의 peak의 parameter(center,width,amplitude)를 기준으로 하는 중
- (label을 무엇으로 잡느냐에 따라 보정작업이 달라짐)

## Conv1D layer
- 공부를 하며 이전 Peak_Fitting을 보니까 매우 잘못된 방향으로 가고 있음을 알게 됨.
- 특히 Fully-connected network를 건드는 것이 아니라 CNN을 신경쓰는 것이 매우 중요하다는것을 알게됨 (따라서 이번 후속작업은 CNN architecture로 방향을 잡음)

## Fully-connected network
- Dense층은 layer가 깊지 않아도 학습에 영향을 주지 않았음( layer를 100(node)-1(node)로 총 2개의 layer로 쌓음)

## 느낀점
- label의 기준을 center로 할때가 area로 할때보다 center의 예측값이 훨씬 줄어든다.(label을 그래프의 가장 오른쪽 peak의 parameter(c,w,a)로 잡는다)(그러나 보정작업도 달라진다)
- architecture을 어떤식으로 모델링을 하느냐에 따라 정확도(mae)가 많이 개선된다.
- 데이터 양은 충분

## 문제점
- 실제 데이터인 p3ht의 peak2는 잘 맞추는 편이지만, ito의 peak3가 마지막 고비다. 아직 잘 안맞춰진다
- 따라서 label의 기준을 현재 area와 (>) center를 고려하는중이다
- 보정을 통해 픽을 하나씩 빼는 과정에서 남은 오차가 두번째, 세번째에 빼질수록 나머지의 영향력이 커져 갈수록 오염된 정보를 받아 다른 예측을 한다.
- 불행인지 다행인지 잘못 빼진 나머지 픽들을 또 비슷하게 맞추긴 함. 따라서 architecture로써 올릴수 있는 정확도는 얼추 마무리 된거 같다. 이제 세부적인 Hyper-parameter를 신경을 써야할 것 같다. -> efficientnet 의 insight를 사용할수 있을까?
