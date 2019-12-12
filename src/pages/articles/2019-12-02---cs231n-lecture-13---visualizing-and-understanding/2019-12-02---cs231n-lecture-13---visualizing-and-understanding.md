---
draft:     false
title:     "cs231n lecture 13 - Visualizing and Understanding"
date:      "2019-12-02 16:08:25"
layout:    post
path:      "/posts/cs231n-lecture-13---visualizing-and-understanding/"
category:  "Machine Learning"
tags: 
    - cs231n
    - Deep Learning
    - Visualization
description: "2019년 스탠포트 cs231n lecture 13 인 13강 강의를 공부하고 정리한 포스트 입니다. 시각화와 해석에 관한 내용을 다룹니다."
---

> 이 포스트는 스탠포드의 [cs231n](http://cs231n.stanford.edu) 13강 강의를 보고 공부 및 정리한 포스트입니다.  
> 잘못된 것이 있을 수 있습니다.  
> 댓글로 알려주시면 감사합니다!  

<small>최종 수정일: 2019-12-09</small>

이번 강의에서는 다양한 예제를 보게 될 것이다.  
_ConvNet_ 을 시각화해보고 이를 통해 어떤 해석을 해 볼 수 있는지 생각해보자.

## Inside ConvNet

우리는 지금까지 _ConvNet_ 을 이용해 classification, object detection, segmentation 등을 수행해보았다.  

이러한 _ConvNet_ 은 어떻게 동작하는 것일까?  
특히, 그 내부에서는 무슨 일이 벌어지고 있는 것일까?  

![what's going on inside convnet](./image1.png)

이제부터 그 내부에서 어떤 일이 일어나는지 살펴보자.  

![first layer visualize filters](./image2.png)

다양한 색상과 선 모양의 패턴들을 확인할 수 있다.

사람도 시각피질의 원초적인 부분(단계)에서 이러한 것을 본다고 알려져있다.  
즉, 사람과 네트워크 모두 초기 단계에 물체를 인식하는데 비슷한 양상을 보인다는 의미이다.  

흥미로운 점은, 어떤 모델을 이용해 학습시키더라도 초기 레이어에서 위와 같이 나타난다는 것이다.

![layer visualize](./image3.png)

그렇다면 레이어를 조금 더 내려가면 어떻게 될까?

한 눈에 보기에도 이해하기 어려워 보이는 결과를 볼 수 있다.  
즉, 초기 레이어와 같은 방식의 시각화 방식으로는 해석이 어렵다는 의미이다.  

두 번째 레이어에서 찾고 있는 것은 무엇일까?  
우리가 보고 있는 것은 _두 번째 레이어의 결과를 최대화시키는 첫 번째 레이어의 출력 패턴이 무엇인지_ 이다.  

이렇게 이미지의 관점에서 레이어를 해석하기란 쉽지 않다.  
이러한 중간 레이어는 조금 다른 기법의 해석 방식이 필요하다.  

이러한 이미지들은 가중치를 0~255 의 값으로 normalize 한 값이다.  
사실 가중치에는 한계가 없기 때문에 시각화를 할 때 필터만 진행하게 된 것이고 bias 는 고려되지 않은 상태가 된다.  
따라서 이러한 결과를 그대로 받아들이면 안된다.  

![last layer nn](./image4.png)

이번에는 _CNN_ 의 마지막 레이어를 생각해보자.  

AlexNet 에서는 4096-dim vector 가 마지막 레이어로부터 출력된다.  
많은 이미지를 돌려서 해당 vector 를 모두 저장한다.  
이를 활용해서 마지막 레이어를 시각화하는 방법은 무엇이 있을까?  

여기서 해 볼 방법은 Nearest Neighbour 이다.  
CIFAR-10 데이터들의 이미지 픽셀 공간에서 NN 을 시도한 결과 위의 이미지 처럼 가장 왼쪽 이미지를 기준으로 비슷한 이미지들을 잘 찾아낸다는 것을 볼 수 있다.  

그런데 이러한 픽셀 공간에서의 NN 은 비슷한 이미지를 나타내게 할 텐데, 2번째 사진(왼쪽 모음 중)의 흰 강아지 사진과 어떤 흰 덩어리 사진은 그 거리가 가깝다고 나타내게 될 것이다.  

그러나 특징 벡터 공간에서의 NN 은 어떨까?  
오른쪽 모음의 두 번째 사진인 코끼리 사진을 보자.  

코끼리의 모습이 반대(좌-우)임에도 잘 찾아내는 것을 볼 수 있다.  
이는 픽셀 공간에서 보면 이미지의 거리가 멀다고도 볼 수 있을텐데 특징 벡터 공간에서는 그그 거리가 가깝다고 하는 것이다.  

이는 네트워크가 학습을 통해 이미지의 semantic content 한 특징을 잘 찾아낸 것이라 볼 수 있다.  

![last layer dimensionality reduction](./image5.png)

또 다른 방법으로는 최종 레이어에서 차원 축소(Dimensionality Reduction)의 개념으로 접근해 볼 수도 있다.  

PCA 는 4096-dim 과 같은 고차원 특징벡터들을 2-dim 으로 압축시키는 기법이다.

이 방법을 통해서 특징 공간을 조금 더 직접적으로 시각화시킬 수 있다.

조금 더 복잡한 t-SNE(t-distributed stochastic neighbor
embeddings)라는 알고리즘을 활용하면 더 잘 시각화 할 수도 있다.

MNIST 예시를 보자. t-SNE dimensionality reduction 를 통해 시각화한 모습이다.  

MNIST의 각 이미지는 Gray scale 28x28 이미지이다.

여기에서는 t-SNE가 MNST의 28x28-dim 데이터를 입력으로 받게 된다(raw pixels). 그리고 2-dim으로 압축해 시각화한다.

그 결과 군집화된 모습을 볼 수 있으며 이는 MNIST의 각 숫자를 의미한다.

![last layer dimensionality reduction](./image6.png)

이런 식의 시각화 기법을 ImageNet 을 학습시킨 네트워크의 마지막 레이어에도 적용해보자.

우선 엄청나게 많은 이미지들을 네트워크에 통과시킨다.

그리고 각 이미지에 대해서 최종 단의 4096-dim 특징 벡터들을 기록한다.

그러면 4096-dim 특징 벡터들을 아주 많이 모을 수 있을 것이고 t-SNE을 적용하면
4096-dim에서 2-dim으로 압축된다.

이를 통해 2-dim 특징 공간의 각 grid에 압축된 2-dim 특징들이 시각화시킨다.

이를 통해 학습된 특징 공간의 모습을 어렴풋이 추측해 볼 수 있게 된다.

<small>온라인에서 고해상도 이미지로 확인해보자.</small>

좌하단의 초록색 군집은 다양한 종류의 꽃들이 있는 곳이고, 다른 곳에는 이제 강아지, 동물, 지역 등등이 있다.  

이를 통해 우리가 학습시킨 특징 공간에는 일종의 불연속적인 의미론적 개념(semantic notion)이 존재하며 t-SNE을 통한 dimensionality reduction version의 특징 공간을 살펴보며 그 공간을 조금이나마 이해해 볼 수 있었다.  

> 지금까지 FC-layer 에 대해 시각화를 진행했는데, 더 상위 레이어에서도 이러한 과정을 진행해 볼 수는 있다.

![visualizing activations](./image7.png)

중간 레이어의 가중치를 시각화 하는 것은 해석이 어렵다.  
그러나 가중치가 아닌 activation map 을 시각화 하는 방법도 있다.  

위의 이미지에서 초록색 부분의 경우 사람의 얼굴 모양을 보고 활성화되는 것 같아 보인다.  
즉, 네트워크의 어떤 레이어에서는 사람의 얼굴을 찾고 있는 것일 수도 있다.

### Maximally Activating Patches

![maximally ctivating patches](./image53.png)

특정 레이어의 활성을 최대로 하는 패치들을 시각화 하는 방법도 있다.  
각 행에 있는 패치들은 모두 하나의 뉴런에서 나온 것이다.  

여기서 해당 레이어가 convolutional layer 이기에 이미지 전체가 아닌 일부분만 보고 있는 것임을 상기하자. 즉, receptive field 가 작다는 의미다.  

오른쪽 예시의 모양을 보면 동그라미(아마도 눈) 모양을 찾고 있는 것 같아 보인다.  

여기서 한 뉴런은 conv5 activation map 의 한 scalar 값을 의미한다.  
이 때, convolutional layer 이기에 한 채널의 모든 뉴런은 모든 같은 가중치를 공유하게 된다.  

오른쪽 아래 이미지의 그룹은 더 깊은 layer 로부터 추출된 패치들인데, 이는 receptive field 가 넓다는 의미다. 즉, 이미지를 더 넓게 보고 있는 것이다.  
이미지를 잘 보면 사람의 모습(얼굴, 2번째 행) 등을 찾고 있음을 알 수 있다.  

## Saliency Map

![saliency via occlusion](./image8.png)

여기 재미있는 실험이 있다.  

'Occlusion Experiment' 라 하여, 어느 부분이 분류를 결정짓는 근거가 되는지 알아보는 실험이다.  

이미지를 가리고 이미지의 평균데이터를 넣었을 때 어느 부분이 가려졌을 경우 네트워크의 예측확률이 크게 변하는지를 체크해본 결과 오른쪽의 Map 을 통해 나타나는 것이다.  

> 이러한 과정은 학습하는데 어떤 도움을 주거나 하는 것은 아니다.  
> 그러나 이는 인간이 네트워크의 학습 과정을 _이해_ 하는 도구로써 사용될 수는 있다.  

이렇게 이미지를 가리는 것에서 더 나아가 _Saliency Map_ 이라는 것을 알아보자.  

이 방법은 입력 이미지의 각 픽셀들에 대해 예측한 클래스 스코어의 gradient 를 계산하는 방법이다.  
이는 1차 근사적 방법으로 어떤 픽셀이 영향력 있는지 알려준다.  

![which pixels matter saliency via backprop](./image9.png)

이를 통해 map 을 만들어본다면 오른쪽 처럼 개의 윤곽이 드러남을 볼 수 있다.  

![saliency maps](./image10.png)

다른 이미지들에 대해서도 잘 되는 것을 확인할 수 있다.  

### Grabcut

![segmentation without supervision](./image11.png)

여기서 _GrabCut_ 이라는 것을 활용하면 segmentation label 없이 segmentation 을 수행할 수 있게 된다.  

_GrabCut_ 은 interactive segmentation algorithm 으로 supervision 으로 만들어지는 결과에 비해 그렇게 좋지는 못하다.

### Intermediate Features via (guided) Backprop

![intermediate features via guided backprop](./image12.png)

또 다른 방법으로는 _guided backpropagation_ 이 있다.

이는 클래스 스코어가 아니라 네트워크의 중간 뉴런을 하나 고른다.
그리고 입력 이미지의 어떤 부분이, 내가 선택한 중간 뉴런의 값에 영향을 주는지를 찾는다.

이 때도 _Saliency Map_ 을 만들어볼 수 있을 것이다.

이 경우에는 이미지의 각 픽셀에 대한 클래스 스코어의 gradient 를 계산하는 것이 아니라 입력 이미지의 각 픽셀에 대한 네트워크 중간 뉴런의 gradient 를 계산하게 된다.

이를 통해 어떤 픽셀이 해당 뉴런에 영향을 주는 지 알 수 있다.

이 방법은 backprop 시 _ReLU_ 를 통과할 때 조금의 변형한다.

_ReLU_ 의 gradient 의 부호가 양수 이면 그대로 통과시키고 부호가 음수이면 backprop하지 않는다.

이로 인해 전체 네트워크가 실제 gradient 를 이용하는 것이 아니라 _양의 부호인 gradient_ 만을 고려하게 된다.

![intermediate features via guided backprop](./image13.png)

방금 보았던 Maximally activating patches 를 상기해보자.  
우리는 첫 행을 보고 동그란 무언가(아마도 눈)를 찾고 있는 것이라고 짐작했는데, _guided backprop_ 의 결과를 보니 우리의 짐작이 어느 정도 맞다고 생각할 수 있게 되었다.  

![intermediate features via guided backprop](./image14.png)

그러나 이 방법은 고정된 입력 이미지 또는 입력 패치의 어떤 부분이 해당 뉴런에 영향을 끼치는가를 말해줄 뿐이다.  

그렇다면 입력 이미지에 고정되지 않은 방법은 없을까? <small>입력 이미지에 의존적이지 않고 해당 뉴런을 활성시킬 수 있는 _일반적인_ 이미지가 있을까?</small>  

## Gradient Ascent

![gradient ascent](./image15.png)

_Gradient Ascent_ 에 대해 살펴보자.

여기서는 네트워크의 가중치들을 전부 고정시킨다.

그리고 Gradient ascent를 통해 중간 뉴런 혹은 클래스 스코어를 최대화 시키는 이미지의 픽셀들을 만들어낸다.

이는 Gradient ascent는 네트워크의 가중치를 최적화하는 방법이 아니다. 가중치들은 모두 고정되어 있고 대신 클래스 스코어가 최대화될 수 있도록 입력 이미지의 픽셀 값을 바꿔주는 방법이다.  

regularization term을 추가함으로서, 우리는 생성된 이미지가 두 가지 특성을 따르길 원하는 것이다.  

하나는 이미지가 특정 뉴런의 값을 최대화시키는 방향으로 생성되길 원하는 것이고 그리고 다른 하나는 이미지가 자연스러워 보여야 한다는 것이다.  

생성된 이미지가 자연 영상에서 일반적으로 볼 수 있는 이미지이길 원하는 것이다.

이런 류의 regularization term의 목적은 생성된 이미지가 비교적 자연스럽도록 강제하는 역할을 한다.

![gradient ascent](./image16.png)

Gradient Ascent를 위해서는 초기 이미지가 필요하다.  
이 이미지는 zeros, uniform, noise 등으로 초기화시킨다.  

초기화를 하고나면 이미지를 네트워크에 통과시키고 스코어나 관심있는 뉴런의 값을 계산하게 된다.

그리고 이미지의 각 픽셀에 대한 해당 뉴런 스코어의 gradient 를 계산하기 위해 backprop 을 수행한다.

여기에서는 Gradient Ascent 를 이용해서 이미지 픽셀 자체를 업데이트한다.  
해당 스코어를 최대화시키게 되고, 이 과정을 계속 반복하고나면 아주 멋진 이미지가 탄생하게 된다.

![gradient ascent](./image17.png)

덤벨과 컵의 예제를 보면 많이 중첩된 모습을 볼 수 있다.  

달마시안의 경우에는 달마시안의 특징이 아주 잘 나타난 것을 확인할 수 있다.

여기서 색상이 무지개 색인 이유는 _Gradient Ascent_ 가 unconstrained value 이기 때문에 0~255 의 pixel value 로 normalize 하면서 나타나는 오류라고 볼 수 있다.  

> regularization term 이 없이 할 경우에도 이미지는 나타날 것이나 이는 random noise 처럼 보이게 된다.  
> 그러나 이는 또 다른 의미를 갖는데 곧 확인해보자.

![gradient ascent](./image18.png)

L2 norm 에다가 주기적으로 Gaussian blur 를 적용하는 방법도 있다.  
그리고 값이 작은 픽셀들은 0 으로 만들고, gradient 가 작은 값도 0 으로 만든다.  
<small>(일종의 projected Gradient descenct)</small>

이렇게 만든 방법은 훨씬 더 깔끔한 이미지를 위와 같이 얻게 만들어준다.  

![gradient ascent](./image19.png)

이러한 방법을 최종 스코어에만 적용하는 것이 아니라 중간 뉴런을 최대화시키는 이미지를 생성해 볼 수도 있다.  

여기서 이미지가 큰 것은 receptive field 가 크다는 것을 의미한다.  

![gradient ascent](./image20.png)

이는 Multimodality 도 최적화 과정 속에서 다루고 있다.  
위의 이미지들은 식료품점(grocery store)으로 labeling 되어 있다.  

여기서 각 class 마다 clustering algorithm 을 수행한다.  
한 class 내에서  서로 다른 mode 들끼리 다시 class 가 나뉜다.  
그리고 나뉜 mode 들과 가까운 곳으로 초기화를 해주게 된다.  

위의 식료품점 이미지는 매우 다른 것을 볼 수 있다.  

많은 클래스들이 이렇게 multimodality를 가지고 있는데, 이미지를 생성할 때 이런 식으로  multimodality를 명시하게 되면 아주 좋은(다양한) 결과를 얻을 수 있게 된다.

![gradient ascent](./image21.png)

위의 이미지들은 ImageNet 의 특정 클래스를 최대화하는 이미지를 생성해 낸 결과물이다.

이를 가능하게 하는 방법은 FC6 를 최적화하는 것이다.

이는 feature inversion network 등을 사용해야한다.  
<small>논문을 읽어보자... 요점은 이미지 생성 문제에서 사전 지식(priors)를 추가하게 된다면 리얼한 이미지를 만들 수 있다는 점이다.</small>

## Fooling Image / Adversarial Examples

![fooling image](./image22.png)

이러한 기법을 응용하게 되면 네트워크를 속이는 이미지를 만들어 낼 수 있다.  

위의 코끼리 사진이 거의 차이가 없어보이지만 코알라로 인식되는 것이 보이는가?  

이는 추후에 조금 더 자세히 다루도록 하겠다.  

## DeepDream

이제 재미난 것들을 보자.  

_DeepDream_ 은 구글에서 발표한 것으로, 재미있는 이미지를 만들기 위해 고안되었다.  

사람이 꿈을 꾸면서 환상(hallucination)같은 것을 보는 느낌과 비슷하도록 이미지를 만들기도 한다고 해서 이와 같은 이름이 붙여졌다.  

추가적으로 모델이 이미지의 어떤 특징들을 찾고 있는지도 살짝 확인이 가능하다.  

![deepdream](./image23.png)

_DeepDeram_ 에서는 입력 이미지를 _CNN_ 의 중간 레이어를 어느정도 통과시키게 된다.

그리고 backprop 을 하게 되는데 해당 레이어의 gradient 를 activation 값으로  설정한다.

그리고 backprop 을 하여 이미지를 업데이트하는 과정을 반복한다.

이러한 행동은 네트워크에 의해 검출된 해당 이미지의 특징들을 증폭시키려는 것으로 해석할 수 있다.

해당 레이어에 어떤 특징들이 있던지 그 특징들을 gradient 로 설정하면 이는 네트워크가 이미지에서 이미 뽑아낸 특징들을 더욱 증폭시키는 역할을 하게 된다.

그리고 이는 해당 레이어에서 나온 특징들의 L2 norm을 최대화시키는 것으로 볼 수 있다.

![deepdream](./image24.png)

코드는 굉장히 간결하다.  
몇 가지 트릭이 있는데, 

1. gradient 를 계산하기에 앞서 이미지를 조금 움직임(jitter)  
    - regularization 의 역할로 이미지를 부드럽게 만듦
2. L1 normalization 사용  
    - 이미지 합성에서 유용함
3. clipping  
    - pixel 이 이미지로 표현되기 위해서 값은 0~255 사이어야 함.

![deepdream](./image25.png)

위와 같은 방법으로 하늘 이미지에

![deepdream](./image26.png)

이런식으로 무늬를 넣어보거나(얕은 층의 레이어)

![deepdream](./image27.png)

뭔가 요란한 이미지를 만들어 낼 수도 있다.(깊은 층의 레이어)

![deepdream](./image28.png)

이러한 이미지들이 많이 발견되는 것을 볼 수 있다.  
강아지가 좀 많은 데이터셋인 것 같다.  

<small>실제로 ImageNet(위의 예제)에는 1000개의 카테고리가 존재하는데 그 중 200개가 __개__ 이다.</small>

![deepdream](./image29.png)

![deepdream](./image30.png)

이러한 _DeepDream_ 은 multi scale processing 을 진행하게 되는데, 작은 이미지로 _DeepDream_ 을 수행한 후 점차 이미지를 늘려나가고, 최종 스케일로 수행 후에는 다시 처음부터 반복하게 된다.  

이렇게 반복적으로 해서 도출해 낸 이미지가 위의 예제 들이다.

## Feature Inversion

또 다른 방법으로 네트워크의 다양한 레이어에서 이미지의 어떤 요소들을 포착하는지 알아보자.

![feature inversion](./image31.png)

_Feature Inversion_ 의 방법은 다음과 같다.

- 이미지를 네트워크에 통과시킨다.
- 통과시킨 특징인 activation map 을 저장한다.
- 특징을 가지고 이미지를 재구성한다.

이렇게 재구성되는 이미지를 볼 때 이미지의 어떤 정보가 특징 벡터에서 포착되는지 알 수 있다.  

이 역시 regularization 으로 gradient ascent 를 사용한다.

여기서는 score 를 최대화시키는 것이 아닌 특징 벡터간 거리를 최소화하는 방법으로 진행한다.  

이는 기존에 계산해 놓은 특징 벡터와, 새롭게 생성한 이미지로 계산한 특징벡터 간의 거리를 측정하는 것이다.

total variation regularizer 라고 하는 것을 사용하는데, 이것이 인접 픽셀 간의 차이에 대한 패널티를 부여한다.

이를 통해 생성된 이미지가 자연스럽게 된다.

![feature inversion](./image32.png)

코끼리와 과일 이미지를 _VGG-16_ 에 통과시키는 예제를 생각해보자.  

relu2-2 를 거친 특징 벡터로 재구성하면 거의 완벽한 이미지가 복원되는 것을 볼 수 있다.  

조금 더 깊은 곳은 어떨까?  

공간적인 구조는 많이 유지되고 있으나 디테일이 많이 약해지고 모호해졌다.  

이처럼 낮은 레벨의 디테일들은 네트워크가 깊어질 수록 소실되는 것을 확인할 수 있다.  

## Texture Synthesis

texture(질감?) 합성(synthesis)에 대해서도 간략히 살펴보자.  

![texture synthesis](./image33.png)

위의 예시처럼 입력 패턴에 대해 유사한 패턴이 더 큰 이미지로 나타나게 만드는 것이다.  

![texture synthesis](./image34.png)

이를 해결하기 위해 Nearest Neighbor 를 활용할 수 있다.  
위에서 보다시피 꽤 좋은 성능이 나오는 것을 알 수 있다.

그러나 이는 간단한 텍스쳐에 대해서 성립하는 것으로, 신경망 없이 복잡한 텍스쳐는 만들기에 많이 힘들다.

### Neural Texture Synthesis : Gram Matrix

![neural texture synthesis gram matrix](./image35.png)

2015년 신경망을 이용해 텍스쳐 문제를 풀려는 시도가 있었다.  
이는 앞서 배운 gradient ascent 와 비슷한데, _Gram Matrix_ 에 대해 알아보자.  

자갈 사진으로 예시를 들어보자.

이 사진을 네트워크에 통과시킨 후, 네트워크 특정 레이어의 특징 맵($C \times H \times W$)을 가져온다.  
여기서 $H \times W$ 그리드는 공간 정보를 나타낸다.  

$H \times W$ 의 한 점에 있는 C차원 특징 벡터는 해당 지점에 존재하는 이미지의 특징을 담고 있다고 할 수 있다.

이제 이 특징 맵을 가지고 입력 이미지의 texture descriptor 를 계산할 것이다.

우선 특징 맵에서 서로 다른 두 개의 특징 벡터를 뽑아내는데, 각 특징 열 벡터는 C차원 벡터이다.

이 두 벡터의 외적(outer product)을 계산해서 $C \times C$ 행렬을 만든다.

이 $C \times C$ 행렬은 이미지 내 서로 다른 두 지점에 있는 특징들 간의 co-occurrence 를 담고 있다.

$C \times C$ 행렬의 (i, j) 번째 요소의 값이 크다는 것은 두 입력 벡터의 i번째, j번째 요소가 모두 크다는 의미이다.

이를 통해 서로 다른 공간에서 동시에 활성회되는 특징이 무엇인지 2차 모멘트를 통해 어느정도 포착해 낼 수 있는 것이다.  
이 과정을 H x W 그리드에서 전부 수행해주고, 결과에 대한 평균을 계산해보면 $C \times C$ _Gram Matrix_ 를 얻게 된다.

이 결과를 입력 이미지의 텍스처를 기술하는 texture descriptor 로 사용한다.

![gram matrix](./image36.png)

이러한 Gram matrix 는 주변 공간 정보를 모두 날려버린다.<small>(이미지의 각 지점에 해당하는 값들을 모두 평균화 시킴)</small>

그 대신에 특징들 간의 co-occurrence 만을 포착해 내고 있다.  
이 방법은 계산이 아주 효율적인데, $C \times H \times W$ 차원의 3차원 텐서가 있다고 해도 행렬을 $C \times (HW)$ 로 바꾼 다음에 한 번에 계산하므로써 매우 효율적인 모습을 보이게 된다.

> 왜 제대로된 공분산 행렬을 쓰지 않고 이런 _gram matrix_ 를 사용할까?  
>> 공분산 행렬을 써도 잘 동작하나 이는 계산 cost 가 너무 크다.

자 이제 texture desrciptor 를 만들었으니 이미지를 생성해보자.  
이는 gradient ascent procedure 와 유사한 과정을 거친다.

텍스처 합성도 앞서 본 특징 재구성(feature reconstruction)와 유사하다.

다만 입력 이미지의 특징맵 전체를 재구성하기 보다는 gram matrix를 재구성하도록 하는 차이가 있다.

![neural texture synthesis](./image37.png)

자 이제 실제로 이루어지는 과정을 정리해보자.

1. 실제로 거치는 단계는, 우선 pretrained model를 다운로드 받는다.<small>(_VGG_ 가 많이 쓰임.)</small>
2. 이미지를 VGG에 통과시키고 다양한 레이어에서 gram matrix를 계산한다.
3. 생성해야 할 이미지를 랜덤으로 초기화한다.
4. 원본 이미지와 생성된 이미지의 gram matrix 간의 차이를 L2 norm을 이용해 Loss로 계산한다.
5. Loss 를 backprop 을 통해 생성된 이미지의 픽셀의 gradient 를 계산
6. gradient ascent 를 통해 이미지의 픽셀을 업데이트
7. 위 과정을 반복.

![neural texture synthesis](./image38.png)

위의 예제를 보면 texture 에 대해 _Gram Matrix_ 를 이용한 texture 합성을 볼 수 있다.  

- 얕은 레이어에서는 색상은 유지하나 공간적 구조는 잘 살리지 못한다.
- 깊은 레이어를 보면 큰 패턴들을 잘 재구성하는 것을 볼 수 있다.

### Neural Style Transfer

![neural style transfer](./image39.png)

이제 텍스쳐 합성을 예술 작품에 적용해보자.

_gram matrix_ 를 이용한 텍스처 합성을 Starry night(Van Gogh) 이나 Muse(Picasso)를 텍스처 입력으로 사용하면 어떻게 될까?

![neural style transfer](./image40.png)

결과를 보면 예술 작품의 아주 흥미로운 부분들을 재구성해 내는 모습을 볼 수 있다.

![neural style transfer](./image41.png)

Content Image 는 네트워크에게 최종 이미지가 __어떻게 생겼으면 좋겠는지__ 를 말한다.
그리고 Style Image는 최종 이미지의 __텍스처가 어땠으면 좋겠는지__ 을 알려준다.

- 최종 이미지는 content image 의 feature reconstruction loss 를 최소화
- Style image 의 gram matrix loss 도 최소화

이 두가지 Loss를 동시에 활용하면, style image 화풍의 content image 가 만들어진다.

이를 위해 최종 출력 이미지를 random noise 로 초기화 시킨 후, 네트워크에 content/style image 를 통과시켜 _Gram Matrix_ 와 feature map 을 계산한다.  

![neural style transfer](./image42.png)

반복된 backprop 과 _gradient ascent_ 를 통해 아름다운 결과 이미지를 얻을 수 있다.  

![neural style transfer](./image43.png)

_Style Transfer_ 는 _DeepDream_ 에 비해 조정해야할 부분이 많다.  

![neural style transfer](./image44.png)

다양한 style image 를 적용해 볼 수 있다.

![neural style transfer](./image45.png)

Content/Style loss 를 joint loss 로 사용하기 때문에 이를 적절히 조절하면 어디에 초점을 둘 것인지 선택이 가능해진다.(hyperparameter)

![neural style transfer](./image46.png)

위의 예시는 같은 Content/Style image 에 대해 style image 의 size 만 다르게 설정한 경우이다.  

이 밖에도 _DeepDream_ 에서 살펴본 Multi scale processing 을 통해 [고해상도 이미지](https://github.com/jcjohnson/neural-style/#Multi-GPU-scaling)를 얻을 수도 있다.  

![neural style transfer](./image47.png)

동시에 여러장의 style image 를 이용해 style loss 의 _Gram Matrix_ 를 계산하는 방법도 있다.

### Fast Style Transfer

그러나 이러한 _Style Transfer_ 의 단점은 속도가 아주 느리다는 것이다.  
이렇게 이미지를 만들기 위해서는 backprop/forward 를 굉장히 많이 해야하는데, 필연적으로 느려질 수 밖에 없게 된다.  

이를 해결하는 방법에 대해 알아보자.

![fast style transfer](./image48.png)

바로 _Style Transfer_ 를 위한 네트워크를 학습시키는 방법이다.  

여기서는 Style Image 를 고정시켜놓는다.  

즉, 합성하고자 하는 이미지의 최적화를 전부 수행하는 것이 아니라 Content image 만을 입력으로 받아서 결과를 출력할 수 있는 단일 네트워크를 학습시키는 방법이다.

학습 시 content/style loss 를 동시에 학습시키고 네트워크의 가중치를 업데이트하게 된다.

학습은 오래 걸릴 수 있으나 그 후 이미지를 네트워크에 통과시키면 결과가 바로 나오게 된다.

![fast style transfer](./image49.png)

아주 잘 나오는 것을 확인할 수 있다.

![fast style transfer](./image50.png)

여러 개의 스타일을 동시에 돌려볼 수도 있다.

알고리즘을 조금 변형한 방법도 있다.

앞서 보여드린 이 네트워크는 앞서 배웠던 segmentation 네트워크와 아주 유사하게 생겼다.

> Sementic segmentation 에서는 다운 샘플링을 여러 번 진행하고 transposed conv로 업샘플링을 한다.  
> Segmentic segmentation과 다른 점이라고는 출력이 RGB 이미지라는 점이다.  
> 그리고 네트워크 중간에 batch norm이 들어간다.  
> 이 논문에서는 batch norm 대신에 instance norm 을 사용해서 더 좋은 결과를 끌어낸다.

![fast style transfer](./image51.png)

지금까지의 방식들의 단점은 네트워크 하나당 하나의 _Style Transfer_ 만이 가능하다는 것이다.  

그러나 최근 구글에 의해 하나의 네트워크로 다양한 _Style Transfer_ 가 가능한 방법이 고안되었다.  

Content/Style image 를 동시에 넣는 방식으로 다양한 스타일을 만들어내게 된다.  
이는 아주 강력해서 실시간으로도 동작할 수 있다.  
뿐만 아니라, style blending 도 가능해 서로 다른 스타일들을 섞어서 표현 가능하게 만들어주기도 한다.  

![fast style transfer](./image52.png)

각 스타일이 자연스럽게 변화하는 것을 확인할 수 있다.  