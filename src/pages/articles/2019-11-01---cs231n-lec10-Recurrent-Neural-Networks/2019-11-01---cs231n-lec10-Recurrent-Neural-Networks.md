---
draft:     false
title:     "[cs231n-lec10] Recurrent Neural Networks"
date:      "2019-11-01T17:27:23"
layout:    post
path:      "/posts/cs231n-lec10-Recurrent-Neural-Networks/"
category:  "Machine Learning"
tags: 
    - cs231n
    - Deep Learning
    - RNN
    - LSTM
description: "RNN 에 대해 학습한다. 그리고 LSTM(Long Short Term Memory)를 알아보고 그 성능에 대해 간략히 살펴본다."
---

## Recurrent Neural Networks

이제부터 우리는 _Recurrent Neural Networks(RNN)_ 에 대해 알아보자.  

우리는 지금까지 One to One 모델의 _Vanilla Neural Network_ 를 배웠다.  

이는 하나의 고정된 크기의 입력(fixed size object input)을 넣으면 hidden layer 를 거쳐서 결과로 나오는 형식이었다. 에를 들면 classification 문제 등에서 활용할 수 있는 경우들이었다.

그러나 머신 러닝의 분야에서는 이러한 입력과 출력에 조금 더 유연함을 주는 것에 대해 생각해 볼 필요가 있다.

![Recurrent Neural Networks](./image1.png)

위 이미지에서 가장 왼쪽에 나타난 것이 바로 우리가 지금까지 알아본 One to One 모델이 되겠다.  

그렇다면 이제 다양한 종류의 입력과 출력에 대해 봐보자.  

고정된 크기의 입력(e.g. 이미지)에 대해 다양한 길이의 순차적인 결과값(sequence of variable length)을 출력으로 갖는 모델을 One to Many 라고 부른다.  
이는 caption 등이 그 예가 될 수 있다.(caption 은 단어의 갯수가 달라질 수 있기 때문에)  

반대로 생각해보면 입력의 길이가 다양할 수 있고 출력이 고정된 결과값이 될 수 있다.  
예를 들어, 입력으로 텍스트 일부분(piece of text)이 들어간다고 했을 때, 결과값으로 그 텍스트의 감정(Sentiment Classification)에 대해 나타내는 경우를 생각해 볼 수 있다. 또는, 비디오를 입력 받을 수 있는데, 비디오의 프레임이 다양한 갯수의 입력이 되게 되고, 이 비디오에 대해 결과적으로 어떤 액션이 취해지고 있는지 분류해보는 것이 해당될 수 있다.  

다음으로는 여러 입력이 들어가서 여러 결과값이 나오는 것이 될 수 있다.  
번역기가 예가 될 수 있는데, 입력으로 영어를 넣었을 때 한국어로 번역된 결과가 나와야하는 경우를 생각해 볼 수 있다.  
이 경우에 대해 영어와 한국어의 길이가 다를 수 있기 때문에 이 경우를 생각해주어야한다.  

그리고 입력과 결과값의 길이가 같을 수 있는데, 비디오를 넣었을 때 각 프레임마다 물체가 어떤 것인지 지속적으로 판단하는 것 등에 대해 생각을 해 볼 수 있다.  

이렇게 입력이 고정된 크기를 가지지 않을 경우 순서에 대해 생각을 해 볼 수 있다. 위의 비디오와 같은 예는 가장 첫 프레임부터 읽어들인다고 생각을 할 수 있는데, 이런 것을 sequence processing 을 통해 처리를 하게 되는 것이다.

그런데 단일 이미지에서 어떤 숫자가 써져있는지 분류하는 MNIST 같은 경우는 이미지가 고정되어있는 것이다. 이런 것을 우리는 Vanilla Neural Network 로 분류기를 만들어서 해결했었는데, 이를 RNN 으로도 해 볼 수 있다.  

![Non-sequence data glimpses](./gif1.gif)

이와 같은 방법으로 RNN 을 이용해 fixed size input 에서 fixed size output 을 만들어 낼 수 있다.  
이는 series of glimpses 를 이용해 classification decision 을 만들어내게 되는 것이다.  
즉, 순차적으로 이미지의 부분들을 본 후 최종적으로 이 이미지에 어떤 숫자가 적혀있는 지를 판단하는 것이다.

이를 이용해 순차적으로 이미지를 만들어내는 데도 사용할 수 있다.  

### Vanilla Recurrent Neural Network

그렇다면 이 RNN 이 어떻게 작동되는 것일까?  

![RNN base architecture](./image2.png)

RNN 의 핵심은 internal state(hidden state) 라고 하는 것을 가지고 있는데, 이것이 순차적으로 작업이 진행될 때 상태를 저장하고 업데이트하면서 출력에 영향을 주게 되는 것이다.

$$
h_t = f_W \left( h_{t-1}, x_t \right)
$$

로 표현할 때, $h_t$ 는 new state, $f_W$ 는 function, $h_{t-1}$ 는 old state, $x_t$는 t time step 에서 input vector 를 의미한다.  
이것이 _recurrence formular_ 이고, 이를 매 단계에서 적용시켜 RNN 을 구현하게 된다.  
이 internal state 를 갱신하면서 작업을 진행하고 출력을 해야할 필요가 있다면 _y_ 를 출력하게 되는 식이다.

이 때, function 과 그 parameter 는 각 time step 마다 동일한 것임을 알고 있어야한다.  

![vanilla rnn](./image3.png)

위의 예시는 간단한(vanilla) RNN 의 모습이다.  
여기서 $W_{hh}, W_{xh}, W_{hy}$ 는 각각 hidden 에서 오는 W, x 에서 RNN 으로 들어오는 W, RNN 에서 y 로 넘어가는 W 를 지칭한다.  
즉, 각 영역별로는 W 가 다를 수 있으나 한 영역에서는 같은 W 를 쓰는 것이다.  

