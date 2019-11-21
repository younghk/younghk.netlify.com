---
draft:     false
title:     "[cs231n-lec12] Detection and Segmentation"
date:      "2019-11-19 17:05:52"
layout:    post
path:      "/posts/cs231n-lec12-Detection-and-Segmentation/"
category:  "Machine Learning"
tags: 
    - cs231n
    - CNN
    - Detection
    - Segmentation
    - RCNN
description: "스탠포드 cs231n lec-12 강의를 보고 정리한 포스트 입니다. detection 과 segmentation 에 대해 설명합니다. 2019년 강의노트를 기반으로 작성되었습니다."
---

<small>최종 수정일 : 2019-11-22</small>

## Computer Vision Tasks

![computer vision tasks](./image1.png)

지금까지 Neural Network 를 이용해 이미지 속에 있는 물체가 무엇인지 판별하는 것을 학습했다.  

컴퓨터 비전에 있어서 이는 단순한 작업이고, 해야할 것들이 더 남아있는데 위의 예시들이 바로 그것을 짧게 요약한 것이다.  

물체를 분류하는 classification.  
각 부분이 무엇을 의미하는지 구별하는 semantic segmentation.  
이미지 속에 있는 물체를 구별하고 위치까지 표현하는 object detection.  
pixel 단위로 object detection 을 수행하는 instance segmentation.  

그 중 object detection 과 instance segmentation 은 여러 물체에 대해서 작동해야하는 것을 볼 수 있다.  

하나씩 살펴보자.

## Semantic Segmentation

![semantic segmentation](./image2.png)

_semantic segmentation_ 에서는 각 pixel 에 대해 category 를 나누게 된다.  
즉, 물체(object)와는 상관이 없이 pixel 단위로 segmentation 이 이뤄진다.

![semantic segmentation idea sliding window](./image3.png)

segmentation 을 하는 방법에 sliding window 기법이 있는데, 일정한 크기의 patch 를 원본 이미지와 쭉 비교해보면서 해당 부분의 pixel 이 어떤 category 값을 가지는지 확인한다. 즉 매 번 _CNN_ 을 통과시키게 된다.  

이는 매우 비효율(computational cost 가 큼)적인 방법인데, 중복되는 patch 들 간의 공유된 feature 를 재사용하지 않는다.

![semantic segmentation idea fully convolutional](./image4.png)

위에 나타난 sliding windows 기법의 단점을 극복하기 위해 전체 이미지를 convolution 하는 방법을 고안하게 되었다. 3x3 filter 를 이용해 이미지 크기를 유지하면서 convolution 을 수행하여 한 번에 이미지 전체를 _CNN_ 에 넣을 수 있고 이를 이용해 pixel 을 한 번에 전부 예측할 수 있다.  

그러나 전체 이미지를 _CNN_ 에 통과시키는 것은 여전히 비효율적인 방법이다.<small>입력 이미지의 spatial size 를 계속 유지해야하기 때문</small>  

![semantic segmentation idea fully convolutional with downsampling and upsampling](./image5.png)

그래서 maxpooling 또는 strided convolution 을 통한 downsampling 을 진행해서 크기를 줄인 후, unpooling 을 통해 upsampling 을 하면서 연산을 효율적이게 만들어 보게 되었다.  

이렇게 하는 이유는 줄어든 spatial resolution 을 원본 크기로 맞추기 위함이다. 이 과정이 바로 upsampling 이다.

### Unpooling

![in network upsampling unpooling](./image6.png)

말 그대로 pooling 을 undo 하는 것이다.  
unpooling 지역의 receptive field 값을 복사해서 채워넣게 된다.  

그러나 오른쪽의 경우와 같이 나머지 값이 0일 경우 매우 좋지 못한 결과(bed of nails)를 얻게 될 것이다.

### Max Unpooling

![in network upsampling max unpooling](./image7.png)

_Max Pooling_ 이 어땠는지 생각해보며 _Max Unpooling_ 에 대해 알아보자.  
이는 max 값과 위치(공간 정보)를 기억해 둔 뒤 나중에 upsampling 할 때 해당 값만 복구 시킨 후 나머지는 0의 값을 채워 넣는 것이다.

이 때 fixed function 을 사용하게 된다.  

### Learnable Unpooling : Transpose Convolution

![learnable unpooling transpose convolution](./image8.png)

_Transpose Convolution_ 은 학습이 가능한 방법이다.  

![learnable unpooling transpose convolution](./image9.png)

위의 방법이 일반적인 convolution 의 연산이다.  
여기서 strided convolution 은 2칸(stride = 2)씩 움직인다. 출력이 한 픽셀 움직일 때 입력은 두 픽셀이 움직이게되는 것이다.

![learnable unpooling transpose convolution](./image10.png)

_Transpose Convolution_ 은 위에서 진행된 과정의 반대이다. 입력과 출력의 크기가 반대가 된 것을 확인하자.  

위에서는 내적(dot product)을 수행하였지만 여기서는 feature map 에서 값을 선택하고 선택한 scalar 값과 필터(3x3)를 곱해준다. 그리고 출력의 3x3 공간에 넣게된다.  
filter 의 크기와 stride 의 크기에 의해 overlap 되는 부분이 생기게 되는데, 이 부분에 대해서는 summation 을 진행한다.

![learnable unpooling 1d example](./image11.png)



![convolution as matrix multiplication 1d example](./image12.png)

![convolution as matrix multiplication 1d example](./image13.png)



<small>이에 대한 보충자료는 [여기](https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0)에서 더 확인할 수 있다.</small>

## Object Detection

![object detection impact of deep learning](./image14.png)

_Object Detection_ 은 _Classification + Localization_ 이라고 볼 수 있다.  
이미지 안에 있는 객체를 분류하고 그 위치까지 판별해야하는 문제이고 그 위치를 범위로써 나타내야한다.

이 때, localization 은 이미지 안에 물체가 하나만 있는 경우를 상정한다.

![object detection single object](./image15.png)

_FC layer_ 가 하나 더 있는 것을 볼 수 있는데 이는 x, y, w, h 로 _bounding box_ 의 위치를 의미한다.  
이로써 두 개의 값을 출력하게 되는데 하나는 class score 가 되고 나머지는 위의 bounding box 의 정보가 된다.  

학습을 진행할 때는 두 loss 를 학습하게 되고 이것이 _Multitask Loss_ 가 된다.  

이 때 _CNN_ 구조를 통과시키게 되는데 처음부터 학습하는 것은 쉽지 않고 효율적이지도 않으므로 보통 _ImageNet_ 의 pre-trained 모델을 사용한다.<small>앞서 배웠던 _transfer learning_ 을 상기하고 가자.</small>

![obejct detection multiple objects](./image16.png)

_Object Detection_ 은 Computer Vision 에서 핵심적인 task 이다.  
특히 우리는 이미지에 하나의 물체만 담지는 않으므로 보통의 이미지에는 다양한 물체들이 있을 수 있다.  

앞서 보았던 _Localization_ 과의 차이점은 _Object Detection_ 에서는 Multiple Object 에 대해 작업을 수행한다는 것이다.  

첫 이미지에서 보듯 고양이가 한 마리만 있는 이미지라면 bounding box 는 4개의 숫자로 이루어진 하나만 나오면 되겠지만,  
두 번째 이미지에서 보듯 두 마리의 강아지와 한 마리의 고양이가 있다면 이는 세 배가 되게 될테고,  
마지막 오리 사진을 본다면 정말 많이 필요하게 될 것이다.

![object detection multiple objects](./gif1.gif)



![obejct detection multiple objects](./image21.png)

![region proposals selective search](./image22.png)

## R-CNN

![](./image23.png)

![](./image24.png)

![](./image25.png)

![](./image26.png)

![](./image27.png)

## Fast R-CNN

![](./image28.png)

![](./image29.png)

![](./image30.png)

### RoI Pool

![](./image31.png)

![](./image32.png)

![](./image33.png)

![](./image34.png)

### RoI Align

![](./image35.png)

![](./image36.png)

![](./image37.png)

![](./image38.png)

## Faster R-CNN

![](./image39.png)

![](./image40.png)

![](./image41.png)

![](./image42.png)

![](./image43.png)

![](./image44.png)

![](./image45.png)

## Single-Stage Object Detectors

![](./image46.png)

![](./image47.png)

## Instance Segmenatation: Mask R-CNN

![](./image48.png)

![](./image49.png)

![](./image50.png)

![](./image51.png)

![](./image52.png)

## Aside

![](./image53.png)

![](./image54.png)

![](./image55.png)

![](./image56.png)

![](./image57.png)

![](./image58.png)

![](./image59.png)

![](./image60.png)




> 이 포스트는 스탠포드의 [cs231n](http://cs231n.stanford.edu) 강의를 보고 공부한 포스트입니다.  
> 잘못된 것이 있을 수 있습니다.  
> 댓글로 알려주시면 감사합니다!  
