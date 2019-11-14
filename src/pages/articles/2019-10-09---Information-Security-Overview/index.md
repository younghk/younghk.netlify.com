---
title: "정보보호(Information Security) 공부를 시작하며"
date: "2019-10-09T02:13:00.1234"
draft: false
path: "/posts/information-security-overview/"
layout: post
category: "Information Security"
description: ""
tags:
  - Information Security
---


## 정보보호란?
간단하게 생각하면 정보를 보호하는 것이다.  
무엇으로부터? 위협으로부터!  
정보보호에 있어서 중요한 3가지 목표가 있는데, `integrity`, `availability`, `confidentiality` 이다.  
우리말로 표현하면 `무결성`, `가용성`, `기밀성` 정도이다.  
이 세 가지를 유지하는 것을 목표로 하는 것이 정보보호에서 말하는 보안이다.  
세 가지 용어에 대해 간략하게 정리하면 아래와 같다.
<br>
- Integrity : 부적절한 데이터의 변경 또는 제거로부터 보호하는 것
- Availability : 허락된 사용자가 데이터에 접근할 때 방해받지 않도록 하는 것
- Confidentiality : 허가받지 않은 객체가 데이터에 접근하거나 확인하는 것을 방지하는 것
<br>

또한 정보보호에서 추가적인 목표가 몇 가지 존재하는데, 이는 아래와 같다.  

- Authenticity\[진정성\] : 허가받은 사용자라고 알린 사용자가 신뢰할 수 있는 source로부터 온 것인지
- Accountability\[책임(추적)성\] : 정보 시스템에서 객체의 모든 활동은 유일하게 추적되어야 함
<br>

## 들어가기에 앞서서

본격적으로 들어가기에 앞서 간단하게 생각을 하고 넘어가보자.  

> 앞에서 보기에 튼튼해보이는 집이 있다. 이 집은 과연 안전한가?
<br>

물론 튼튼할 수도 있다. 그러나 벽이 약할 수도 있고, 창문을 깨서 들어갈 수도 있을 것이다.  
앞에서 보기에 튼튼해보이지만 집의 뒷모습은 허름할 수도 있을 것이다.  <br>

이렇듯 보이는 것 외에도 생각을 해봤을 때 취약할 부분을 생각해 볼 수 있다.  
이러한 것이 보안취약점이 되는데, 그렇다면 우리는 이러한 보안취약점을 어떻게 막을 수 있을까?

여기서 우리는 세 가지 중요한 질문을 던지고 가야한다.

1. 어떤 것을 보호해야하는가?
2. 이것들이 어떻게 위협을 받을 수 있는가?
3. 이 위협을 어떻게 대처할 수 있을까?

_우리는 무엇이 중요한지 먼저 정의해야한다._ __(1)__  
경우에 따라 중요할 수도 있고, 중요하지 않을 수도 있기 때문이다.  
모든 것을 보호할 수도 없고, 어떤 정보는 공개되어야만 할 수도 있다. 그렇기에 무엇이 중요한지 먼저 정의를 해야만 한다.  
위의 집에 대해 생각해보자. 만약 집 안에 귀중품이 있다. 그렇다면 집을 보호해야할 이유가 생기는 것이다. 그러나 폐가라면? 굳이 보호할 필요가 없지 않을까?  
중요한 것을 정의했다면, _이것이 어떻게 위협을 받을 수 있는지 생각해 보아야 한다._ __(2)__  
다양한 위협이 있을 수 있다. 최대한 다양한 방법을 생각해 볼 수 있다.  
집의 지붕에 구멍을 뚫어서 보물을 훔치러 들어올 수도 있는 것 아니겠는가?  
_위협을 생각했다면, 이제 그에 맞는 예방책을 세워야 할 것이다._ __(3)__  
유리창을 방탄유리로 만들던가, 골조를 튼튼하게 한다던가, 최대한 막아볼 수 있겠다.  <br>

위협과 그 예방책은 다양할 것이다.  
그러나 안타깝게도 위의 세 질문에 대한 ___완벽한 답은 없다.___  
최선을 다해 설계하고 보완해나가야하는 것이다.  

## 위협

우리는 위에서 위협에 대해 얘기를 해보았다.  
이는 큰 범위에서 뭉뚱그려서 생각한 것인데, 세분화한다면 다음과 같다.  


1. Vulnerabilities\[취약점\] : asset의 약한 부분
    - Corrupted(loss of integrity)
    - Leaky(loss of confidentiality)
    - Unavailable or very slow(loss of availability)
2. Threats\[위협\] : 잠재적 공격 가능성
    - 취약점을 드러나게 함으로써 위협이 가시화 됨
    - potential security harm
3. Attacks : 위협의 실체화
    - based on action
        - Passive attack : system resource에 영향을 끼치지 않으면서 정보를 얻음
        - Active attack : system resource나 process에 직접적인 영향을 미침
    - Based on authority
        - Inside attack : 허가된 내부자로부터 발생. 보안정책이 무효화되는 문제가 있음
        - Outside attack : system 외부에서 발생
<br>

이러한 것들로부터 보안을 유지하는 것은 쉽지 않은 일이다.  
그래서 사람들은 Security Architecture에 대해 쳬계적으로 접근하고자 한다.(ex. [ITU-T X.800](https://www.itu.int/rec/T-REC-X.800-199103-I))  

## Security Architecture

우리는 OSI Security Architecture에 대해 짧게 생각해보고 갈 것이다.  
이는 `Security Attack`, `Security Mechanism`, `Security Service` 가 있다.  

### Security Attack

먼저, `Security Attack` 을 살펴보자.  
위에서 살펴봤던 Attacks의 action 부분인데, Passive attack은 감지하기 어렵고, Active attack은 막는데 어려움이 있다.  

#### Passive Attack

Passive attack 을 보면, 네트워크 상에서 지나다니는 정보를 몰래 지켜보고 있는 것이다.  
네트워크는 그 구조와 특성상 보안에 취약해서 다른 사람이 정보를 쉽게 읽을 수 있고, 누가 보는지 알아차리기가 어렵기에 감지하기가 어렵다. 정보를 가리기 위해 암호화(encryption)을 수행해도, 여전히 많은 정보(ex. 송/수신자 정보)를 얻을 수 있다. <small>이러한 송/수신자 정보를 가리기 위해 `tor`를 사용하기도 한다.</small>  

#### Active Attack

Active attack 은 조금 더 위험할 수 있다. system에 접근을 시도하는 공격일 수 있는데, 아래와 같은 질문에 대해 생각해보자.  

> A와 B는 서로 신뢰하는 사용자이다. 여기서 허가받지 않은 C(공격자)가 A인척 B에게 보낸다면?
  
여기서 B의 입장에서 위와 같은 문제를 방지하고자 한다면 certificate을 이용해 진짜 A가 보낸 것이 맞는지 확인하는 과정을 통해 방어해 볼 수 있겠다.  
그렇다면 다음과 같은 가정을 달면 어떨까?

> A가 B에게 메세지를 보내고 있다. 중간에 C가 A의 메세지를 탈취해서 A인척 B에게 보낸다면?
  
마찬가지로 certificate 을 이용하면 되지 않겠느냐 물을 수 있겠지만, 정말 메세지를 그대로 보내서 B의 입장에서 A가 보냈는지, C가 보냈는지 알 수 없다면? 여기서 A가 B에게 보내는 메세지는 시스템에 위해를 가하는 attack 이 아니다. 이럴 때도 C의 행위는 attack 일까?  

__그렇다.__  

어째서 이는 attack 이 될 수 있을까?  
만약 A가 B에게 보내는 메세지가 "계좌에서 100만원 출금해줘." 였다면?  
C는 해당 메세지를 반복적으로 보냄으로써 A의 계좌를 텅장으로 만들어 버릴 수 있을 것이다!!  
이처럼 같은 action을 취하면서 victim을 만들어 낼 수도 있다.

이러한 attack을 방지하기 위해서는 counter를 이용해 예방해 볼 수 있다.
  
이 외에도 다양한 active attack이 있을 수 있다.  
DDoS(Distributed Denial of Service, 분산서비스거부공격)도 쉽게 생각해 볼 수 있는 active attack이다. 그러나 요즘 대형 서버들의 성능은 이러한 ping 공격으로는 쉽게 다운시킬 수 없다. 그러나 system자체가 감염되지 않도록 방어는 해야 한다.

### Security Mechanism

다음으로는 `Security Mechanism` 이다.  
이는 Security Attack 을 탐지, 방어 또는 공격으로부터의 복구를 위한 것이고, 다양한 매커니즘이 있다.
그 중, 가장 많이 쓰이는 것은 바로 _암호화 기술(Cryptographic Techniques)_ 을 사용하는 것이다.

### Security Service

여기서 우리는 다음의 용어들을 보게 된다.
- Authentication\[인증\]
- Access Control\[접근 제한\] : 허가되지 않은 접근을 막는 것
- Data Confidentiality\[기밀성\]
- Data Integrity\[무결성\]
- Non-Repudiation\[부인 방지\] : 정보를 보낸 사람이 나중에 보냈다는 사실을 부인하는 것을 막는 것
- Availability\[가용성\]

## Model for Network Access Security

네트워크에서는 다음과 같은 것들이 필요하다.
1. 정보보호를 위한 적절한 algorithm
2. 이를 통한 보안 정보(secret information)의 생성
3. 보안 정보를 공유할 안전한 방법
4. 이를 전송할 특정한 protocol
  
대부분의 네트워크 구조는 client-server 구조이고, 여기서 네트워크를 통해 흘러 들어오는 관문인 __gatekeeper__ 가 중요하다.
  
> 본 포스트는 _정보보호_ 를 공부하며 정리한 글 입니다.  
> 잘못된 내용이 있다면 알려주세요!  
> 감사합니다 :)
