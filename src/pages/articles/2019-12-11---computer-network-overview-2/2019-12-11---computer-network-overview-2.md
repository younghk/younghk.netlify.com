---
draft:     true
title:     "Computer Network Overview Part 2"
date:      "2019-12-11 17:41:09"
layout:    post
path:      "/posts/computer-network-overview-part-2/"
category:  "Computer Network"
tags: 
    - Computer Network
    - ARP
    - ICMP
    - NAT
    - Mobile IP
    - IPsec
    - IP Routing Protocol
    - TCP Connection
    - Flow Control
    - Congestion Control
description: "컴퓨터 네트워크를 개략적으로 공부하고 정리한 포스트 입니다. ARP, ICMP, NAT, Mobile IPv4, IPv6, IPsec, IP Routing Protocol, TCP Connection, Flow Control, Congestion Control 등을 다룹니다."
---

지난 [포스트](https://cheong.netlify.com/posts/computer-network-overview-part-1/)에 이어서 컴퓨터 네트워크와 관련해 개략적으로 알아보자.  
이번 포스트에서 다룰 것들은 다음과 같다.  

- ARP
- ICMPv4
- NAT
- Mobile IP(v4, v6)
- IPsec
- IP Routing Protocol
- TCP Connection
- Flow Control
- Congestion Control

하나씩 가볍게 알아보자.

## Address Resolution Protocol(ARP)

_Address Resolution Protocol(ARP)_ 은 32-bit IP address 를 48-bit Ethernet address(MAC address) 로 변환해주는 프로토콜이다.  

_ARP_ 는 수동과 동적(dynamic)으로 mapping 이 가능하다.  

수동의 경우는 필요에 의해서 하게되는데, 단점으로 너무 느리고, 에러가 있을 수 있으며 manual updating 이 필요하다는 점이다. 이 말은 새로운 기기가 등록이 되면 다른 모든 것을 건드려줘야한다는 의미다.  

따라서 dynamic mapping 의 경우가 여러모로 이점이 많게 된다.  

- 우선 application 과 user 를 생각할 필요가 없게되고 system administrator 도 걱정의 대상이 아니게 된다.  
- 또한 어떤 network layer protocol 로도 사용이 가능하다.
  - 즉, IP-specific 한 것이 아니라는 의미이다.  
- supported protocol in datalink layer
  - datalink layer protocol 은 아니고 도와주는 프로토콜이다.
- need datalink with broadcasting capability
  - broadcasting 으로 진행되기 때문에 datalink 가 필요하게 된다.
  - e.g. ethernet shared bus

### ARP Operation

sender 에서 receiver 로 packet 을 전송하기 위해서는 서로의 주소를 알고 있어야 한다. 이 때, 같은 네트워크에 있는 것을 안다면(network prefix 를 검사)

![routing table](./image2.png?nf_resize=fit&h=400)

1. 이 때 sender 는 ARP table 을 보고 receiver 의 MAC address 를 알아 내려고 한다. 그러나 초기에는 비어있을 것이고, 이 때 lan1 port 로 ARP request 를 보내서 receiver 의 MAC address 를 구하고자 하게 된다.  
2. ARP request 는 broadcast 로 진행된다 하였으므로 위의 예제에서는 1.1.1. 의 네트워크의 모든 기기가 ARP request 를 수신하게 된다.(같은 네트워크 기기이므로)

이 때 보내게되는 packet 의 헤더를 보면 다음과 같다.

![](./image3.png)

이 때 ARP Header 에서 sender 인 나의 주소를 보내는 이유는 중복되는 행동을 막고자 함이다.  

![](./image4.png)

현재까지의 상황을 표시하면 이렇다.

![](./image5.png)

3. 여기서 switch 1(S1) 이 ARP packet 을 받고 incoming packet 의 source MAC address 를 기록한다.  
이로 인해 S1 의 MAC table 은 {MAC address m1 is connected to port fe1} 과 같이 기록을 진행하게 된다.  
이 과정을 switch 의 self-learning 이라고 한다. 이는 소스 주소가 어떤 port 로 들어왔는지 계속 기억하려는 과정이다.  
이는 나중에 어떤 packet 을 받았을 때 주소를 콕 집어서 넘겨주기 위한 작업으로 볼 수 있다.  

4. S1 은 목적지 MAC 주소를 확인한다. 이 값이 broadcast address 이므로 receiving port 를 제외한 나머지 port 들로 이 packet 을 흘려보낸다.  
이로 인해 위의 예제에서 (2)번 과정으로 R1 과 SVR2 가 패킷을 받게 된다.

5. R1 은 이 패킷의 target IP address 를 보고 이 주소가 자신의 것이 아님을 알고 discard 시킨다. 이는 broadcast 를 router 가 forward 시키지 않기 때문이다.

6. SVR2 는 자신을 호출한 것을 확인하고 ARP reply 를 보내게 된다. 이 때의 패킷은 unicast packet 이 된다.  
  ![svr2 arp table](./image6.png)
  SVR2 의 ARP table 이 ARP learning 을 통해 table 을 작성하는 것을 확인하자.  
  ![svr2 responds with arp reply](./image7.png)
  ARP reply 가 전송된다.  
  ![ether arp headers](./image8.png)
  이 때 Ethernet Header 와 ARP Header 는 위와 같다.

7. S1 은 ARP reply 를 받고 source MAC learning 을 진행한다.

8. S1 은 MAC Table 에서 m1(sender=SVR1) 의 포트를 lookup(Destination MAC Lookup) 해 ARP reply 를 forwarding 한다.
  ![s1 mac table](./image9.png)

9. SVR1(sender) 가 ARP reply packet(MAC address m2 of 1.1.1.20) 의 값을 ARP table 에 기록한다.  
  ![svr1 arp table](./image10.png)

10. 이제 SVR1(sender) 는 SVR2(receiver) 에게 IP packet 을 보낼 수 있게 되었다.  
  sender 는 IP packet 을 port lan1 을 통해 전송하게 된다.  
  ![svr1 sends ip packet to svr2](./image11.png)

![arp operation](./image12.png)

### Routing with ARP

방금 전 예제는 Router 를 넘어가지 않는 경우였다.  
이번에는 Router 를 넘어가서 다른 네트워크로의 ARP 과정을 살펴보자.  

![routing](./image13.png)

위와 같이 routing 이 되는 경우를 생각해보자.  
일단은 위의 단순 ARP 와 초기 단계는 비슷하다.

1. SVR1 이 ARP request 를 보내고 router R1 이 ARP reply 로 응답한다. 그 후 SVR1 이 IP packet 을 R1 에 보내게 된다.
2. 이제 R1 에서 ARP request 를 보낸다.  
  ![r1 routing table](./image14.png)
  
3. switch S2 는 R1 이 보낸 ARP request packet 의 source MAC address(R1 MAC) 을 기록한다.  
  ![s2 mac table](./image15.png)

4. SVR3 가 ARP reply 로 응답하게 된다.
  ![svr3 arp table s2 mac table](./imag16.png)  

5. R1 은 ARP reply packet 으로 ARP table entry 를 추가하게 된다.  
  ![r1 arp table](./image17.png)

6. R1 이 IP packet 을 SVR3 에게 보낸다.  
  ![traffic flow packet format](./image18.png)

![packet flow with header format](./image1.png)

여기서 우리는 layer 마다 scope 가 다르다는 사실을 확인할 수 있다.

### Gratuituous ARP

내 IP 를 누군가 쓰고 있는지 check 하고 싶을 때 사용하게 된다.  

![arp packet](./image19.png)

이는 send IP address = target IP address 로 패킷을 보내게 된다.

이 외에도 ARP table 을 update 하고 싶을 때도 사용하게 된다.  
예를 들어 A, B 가 있을 때 B 의 MAC 주소가 변경되었다.(1 -> 2)  
A 입장에서는 update 를 해야하므로(1이 저장되어 있는 상태) B 가 _gratuituous ARP_ 를 보내서 받게 된다.

### Proxy ARP

_ARP_ 는 어떤 상황이던지 직접 대답을 하는것인데, 중간에서 대신 대답하는 것이 _Proxy ARP_ 이다.

이러한 _Proxy ARP_ 를 사용할 경우 네트워크 수정 없이 하나의 router 로 두 개의 subnet 을 관리할 수 있으나 entry 가 늘어나는 단점이 있다. 또한 ARP traffic 이 증가하기도 한다.

## ICMPv4

_ICMP_ 는 __reporting(error)__ 를 위해 주로 사용된다.  
이는 IP header 에 쓸만한 정보가 몇 없기 때문에 기능적 한계를 지니게 되는 태생적 문제점을 메꾸고자 도입되었다.  

_ICMP_ 의 __error reporting message__ 는 IP packet 의 전송과정에서 문제가 생겼을 경우 사용하게 된다.

1. Destination Unreachable
    - router 나 host 가 datagram 을 forward 하지 못하는 경우  
    - routing table 이 잘못 설정되어있는 경우 일어날 수 있으며 빈번함  
2. Redirect  
    - 다른 더 좋은 경로가 있을 경우 이를 알려줌
3. Source Quench(Optional, 발신지 억제)
    - router 성능이 떨어지거나 네트워크가 혼잡할 경우 패킷을 천천히 보내도록 함
4. Time Exceeded
    - time to live exceeded in transit(gateway)
    - fragment reassmebly time exceeded(host)
5. Parameter Problem  
    - datagram header 가 모호할 경우(ambiguity)

또한 __query message__ 로도 사용되는데, 관리자가 router 나 다른 호스트로부터 특정 정보를 얻기 위함이다.

1. Echo  
    - echo&echo reply 가 많이 쓰인다.
    - 주로 네트워크의 상태(alive)를 파악하기 위함
2. Router solicitation and Router advertisement  
    - host 에서 IP datagram 을 subnet 안의 어떤 기기에 보낼 때, 적어도 하나의 operational router 가 subnet 안에 탐지된 상태여야 한다. 이를 위해서는 _router discovery_ 가 필요하다.  
    - Router Discovery :  
        - Router advertisement : router 가 자신을 알려줌, 보통 7~10분 마다 보냄
        - Router solicitation : host 에서 router 를 찾아봄
        - 최초 연결 시에는 router solicitation 을 즉각적으로 사용해서 advertisement 를 바로 받을 수 있게끔 한다. 안그러면 통신을 못하기 때문!
3. Timestamp
4. Address Mask

## Network Address Translator(NAT)

_NAT_ 의 동작은 source 또는 destination 의 주소를 __re-writing__ 해서 router 나 firewall 을 통과할 수 있게 한다.  
이를 이용하면 private 주소를 할당 받은 기기도 외부와 통신을 할 수 있게 된다.  
re-writing 을 할 때 IP header 를 고쳐야한다.  

이는 IP address 의 양적 부족함을 극복하기 위한 방법이고, 상당히 잘 작동하므로 Ipv6 가 적극적으로 쓰이지 않는 하나의 이유가 된다.  

- public IP address sharing
- 쉽게 확장이 가능하다.(쉽게 local 에 새로운 기기를 연결할 수 있다.)
- greater local control : 관리자가 private network 의 control 은 여전히 할 수 있으면서도 외부와의 통신이 가능하게 된다.
- Increased security : NAT translation 으로 인해 level of indirection 이 증가하고 이를 통해 client device 에 대한 악의적 접근을 더욱 어렵게(접근 단계 증가) 만들 수 있다.  
- (mostly) Transparent : host 는 따로 바꿔줘야할 부분이 없이 사용할 수 있다.  

NAT 밑에(안에) NAT 가 또 있을 수도 있다.  

그러나 이런 NAT 의 막강한 장점들과 더불어 단점들도 존재한다.  

- Complexity  
  - header manipulation : 악의적으로 했는지 구분이 불가능하게 된다.
- Lack of public address  
  - 몇몇 application 은 public IP address 가 없어서 작동이 되지 않을 수도 있다.
- Security Protocol Issues  
  - IPsec 과 같이 detect header tampering 을 위한 protocol 들은 NAT 로 인한 변경점을 malicious datagram hacking 로 간주할 수도 있다.
- P2P communication difficulty  
  - IP 주소를 찾는게 어려워질 수 있어서 peer-to-peer 통신이 어려워질 수 있다.  

NAT 는 _Basic NAT_ 와 ___Network Address Port Translation(NAPT)___ 로 구분된다.  

### NAPT 

![napt operation](./image20.png)  

private network 안의 node 들은 동시에 외부 network 에 접근할 수 있도록 single registered IP address 를 NAPT 로부터 할당받을 수 있다.(여러 개의 IP 를 받을 수 있으나 보통 1개)

![napt operation](./image21.png)

![napt operation](./image22.png)

이런 NAPT 의 단점은 다음과 같다.

1. Performance
2. port-sensitive services(Talk, RealPlayer, ...) 는 special application layer gateway 를 쓰지 않는 이상 불가능하다.
3. packet reassmebly issues  
    - UDP-DATA 의 구조에서
    - IP-UDP-Part of DATA
    - IP Part of DATA ... 이런식으로 쪼개지게 되어 다시 하나의 packet 으로 만들어야만 packet 해석이 가능하게 된다.
4. P2P communication is very hard

### NAT Filtering Behaviour  

firewall 같은 기능을 하게 된다.

![nat filtering behaviour](./image23.png)

### Hairpinning Behaviour

이제 NAT 안에서 통신을 하는 경우에 대해 살펴보자.  

이 때, 서로 같은 네트워크에 있는지 모를 수 있기 때문에 서버에서 알려주게 된다.  

host A 가 10.1.1.1 IP address 를 쓰고 있는데 외부에서 볼 때는 5.5.5.1 로 보인다고 해보자.  
host B 는 10.1.1.2 IP address 를 쓰지만 외부에서는 5.5.5.2 로 보인다고 해보자.  

![hairpinning behaviour](./image24.png)

host A 가 {5.5.5.2:1001} 로 packet 을 보내고 {5.5.5.2:1001} 에서 받을 것이라고 예상한다고 하면, 위와 같이 Binding Table 과 패킷들이 변하는 것을 확인할 수 있다.  

![hairpinning behaviour](./image25.png)

B -> A 도 역시 source, dst 가 다 바뀌게 된다. 이는 같은 private 망에 있음에도 불구하고 정보들이 바뀌는 것인데 서로 transparent 하다는 것을 의미하게 된다.(속성)

이 때, Host A 가 {5.5.5.2:1001} 로 받게 되어 Accept 됨을 알 수 있다.

![hairpinning behaviour](./image26.png)

host A 의 external address (5.5.5.1) 와 host B 의 external address (5.5.5.2) 는 서로 다른 값을 나타내고 있는데 이 두 값은 같은 NAT 의 behaviour 에 의존하고 있는 것일 가능성이 크다.

![question internal source ip address and port](./image27.png)  

host A 가 {5.5.5.2:1001} 로 보내고 {5.5.5.2:1001} 에서 receive 할 것을 예상했으나 {10.1.1.2:5001} 로 받게 되었다. 이 경우에는 Discard 인가 Accept 인가?  

> Discard!  
> tcp 를 생각할 때 4 tuple 이 맞아야 받게 되는데 privae 을 쓰면 안 맞게 되기 때문에 바꿔주게 되는 것이다.

## Mobile IP

이제 유선망에서 탈피해 무선망의 경우를 생각해보자!<small>지금까지 고작 유선으로 이렇게 많은 것을 다뤘다니...^^</small>

무선 환경에서는 단말이 움직이는 경우가 빈번하다.  
단말이 이동하게 되는 경우 기존의 router 가 커버하고 있던 구역을 벗어나게 되면 더이상 signal 을 받지 못하고 해당 구역의 router 에 다시 연결이 되어야 통신이 가능할 것이다.(router 의 통신 범위는 유한하니까)  

그러나 이렇게 될 경우 기존 서버와 유지하던 통신의 connection 을 잃어버리게 되는 것이고 다시 connection 을 맺어야하는데 이는 앞서 보았던 Router Advertisement(RA) 를 받아야 하는 것이다. 그러나 이를 기다리는 시간이 필연적으로 생길 수 밖에 없고 그 동안은 통신을 할 수 없게 되는 것이다.  
그렇다고 해서 이전 connection 을 계속 유지하고 있으면 connection 은 살아있을지 몰라도 통신 자체는 되지 않게 된다. router 가 패킷을 보낼 수 있는 영역에서 벗어났기 때문! 결국 packet 의 손실이 일어나게 되는 것이다.

![ip routing model](./image28.png)

이를 해결하기 위해 나타난 것이 바로 _Mobile IP_ 되겠다.  

여기서 잠깐 IP 주소가 갖는 특징을 복습해보자.

1. identifier
2. locator

이 두 가지 속성을 동시에 가지고 있는데 이러한 동시성이 mobile 환경에서는 문제가 되었다.  
따라서 _Mobile IP_ 에서는 1, 2 를 분리한다.  

- 서버 입장에서는 1과 계속 통신을 하며 packet 을 보낸다.
- 1의 router 가 변경 router 에게 forwarding 을 하게 된다.  

이를 위해서는 두 개의 IP header 가 필요하게 된다.  

src|dst|src|dst
:--:|:--:|:--:|:--:
router|locator|server|original

![two tier ip addressing](./image29.png)

이것을 좀 더 전문적으로 말하면 two-tier IP addressing 이라고 부른다.  

- HoA(Home Address) - the original static IP address - 163.152.39.10
- CoA(Care-of Address) - the temporary IP address - 220.68.82.10

### Terminology

- Agent Discovery  
  - Agent Solicitation/Agent Advertisement(ICMP messages)
  - 자기가 이동한 곳의 FA 를 찾고 찾았으면 advertisement 를 해준다
- Registration
  - Registration Request/Registration Reply(UDP messages)

<br />

- MN  
  - listen for agent advertisement 하고 registration 을 시작해 준다.
- HA
  - 일반적으로 mobility services 를 구성(coordinate)하고 process 한다.
- FA
  - registration request 를 relay 하고 HA-MN 사이에서 reply, MN 에게 datagram 전송을 위해 decapsulate.

### Mobile IPv4 Operation

Mobile IPv4 가 어떻게 동작하는지 확인해보자.

![ipv4 operation](./image32.png)

![ipv4 operation](./image33.png)

![ipv4 operation](./image34.png)

![ipv4 operation](./image35.png)

![ipv4 operation](./image36.png)

![ipv4 operation](./image37.png)

![ipv4 operation](./image38.png)

![ipv4 operation](./image39.png)

### Mobile IPv4 Features

- Triangle Routing
  - CN -> HA -> MN, MN -> CN 의 삼각형 모양의 routing 을 진행하게 된다.
- DHCP 의 방식을 응용하며 tunneling 을 어떻게하느냐에 따라 달라진다.
- FA 는 방문한 MN 에 대한 visitor list 를 관리한다.  
  - 여기에는 HoA, Lay 2(MAC) address 등이 있다.
- Two CoA Modes
  - FA-CoA
    - MN 들은 CoA 를 FA 로부터 받는다.
    - 새 CoA 에 대한 중복은 없다.
  - Co-located CoA
    - DHCP-based CoA allocation
    - DHCP server 가 CoA 의 유일성을 보장해야한다.

![difference bet fa-coa and cl-coa](./image30.png)

![difference bet fa-coa and cl-coa](./image31.png)

위의 이미지를 보고 FA-CoA 와 Co-located CoA 의 차이를 확인해보자.  

FA-CoA 가  IPv4 주소의 양적 한계 때문에 좀 더 선호된다.

### Ingress Filtering

Ingress Filtering 이란 하나의 policy 를 말하는 것인데 router 의 packet filtering technique 중 하나이다.  
inbound traffic 내에서 source address spoofing 을 방지하고자 사용된다.  

_Ingress Filtering_ 은 source address 까지 보는 것.(보통은 dst 만 보고 어디로 forwarding 할지 생각함)  
네트워크 상 이상한 source address(외부 네트워크 등) 에서 발송된 패킷이 있다면 밖으로 내보내지 않는다.  

왜 Mobile IPv4 에서 이것이 나오는가?  
MN 이 HoA 로 만듦(connection 유지를 위해) -> 이걸 받은 foreigner agent 입장은 이게 과연 내가 만든 건가? MN 이 HoA 로 header 를 만들었는데 이상하다고 생각을 할 수 있게 된다.  
이렇게 하게 되면 통신이 안되게 하므로 이런 상황에서도 통신이 되게끔 하려면 source 주소를 HoA 로 하는게 아니라 CoA 로 만들어서 통신이 되게 하는 방법을 시도할 수 있다.
그런데 4쌍이 모두 일치해야 통신이 되는데 HoA -> CoA 로 되면 CN 의 transport layer 에서 받지 않을 것이다.  

따라서 헤더를 하나 더 만들어서 하게 된다.

![ingress filtering](./image40.png)

이 때 Reverse Tunneling 을 하게 되는데, HA 한테 굳이 보내는 이유는 무엇일까?  
FA 가 할 수도 있지만 outer header 를 벗기는 종착지가 HA 나 FA 에서 하는데 HA 를 지나갈 때 packet 을 까지 않기 때문에 outer header 의 종착지가 HA 가 되도록 설계를 한 것이다.  

여기서 그럼 퀴즈 하나를 보고 가자.  

> 이 패킷을 MN 이 만들었는데 왜 만들었나?  
> ingress filtering 을 FA 에서 돌리고 있는데 이를 피하기 위해서!  

### Mobile IPv6

Mobile IPv4 에서 가장 큰 단점은 triangle 부분이었다.  
이를 IPv6 에서는 해결하려 하였는데, 다음의 차이점이 있다.  

1. infinite address space
2. autoconfiguration service  
    - DHCP 써도 되지만 안쓰고 router 가 주기적으로 쏘는 ICMP(RA) 받으면 알아서 IPv6 주소를 만들게 된다.  
        - 64-bit prefix(from RA), 나머지 64-bit 는 MAC 주소를 EUI64 로 채움
3. efficient routing
4. built-in security

뭐 어쨌든 IPv4 의 단점들을 보완하는 부분으로 고안되었다.  

특히 FA 가 없는데, CoA 를 할당할 수 있는 Autoconfiguration 방식이 있기 때문에 나머지 두 개(HA, MN)로 한다.  

### Mobile IPv6 Operation

![mobile ipv6 operation](./image41.png)

이 때 할당받는 방식은 autoconfiguration 이다.

![mobile ipv6 operation](./image42.png)

내가 어느 subnet 에 위치하는지 알아야 하기 때문에 RA 를 쏘고 Router Solicitation 을 진행한다.

![mobile ipv6 operation](./image43.png)

MN 이 원래 속해있던 HA 에게 binding update 를 한다.

![mobile ipv6 operation](./image44.png)

Packet Tunneling 은 IPv4 와 비슷하다.  
  
이전에는 CN 이 무조건 HA 를 거쳐서 패킷을 전송했다.  
이는 어찌보면 비효율적인 방법인데, direct 하게 CN -> MN 으로 보낼 수는 없을까?  
그렇다면 우선 CN - MN 간의 key 가 필요할 것이다.  
그리고 MN 이 CN 에게 자신의 CoA 를 알려줘야 통신이 될 것이다.  

이 때 악의적인 목적으로 CoA 를 잘못 알려주게되면 패킷이 오지 않게 된다.  

![mobile ipv6 operation](./image45.png)

그렇기 때문에 MN 이 HoTI(Home Test Init), CoTI(Care-of Test Init) 을 보내게 된다.

이러한 direct 통신을 가능하게 하는 것이 _Return Routability_ 이다.  
바로 통신을 할 때 routing header option 을 쓰고 수신한 IP layer 에서 변환해서 올리도록 하게 한다.  
이 때 바로 보내면 보안 문제가 발생할 수 있으므로 HoTI 를 먼저 HA 를 거쳐서 CN 으로 보내고 그 다음 CoTI 를 CN 으로 바로 보내게 된다.

![mobile ipv6 operation](./image46.png)

이렇게 되면 초기에 비어있던 binding cache 에 추가로 알게된 값(CoA)을 작성할 수 있고, 이렇게 상호 검증이 되어야 binding cahce 에 저장하는 것이다.

![mobile ipv6 operation](./image47.png)

CN 입장에서는 MN 의 HoA 와 통신중이라 패킷이 src=CN주소, dst=MN의 CoA 로 헤더를 써서 보낸다면 어떻게 될까?  
이를 해결하기 위해 어떤 layer(IPv6) 가 CoA->HoA 로 바꿔주어야만 한다. 이 때 binding cache 를 이용하게 된다.  

이는 헤더 구성을 

src|dst|option
:---:|:---:|:---:
\ |\ |HoA

이런 식으로 HoA 를 넣게 되면 MN 까지는 가게 된다.  
CoA -> HoA 는 header option 을 근거로 바꾸게 되며 이렇게 IP layer 에서 바꿔서 올리게 될 경우 App. layer 에서는 HoA 와 통신하고 있다고 생각하게 된다.(transparent)  

![mobile ipv6 operation](./image48.png)

### Ingress filtering in MIPv6

![ingress filtering in mipv6](./image49.png)

Mobile IPv6 에서 Ingress Filtering 을 피하기 위해서는 Home Address Destination Option 이란 것을 사용한다.  

이는 CoA 로 처음에 전송되는 것을 수신자가 HoA 로 바꿔서 App. layer 로 올릴 수 있게끔 해준다.

IPv4 에서는 두 개를 붙여서 써서 header 20 byte 가 추가되는 식으로 해결했는데  
IPv6 에서는 원래 header 에 destination option 을 붙인 셈이다.

## IPsec

IP 는 처음에 너무 대충 만들어져서 정보가 별로 없다.  
그러나 세상이 복잡해지고 발전하면서 security 측면에서도 생각을 할 수 있어야 되게 되었고, 이로 인해 IPsec 이 만들어지게 되었다.  

IP 에서는 identifier 가 바로 나오기 때문에 IPsec header 안에 MAC 를 이용, 암호화와 integrity + authentication 을 챙길 수 있도록 한다.  

이러한 IPsec 은 IP layer 밑 정도에 위치하게 되는데, IP 기능이 끝난후 IPsec 을 불러 security(en/decryption) 를 담당하게 한다.  

![ipsec services](./image50.png)

IPsec 은 두 파트로 나뉜다.

1. key 를 나와 나의 목적지에 대해 생성(IKE, IKEv2)
2. en/decryption

![ipsec mode](./image52.png)

이 때, IPsec - IP Header - UDP Header - Data 로 packet 을 구성하고 UDP - Data 를 암호화하게 되면 어떻게 될까??  
이 방식은 IP 가 드러나서 위험하다. 그러나 IP header 도 암호화하면 아무도 forwarding 할 수가 없게 된다.  

따라서 IP header(new) - IPsec - IP header(old) - UDP header - Data 로 packet 을 구성, end-to-end 로 통신이 가능하도록 한다. 이렇게 하면 기존 IP header(old) 도 암호화해서 통신이 잘 이루어질 수 있게 된다.

이러한 IPsec 은 크게 3가지를 필요로 한다.  

- Diffie-Hellman key exchange
- Encryption
- Keyed hash algorithm

### IPsec Architecture

![ipsec architecture](./image51.png)

IPsec 은 위와 같은 구조를 가지게 된다.  
IPsec 의 동작은 security association(SA) 를 맺는다고도 표현할 수 있다.  
이 SA 가 맺어지면 SPD 에 기록하게 된다.

- AH(Authentication Header)  
  단순히 authentication + integrity 를 수행하는 역할을 한다.  
  ![ah](./image53.png)  
  SPI 는 상대와 secure 한 통신을 하려면 사전에 키를 주고 받아야하기 때문에 SA 를 맺었을 때 서로 다를 텐데 이를 SPI 라는 index 로 서로 통신하는지 판단하게 해주는 것이다.  
  Seq. No. 는 한 번에 하나씩 증가할 것이고, integrity 는 MAC 같은 것을 쓸 것이다.  
- ESP(Encapsulating Security Payload)  
  ![esp](./image54.png)  
  이 것을 달았을 경우 뒤의 data 가 암호화되게 된다.  
  ![esp](./image55.png)  
  대부분 AH 말고 ESP 를 사용한다.  
  
## IP Routing Protocol

routing protocol 은 간단하게 설명하면 network topology information 에 대해 routing table 을 작성하는 것이라고 볼 수 있다.  
이를 통해 나중에 forwarding 될 때 경로로 사용하도록 한다.  

핵심적으로는 hop-by-hop routing 이 진행되게 되고, 두 가지 알고리즘에 대해 학습해보자.  

### Distance Vector Algorithm

거리와 방향을 함께 가지고 있는 방법으로 예를 들면 R3 입장에서 R2->R1 으로 패킷을 전송하고 싶으면 나를 거칠 때 8만큼의 cost 가 필요하다고 알려주는 방식이다.  
이를 위해 routing table 에는 best distance 와 the route(next hop) 을 기록하고 있게 된다.  

간단한 예를 보면 다음과 같다.  

![distance vector algorithm example](./image56.png)

그런데 여기서 link cost 가 변경될 수도 있다.  

![distance vector algorithm link cost change](./image57.png)

위에서 볼 수 있듯이 지속적으로 table 이 수정되게 되는데, 이는 z->x 로 가상의 경로(cost=5)가 생긴줄 착각해서 나타나는 현상이다.  
이 현상의 근본적인 원인은 어디를 거쳐서 해당 cost 가 필요한지를 모르기 때문이다.(거리 + 방향만 있을 뿐)

실제 routing table 이 어떤식으로 작성되는지 아래의 예제를 통해 확인해보자.  

![distance vector algorithm example](./image58.png)

여기서 처음으로 router A 에 대해서는 다음과 같이 진행된다.

![distance vector algorithm example](./image59.png)

처음 router B 에 대해서 다음과 같이 진행된다.

![distance vector algorithm example](./image60.png)

처음 router C 에 대해서 다음과 같이 진행된다.

![distance vector algorithm example](./image61.png)

두 번째로 다시 router A 에 대해서 다음과 같이 추가적으로 진행된다.

![distance vector algorithm example](./image62.png)

쉽게 채워볼 수 있을 것이다.  

### Link-State Algorithm

_Distance Vector Algorithm_ 은 잠깐 보았던 경로 착각의 문제가 있었다.  
이를 해결할 수 있는 방법인 _Link-State Algorithm_ 에 대해 알아보자.  

이는 구조적으로 현재 노드에 대해서 tree 를 구성하게 되며, shortest path algorithm(e.g., Dijkstra's algorithm) 을 활용하게 된다.  

![link state algorithm example](./image63.png)

![link state algorithm example](./image64.png)

![link state algorithm example](./image65.png)

![link state algorithm example](./image66.png)

![link state algorithm example](./image67.png)

최종적으로 router A 에 대해서 위와 같은 tree 가 만들어지게 된다.

## TCP Connection

일전에 배웠던 TCP 를 잠깐 복습해보고 가자.  
TCP 는 transparently reliable 한 통신을 할 수 있도록 해주는 protocol 로, packet 손실, 중복, flow / congestion control 등의 이슈를 다른 쪽(application programmer)에서 신경쓰지 않도록 해준다.

통신은 socket 을 통해 이루어지는데, TCP 는 stream socket 을 이용한다.

여기서 Connection 이란 무엇을 의미할까?  
이는 application 의 socket 이 connected 된 상태를 의미하는데, 이것은 socket address 의 쌍(host, peer)으로 정의된다.  

![what is a connection](./image68.png)

다시 한 번 상기하자면, 서버에 대한 정보는 well-known 이라 쉽게 알 수 있는 정보이다.  

### Socket

통신에 쓰이는 socket 에 대해 간략히 알아보자.  
다음은 socket 과 관련된 함수들이다.

- Create  
  소켓의 형식에는 PF_INET(protocol family), AF_INET(address family) 두 가지가 있다. 이는 소켓이 디자인될 때 하나의 protocol 이 아니라 여러 protocol 을 지원할 수 있도록 하려했기 때문이다. 그러나 뭘 써도 상관은 없는 요즘이다.  
- Bind  
  서버측에서 port 를 random 하게 생성할텐데 client 가 아는걸로 나의 주소와 port 를 일치시켜줘야 client 입장에서 통신이 가능해진다. 이를 위한 것이 _Bind_ 작업이다.  
  Bind 함수를 이용해 특정 IP 와 포트 번호를 연결할 수 있게 된다.  
  이 때, 서버에서만 bind function 을 호출하게 되는데, client-server 통신 모델에서 항상 서버 프로그램이 먼저 수행되고 있어야 하고, 서버는 socket 을 호출하여 통신에 사용할 소켓을 하나 만들고 이때 return 된 소켓 번호와 자신의 소켓 주소를 bind 로 연결시키게 된다.  
  서버에서 bind 가 필요한 이유는 소켓 번호는 응용 프로그램이 알고 있는 통신 창구 번호이고, 소켓 주소는 네트워크 시스템(TCP/IP)이 알고 있는 주소이므로 이들의 관계를 묶어 두어야 응용 프로세스와 네트워크 시스템 간의 패킷 전달이 가능하기 때문이다.  
- Connect  
  이 함수는 client 가 서버에 연결을 요청할 때 사용하는데, 서버측에서는 이 요청을 listen 으로 기다리고 있다.  
- Accept  
  서버가 client 의 요청을 수락하게 된다.  
  서버는 접속 요청을 허락하고 client 와 통신하기 위해 커널이 자동으로 소켓을 생성하도록 한다.  
  이 수락 요청의 과정은 3-way handshake 로 진행된다.  

### Layout of TCP header

TCP 는 손실 없이 packet 을 받게 되고, 이를 reliable 한 통신이라고 하게 된다.  
이러한 reliable 한 통신을 위해 sequence number 와 acknowledgment number 가 필요하다.

![layout of tcp header](./image69.png)

TCP Header 는 위와 같이 구성되게 된다.  

- Sequence Number
  
  ![sequence number](./image70.png)

  통신을 하는데 있어서 sequence number 는 connection 때 미리 만들게 된다. 이 때 보통 추적을 피하기 위해 random 으로 생성하게 되고, 각각의 Initial Sequence Number(ISN) 을 유지하게 된다.  

- Acknowledgement Number

  ![acknowledgement number](./image71.png)

  ACK 는 receiver 가 어떤 packet 을 받은 다음에 받아야할 packet 을 알리는 것으로, packet 1,2,3,4 를 받았다면 ACK=5 를 sender 에게 보내게 된다.

- TCP Header Length

  ![tcp header length](./image72.png)

  TCP header length 는 IP header 에서 처럼 32-bit(4 bytes)로 표현이 되며 최솟값은 5이다. 이 경우에는 어떠한 TCP Option 도 존재하지 않는 경우이다.

- Control Flags
  - 0 : FIN = Terminate the connection
  - 1 : SYN = Synchronize sequence numbers
  - 2 : RST = Reset the connection
  - 3 : PSH = Push the data
  - 4 : ACK = The value in the acknowledgement field is valid
  - 5 : URG = The value in the urgent pointer field is valid  
    - 이는 남는 위치에 특정한 data 를 넣은 후 tcp 에 알려주는 것이다. 이렇게 되면 원래 data flow 와 다르게 되는데 급한 경우에 사용되는 것이다.

### 3-Way Handshake

이제 connection 을 연결하는 3-way handshake 에 대해서 간략하게 살펴보자.  

![3 way handshake](./image73.png)

1. client 는 server 에게 접속 요청 메시지(SYN)를 전송하고 SYN_SENT 상태
2. server 는 SYN 요청을 받고 client 에 요청을 수락(SYN+ACK) 하고 SYN_RECEIVED 상태
3. client 는 server 에게 수락 확인(ACK)을 보내고 server 는 ESTABLISHED 상태

아주 간단하다!

### 4-Way Handshake

connection 을 해제하는 과정도 살펴보자.

![4 way handshake](./image74.png)

<small>필기..ㅎㅎ</small>  

1. 최초에 client-server 모두 ESTABLISHED 상태
2. client 에서 FIN=1 을 보내고 소켓의 FIN_WAIT_1 상태로 변경
3. server 에서 FIN=1 을 받고 CLOSE_WAIT 상태로 변경, ACK 를 보내고 client 가 이를 수신하면 FIN_WAIT_2 상태로 변경
4. server 에서 연결 종료를 위해 FIN 을 보냄, LASK_ACK 상태로 변경
5. client 에서 FIN 을 받고 TIME_WAIT 상태로 돌입, ACK 를 전송
6. server 에서 마지막 ACK 를 받으면 CLOSED 상태로 연결 종료, 시간이 지나면 client 도 CLOSED 상태로 연결 종료

## Flow Control

Flow Control 은 쉽게 말해 router 는 멀쩡한데 end node 에서의 처리가 느려서 송신 속도를 줄이도록 하는 것이다.  
이는 TCP Sliding Window 로 조절되는데, 이 window size 만큼의 패킷을 한 번에 보낼 수 있게 된다. 이 사이즈가 조절되면 한 번에 전송되고 있는 패킷의 양이 바뀌게 되고, 이로 인해 flow control 이 동작하게 된다.  

### Sliding Window

이 _sliding window_ 는 sender 와 receiver 두 군데에서 모두 운용된다.  

![sliding window](./image75.png)

window size=7 인 상황에서 sender-receiver 가 각 패킷을 전송하면서 일어나는 window 의 변화를 위에서 확인해 볼 수 있다.

![example operation](./image76.png)

이번에는 실제 패킷을 보면서 확인해보자.  

- packet 181 : win=0, 즉 더 보낼 수 없다는 것이다.
- packet 182 : win=22656 으로 더 보낼 수 있게 되었다는 것이다.
- packet 183-197 : 더 보낼 수 있는 만큼 packet 을 쪼개서 전송하고 있다.  
  - 여기서 총 보내게 된 Bytes 는 20736B 이다. 따라서 남은 window size 는 1920B 이다.
  ![example operation](./image77.png)
- packet 198 : 여기서 Ack=4109684389 로 받은 것이 바로 처음 packet 183 을 받았다는 것을 의미하게 된다.(다음을 저부분서부터 원하는 것이니까) 이는 앞서 설명했듯이 client/server seq. no. 이 다르기 때문이다. 또한 win=22272 이므로 4109684389 에서 부터 22272B 만큼 전송을 할 수 있다는 의미다.
  ![exmample operation](./image78.png)
  
### Silly Window Syndrome

이러한 sliding window 기법을 이용할 때의 문제점이 무엇이 있을까?  

![silly window syndrome](./image79.png)

sender 입장에서 data 를 빠르게 생성할 경우에 위와 같이 하나씩 window 가 이동하는 모습을 볼 수 있게 된다. 즉 작은 packet 이 하나씩 전달되어 sliding window 가 의미가 퇴색되게 되는 경우이다.  

- Nagle's Algorithm  
  최대한 한 번에 많은 양을 보내도록 기다렸다가 보내는 방법이다.  
  이를 위해 Maximum Segement Size(MSS) 를 이용하고, MSS 보다 window 가 작으면 보내지 않거나, 너무 오래 window 를 기다리고 있으면(timer) 현재 window 크기 만큼 보내는 방법이다.  
  timer 는 서버가 보낸 data 가 있으면 언젠가 ACK 가 올 것이라고 가정하고 이를 시발점으로 생각해 시간을 재게 된다.  

  이 때, receiver 는 receive window size 가 충분히 크지 않다면 0 을 보내다가 MSS 이상일 경우 제대로된 window size 를 알려주는 방식으로 해 줄 수 있다.  

- Delayed Acknowledgement  
  receiver 가 packet 을 받고 나서 ACK 를 보내는데 여러 ACK 를 보내지 않고 효율적으로(적게) 보내는 방법이다. delayed ack 가 만료될 때까지 새로운 패킷을 받지 못하면 ACK 를 보내거나, 만료 전 새로운 packet 이 온다면 ACK 를 보내는 방식이다.  
  sender 측에서 Nagle's algorithm 을 쓰고 있을 경우 오래 기다리게 되는 경우가 생길 수 있다.  

### Retransmissions in TCP

ACK 가 timeout 되거나 multiple ACKs(3 ACKs) 를 받게 될 경우 _retransmission_ 이 일어나게 된다. 어딘가에 문제가 생겼다고 생각하기 때문!

![receiving duplicate acks](./image80.png)

위의 도식은 중복된 ACK 를 수신하는 경우이다.  
이처럼 3개의 같은 ACK 를 받게 되면 해당 SeqNo 부터의 패킷을 재전송 하게 된다.  
아마도 손실되거나 유실되었다고 생각을 하게 되기 때문!  

- Retransmission Timer :  
  적당한 기간의 timer 를 설정해서 ACK 가 안오는 경우에 다시 보내주도록 한다.  

  ![rto](./image81.png)
  round-trip time(RTT)를 이용해 시간을 측정하고 이를 바탕으로 RTO(retransmission timeout) 값을 정한다.  

  ![setting the rto value](./image82.png)
  RTO 를 정하는데 측정되는 RTT 값을 exponential moving average(EMA) 로 구해주게 된다.  

  이 때, retransmission 된 패킷의 경우에 RTT 값을 update 하지 않도록 해서 모호함을 제거하는 방식이 Karn's algorithm 이다.  

## Congestion Control

간단하게 _Congestion Control_ 에 대해 설명하면, 네트워크가 너무 혼잡해서 보내는 양을 줄여야할 때 필요한 것이다.  
<small>_Flow Control_ 과 주의해서 구분할 수 있도록 하자!</small>  

여기서는 _Congestion Window_ 가 이용되게 된다.  

![tcp congestion window](./image83.png)

위와 같이 생각을 할 수 있으며 network 의 상황을 보고 sender 가 알아서 바꾸게 된다.  


### TCP AIMD & Slow Start

congestion window 의 크기는 시간이 지날 수록 증가하게 되는데, drop 이 일어날 경우(timeout or 3 duplicated ACKs) window 의 크기를 반으로 줄이게 된다.  

timeout 이 일어날 경우 window 크기를 엄청 줄인다(=1)

TCP Slow Start 는 window size 가 지수적으로 증가하는 것을 의미한다.

### TCP Congestion Policy  

![tcp congestion policy](./image84.png)

policy 는 4가지가 있다.  

1. slow start : 최초에 지수적으로 증가
2. congestion avoidance : 천천히 window 가 커짐
3. fast retransmission : 3-ACK 가 발생했을 때 window 사이즈를 반으로 줄이고 재전송하게 되는 것이다.
4. fast recovery : 반으로 깎인 상태에서 다시 linear 하게 증가하는 모습을 의미한다.

위의 푸른색 영역이 slow start, 노란색 영역이 congestion avoidance 가 된다.  

### TCP Window Control - Tahoe / Reno

TCP congestion window 를 control 하는 방식은 여러 개가 있는데 그 중 2개를 살펴보자.  

- Tahoe :  
  - initial threshold 크게 잡음
  - timeout 이 발생하면 window 를 아주 작게(=1) 한다.
  - 3-ACK 의 경우 fast retransmit 을 실행한다. 이 때 threshold 는 절반으로 줄인다. window 를 아주 작게(=1)로 한 뒤 threshold 까지 slow start 이후 linear 하게 증가 시킨다.
  ![tahoe](./image85.png)
- Reno :  
  - 3-ACK 가 발생할 경우 fast retransmit 을 실행한다. window 를 반으로만 줄이고 linear 하게 증가
  - timeout 의 경우 window 크기를 아주 작게(=1) 하고 slow start 이후 fast recovery(linear 한 증가) 에 진입한다.
  ![reno](./image86.png)

<small>threshold 가 반으로 되는 기준점은 timeout 또는 3-ACK 가 발생한 시점의 window size 라고 보면 된다.</small>  
