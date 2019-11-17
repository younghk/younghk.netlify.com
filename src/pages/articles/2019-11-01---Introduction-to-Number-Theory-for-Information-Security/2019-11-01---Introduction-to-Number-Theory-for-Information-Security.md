---
draft:     true
title:     "Introduction to Number Theory for Information Security"
date:      "2019-11-01T17:31:13"
layout:    post
path:      "/posts/Introduction-to-Number-Theory-for-Information-Security/"
category:  "Information Security"
tags: 
    - Information Security
    - Number Theory
---

정보보호와 암호학에 대해 공부를 하다보면 마주하게 되는 수학이 있다.  
여기서 정수론(Number Theory)에 대해 간략하게 맛보고 가보자.  

왜 정수론의 내용이 필요할까?  
아마 이런 이야기를 들어봤을 수도 있겠다.  

> 소수를 구할 수 있는 공식이 발견되면 암호체계가 붕괴될 것이다!  

<small>그러나 그런 일은 벌어지지 않았다고 한다...</small>  

위의 말이 틀렸다고 볼 수는 없다. 소수를 이용한 암호화 알고리즘들이 존재하기 때문이다.  

정수론. 각종 수에 대한 성질을 대상으로 하는 수학의 한 분야이며, 소수와 같은 것을 다루는 학문이다.  

수학이 조금 어려울 수 있으나, 여러 암호체계에서 이용되는 정수론의 기반 내용을 모르면 작동 방식이나 원리를 이해하기 어려우니 최선을 다해보자! ^_^

## Euclidean Algorithm

유클리드 알고리즘은 두 양의 정수에 대한 최대공약수(Greatest Common Divisor, GCD)를 구하는 알고리즘으로 gcd(60, 24) = 12 와 같은 결과를 의미하게 된다.  

여기서 gcd(0 0) = 0 으로 정의한다.  

만일, gcd(a, b) 에서 a와 b 가 서로소(relatively prime)라면, gcd(a, b) = 1 이 된다.  

이 알고리즘의 작동 방식은 매우 간단하다.  

$$
a = q_1b + r_1 \qquad \text{ 0 < } r_1 \text { < b } \\
b = q_2r1 + r_2 \qquad \text{ 0 < } r_2 \text { < } r_1 \\
r_1 = q_3r_2 + r_3 \qquad \text{ 0 < } r_3 \text { < } r_2  \\
... \\
r_{n-2} = q_nr_{n-1} + r_n \qquad \text{ 0 < } r_n \text{ < } r_{n-1} \\
r_{n-1} = q_{n+1}r_n + 0 \\
d = gcd(a, b) = r_n
$$

## Modular Arithmetic

## Groups, Rings, and Fields

## Finite Fields of the Form GF(p)

## Polynomial Arithmetic

## Finite Fields of the Form GF($2^n$)

## Prime Numbers

## Fermat's and Euler's Theorems

## Testing for Primality

## The Chinese Remainder Theorem

## Discrete Logarithms

