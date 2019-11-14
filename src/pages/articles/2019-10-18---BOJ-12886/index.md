---
draft: false
title : "[BOJ 12886] 돌 그룹"
date : "2019-10-18T03:08:00.1234"
layout: post
path: "/posts/boj-12886/"
category: "BOJ"
description: ""
tags :
  - Algorithm
---

문제 출처 : https://www.acmicpc.net/problem/12886

## 풀이

이 문제를 BFS 를 이용해 완전 탐색의 기법으로 풀어낼 수 있다.  

이 때, $500^3$ 을 배열로 잡을 수가 없어서 헤맬 수 있으나,  

$$
A+B+C = sum \\
C = sum-A-B
$$

이라는 간단한 수학을 이용하면 $500^2$ 의 공간을 가지고 탐색을 진행할 수 있다.  

물론 500 만큼만 잡으면 탐색을 하는데 조금 문제가 있을 수 있는데, 500보다 큰 범위로 배열을 설정해주어야 한다.  

X+X 와 Y-X 로 각각 변화하는 숫자에 있어서 최대 최소는 각각 0~1000 임을 생각할 수 있다.  

따라서 공간 복잡도는 $1000^2$ 이 된다.  

이것을 구현하는 방법은 다양할 수 있는데, 아래는 아주 대충 짠 코드이다.

```cpp
#include <iostream>
#include <queue>
#include <algorithm>

using namespace std;
typedef pair<int,int> pii;

const int MAX = 1010;

bool check[MAX][MAX];
queue<pii> q;
int main(){
    int A, B, C;
    cin >> A >> B >> C;
    int sum=A+B+C;
    if(sum%3){
        cout << 0;
        return 0;
    }
    if(A==B && B==C){
        cout << 1;
        return 0;
    }
    int M = sum/3;
    if(A>C){
        swap(A,C);
    }
    if(A>B){
        swap(A,B);
    }
    // A < B < C;
    check[A][B]=1;
    pii k;
    k = make_pair(A,B);
    q.push(k);
    check[A][C]=1;
    k = make_pair(A,C);
    q.push(k);
    check[B][C]=1;
    k = make_pair(B,C);
    q.push(k);
    while(!q.empty()){
        k=q.front();
        q.pop();
        A=k.first, B=k.second;
        C = sum-A-B;
        int na = A*2, nb=B-A;
        if(na>1000 || nb>1000 || C>1000) continue;
        if(na<=0 || nb<=0 || C<=0) continue;
        //cout << A << " " << B << " " << C << endl;
        pii nk;
        if(na>nb){
            if(!check[nb][na]){
                check[nb][na] = 1;
                nk = make_pair(nb,na);
            }
        }
        else {
            if(!check[na][nb]){
                check[na][nb] = 1;
                nk = make_pair(na,nb);
            }
        }
        q.push(nk);
        if(na<C){
            if(!check[na][C]){
                check[na][C] = 1;
                nk=make_pair(na,C);
            }
        }
        else{
            if(!check[C][na]){
                check[C][na]=1;
                nk=make_pair(C,na);
            }
        }
        q.push(nk);
        if(nb<C){
            if(!check[nb][C]){
                check[nb][C] = 1;
                nk=make_pair(nb,C);
            }
        }
        else{
            if(!check[C][nb]){
                check[C][nb]=1;
                nk=make_pair(C,nb);
            }
        }
        q.push(nk);
    }
    cout << check[M][M];
    return 0;
}
```

---

## 수학적 접근

그러나 이 문제는 이러한 완전탐색 기법을 적용하는 문제는 아니다.  

세 가지 수가 모두 같은 수가 되려면 어떤 상태인 것인가?  

바로 __평균__ 이다.  

즉, 세 수의 합이 3의 배수가 아니면 답은 자명하게도 0 이 나와야 한다.  

그렇다면 세 수의 합이 3의 배수이면 답은 1 인가?  

_그렇지는 않다._  

해당 문제를 보면 작은 수 X 는 X+X 가 되어 2X 가 되게 되고, 큰 수 Y 는 Y-X 가 된다.  

여기서 착안해서 다음과 같은 접근을 할 수 있다.  

1. 입력값의 합은 3의 배수여야한다.
2. 입력값들의 최대공약수로 합을 나눴을 때도 3의 배수여야 한다.(1의 계산은 여기서 합쳐진다.)
3. 2에서 계산된 값은 $2^k \times 3$ 꼴이어야 한다.

어떻게 이런 접근 방법이 나왔을까?  

핵심은 X+X=2X, Y-X 가 되면서 다음으로 진행된다는 것이다.  

즉, 한 step 마다 수식(또는 인수라고도 생각할 수 있다)에 소인수 2가 추가되는 것이고, 결국 최종적으로 세 수가 같아지는 평균 M에는 몇 개의 소인수 2가 있을 것이다. 즉 $2^k$ 형태가 되게 된다.  
조금 엄밀하게 말하자면, $M = 2^k * G$ 의 꼴이 될 것인데, 여기서의 $G$ 가 바로 최대공약수(gcd) 이다.  


```cpp
#include <iostream>

using namespace std;

int gcd(int x, int y){
    if(y==0) return x;
    x%=y;
    return gcd(y,x);
}

int cal_bit(int k){
    int ret=0;
    for(int i=0; i<32; i++){
        ret+=k%2;
        k=k>>1;
    }
    return ret;
}

int main(){
    int A, B, C;
    cin >> A >> B >> C;
    int q=(A+B+C)/(gcd(A, gcd(B,C)));
    cout << (q%3==0 && cal_bit(q/3)==1);
    return 0;
}
```

## 개선할 점

1. naive BFS 코드에서 모듈화하면 코드가 간결해진다. 리팩토링을 해보자.
2. 수학적 접근 방법에서 엄밀한 증명이 필요하다.
3. gcd 와 cal_bit 를 내장 함수(__gcd, __builtin_popcount)를 사용할 경우 코드가 간결해진다.  
