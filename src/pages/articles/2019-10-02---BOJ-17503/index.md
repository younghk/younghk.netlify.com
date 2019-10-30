---
title: "[BOJ 17503] 맥주 축제"
date: "2019-10-02T01:31:00.1234"
layout: post
path: "/posts/BOJ-17503/"
draft: false
category: "BOJ"
tags:
  - PS
  - Algorithm
  - C++
  - BOJ
description: "문제 출처 : https://www.acmicpc.net/problem/17503"
---

## [BOJ 17503] 맥주 축제

문제 출처 : https://www.acmicpc.net/problem/17503

## 풀이

문제를 보았을 때 
  
1. K의 범위가 2^31-1 까지이므로 Parametric Search로 답을 특정하는 방법
2. 우선순위 큐를 이용해서(또는 min-heap) 푸는 방법

2번을 생각해내는게 조금 핵심적인 문제라 생각하는데, __N개의 맥주를 무조건 먹어야__ 하고, __M보다 같거나 많이 먹어야 하기__ 때문에 현재까지 어떤 것을 선택해서 먹고 있는지를 기억해야하기 때문이다.

코드는 다음과 같다.

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;
typedef long long ll;

const int MAX = 2e5;

struct Beer{
    int v, c;

    bool operator<(const Beer &x)const{
        return{ c<x.c || (c==x.c && v<x.v)};
    }
};

priority_queue<int, vector<int>, greater<int> > pq;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M, K;
    vector<Beer> beer;
    int sum[MAX]={0,};

    cin >> N >> M >> K;
    for(int i=0; i<K; i++){
        Beer k;
        cin >> k.v >> k.c;
        beer.push_back(k);
    }
    sort(beer.begin(),beer.end());
    sum[0]=beer[0].v;
    pq.push(beer[0].v);
    int ans=-1;
    for(int i=1; i<K; i++){
        int k = 0;
        pq.push(beer[i].v);
        if(i>=N){
            k = pq.top();
            pq.pop();
        }
        sum[i]=sum[i-1]-k+beer[i].v;
        if(sum[i]>=M && i>=N-1){
            ans=beer[i].c;
            break;
        }
    }
    
    cout << ans;
    
    return 0;
}
```

## 조심해야할 부분
1. 우선순위 큐에서 빼고 현재 value를 넣을 때, 현재 value가 큐에서 나온 값보다 작을 경우
2. N개의 맥주를 먹기 전에 M이상의 선호도를 채울 경우
  
    
      
  
## 코드 개선점
1. sum이 배열일 필요는 없다.