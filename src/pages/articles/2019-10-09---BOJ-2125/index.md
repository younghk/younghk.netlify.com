---
title: "[BOJ 2125] Mothy"
date: "2019-10-09T02:11:00.1234"
draft: false
layout: post
path: "/posts/boj-2125/"
category: "BOJ"
tags:
  - PS
  - Algorithm
  - C++
  - BOJ
description: ""
---

## [BOJ 2125] Mothy

문제 출처 : https://www.acmicpc.net/problem/2125

## 풀이

기하문제이다. 기하문제는 정말이지 즐겁다^^  
  
기하문제이기 때문에 기본적으로 CCW를 사용하게 되는데, 문제의 조건에 따라

1. 볼록다각형의 테두리는 지나갈 수 있음
2. 그러므로 다각형이 맞닿아 있는 곳도 지나갈 수 있음

을 생각해서 CCW를 구현해 주어야 한다.  
  
뿐만 아니라, 소수점까지 계산해야하는 문제이기에, 정수 또는 소수 처리를 잘 해주어야 한다.  

의식의 흐름대로 풀이를 정리하면 아래와 같다.  

1. 모든 볼록다각형의 정점(_convex_), 시작점과 끝점을 포함한 모든 다각형의 모든 정점(_node_)를 기억한다.
2. 각 정점들이 볼록다각형(_convex_)안에 존재(_isIn_)하는지 미리 판단한다. 볼록다각형에 존재한다면 해당 정점은 향후에 계산해 줄 필요가 없다. 갈 수 없는 정점이므로.
3. 다각형 밖에 있는 정점(_out_conv_, _i_)인 정점들을 대상으로 다른 다각형 밖에 있는 정점(_j_)들로 직선 거리로 갈 수 있는지 계산(_isCross_)한다.
4. 이 때, _i_ 에서 _j_ 의 중점 역시 다각형 안에 존재하는지 확인(_isIn_)한다.
5. 3과 4의 과정을 모두 통과한다면 _i_ 에서 _j_ 의 직선 거리를 계산해서 기억한다.
6. 마음에 드는 최단 경로 알고리즘으로 시작점에서 끝점까지의 최단 경로를 계산한다.

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <algorithm>
#define min2(x,y) (x<y?x:y)
#define max2(x,y) (x>y?x:y)

using namespace std;
typedef long long ll;
typedef pair<double, int> pdi;

const int MAXN = 310;

struct Point {
	int x, y;
	Point operator- (const Point a)const {
		return{ x - a.x, y - a.y };
	}
	Point operator+ (const Point a)const {
		return{ x + a.x, y + a.y };
	}
} s, e;

Point operator* (const Point &a, const ll b) {
	return{ a.x*b,a.y*b };

}
vector<Point> convex[MAXN], node;
double d[MAXN];
priority_queue<pdi, vector<pdi>, greater<pdi> > pq;
bool out_conv[MAXN];
vector<pdi> dist[MAXN];

ll outer(Point A, Point B) {
	ll t = A.x*B.y - A.y*B.x;
	return t < 0 ? -1 : t>0;
}

ll CCW(Point A, Point B, Point C) {
	return outer(B - A, C - A);
}

bool isCross(Point A, Point B, Point C, Point D) {
	ll abc = CCW(A, B, C);
	ll abd = CCW(A, B, D);
	ll cda = CCW(C, D, A);
	ll cdb = CCW(C, D, B);
	ll ab = abc*abd;
	ll cd = cda*cdb;
	if (ab < 0 && cd < 0) return 1;
	if (cd < 0 && (abc^abd)) return 1;
	return 0;
}

ll square(ll x, ll y) {
	return x*x + y*y;
}

bool isIn(Point M, int idx) {
	ll ccw = CCW(convex[idx].back() * 2, convex[idx][0] * 2, M);
	for (int i = 0; i < convex[idx].size() - 1; i++) {
		if (ccw != CCW(convex[idx][i] * 2, convex[idx][i + 1] * 2, M)) return 0;
	}
	return 1;
}

int main() {
	ios::sync_with_stdio(false);
	cin.tie(NULL);
    
	int N;
	cin >> N >> s.x >> s.y >> e.x >> e.y;
	node.push_back(s);
	for (int i = 1; i <= N; i++) {
		int M;
		cin >> M;
		for (int j = 0; j < M; j++) {
			Point K;
			cin >> K.x >> K.y;
			convex[i].push_back(K);
			node.push_back(K);
		}
	}
	node.push_back(e);
	for (int i = 0; i < node.size(); i++) {
		bool flag = true;
		for (int k = 1; k <= N && flag; k++) {
			flag &= !isIn(node[i] * 2, k);
		}
		out_conv[i] = flag;
	}
	for (int i = 0; i < node.size(); i++) {
		if (!out_conv[i]) continue;
		for (int j = i + 1; j < node.size(); j++) {
			if (!out_conv[j]) continue;
			bool flag = true;
			for (int k = 1; k <= N && flag; k++) {
				for (int l = 0; l < convex[k].size() - 1 && flag; l++) {
					flag &= !isCross(node[i], node[j], convex[k][l], convex[k][l + 1]);
				}
				flag &= !isIn(node[i] + node[j], k);
				flag &= !isCross(node[i], node[j], convex[k].back(), convex[k][0]);
			}
			if (!flag) continue;

			double dista = sqrt(square(node[i].x - node[j].x, node[i].y - node[j].y));
			dist[i].push_back({ dista,j });
			dist[j].push_back({ dista,i });
		}
	}
	for (int i = 1; i < node.size(); i++) d[i] = 1e9;
	d[0] = 0;
	pq.push({ 0,0 });
	while (!pq.empty()) {
		double val = pq.top().first;
		int here = pq.top().second;
		pq.pop();
		if (d[here] < val) continue;
		for (pdi k : dist[here]) {
			if (d[k.second] > d[here] + k.first) {
				d[k.second] = d[here] + k.first;
				pq.push({ d[k.second],k.second });
			}
		}
	}
	if (d[node.size() - 1] >= 1e9) cout << -1;
	else {
		cout.precision(5);
		cout << fixed << d[node.size() - 1];
	}
	return 0;
}
```

## 조심해야할 부분
1. 정점이 볼록 다각형 내부에 위치하는지 판별을 정확히 해주어야한다.
2. 맞닿은 부분 및 다각형의 모서리 위를 지나갈 수 있기에 CCW를 잘 구현해야한다.

## 개선할 점
  
잔 실수를 줄여보자.