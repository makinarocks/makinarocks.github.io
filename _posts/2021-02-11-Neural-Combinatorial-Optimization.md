---
layout: post
title: Neural Combinatorial Optimization
author: jinwoo park
categories: [combinatorial_optimization, reinforcement_learning]
image: assets/images/2021-02-11-Neural-Combinatorial-Optimization/cover.png
---

Neural Combinatorial Optimization은 딥러닝을 사용하여 조합최적화문제(Combinatorial Optimization Problem)를 풀고자 하는 연구분야입니다. 이번 포스팅에서는 그 중에서도 강화학습을 이용하여 조합최적화문제를 푸는 대표적인 연구[[1]](#ref-1)와, 이 기술을 산업에서의 실제 문제에 적용하여 인상적인 성능을 보인 사례[[2]](#ref-2)를 차례대로 소개하려고 합니다.

## Combinatorial Optimization Problem

조합최적화문제란 유한한 탐색공간(search space)에서 최적의 해를 찾는 문제이며, 그 탐색공간은 보통 이산적(discrete)으로 표현할 수 있습니다. 대표적인 문제로는 [순회 세일즈맨 문제 (Traveling Salesman Problem)](https://en.wikipedia.org/wiki/Travelling_salesman_problem), [작업공정 스케줄링 (Job Ship Scheduling)](https://en.wikipedia.org/wiki/Job_shop_scheduling), [배낭 문제 (Knapsack Problem)](https://en.wikipedia.org/wiki/Knapsack_problem) 등이 여기에 해당하며, 많은 조합최적화문제들이 [NP-Hard](https://en.wikipedia.org/wiki/NP-hardness) 군에 속하는 것으로 알려져 있습니다. 이 중 "순회 세일즈맨 문제"에 대해 좀 더 자세히 살펴보도록 하겠습니다.

#### Traveling Saleman Problem (TSP)

순회 세일즈맨 문제(이하 TSP)는 여행거리의 총합이 최소화되도록 전체 노드의 순회순서를 결정하는 문제입니다. 아래 그림처럼 노드의 순회순서를 결정함에 따라 전체 여행거리의 총합은 천차만별로 달라질 수 있습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-11-Neural-Combinatorial-Optimization/tsp.png" alt="tsp solutions comparison">
  <figcaption style="text-align: center;">[그림1] 주어진 노드에 대한 두 개의 솔루션 비교 [3]</figcaption>
</p>
</figure>

이는 N개 지점에 대한 모든 순열(permutations)을 탐색하는 문제로, [brute-force search](https://en.wikipedia.org/wiki/Brute-force_search)의 경우 $O(N!)$, [dynamic programming](https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm)의 경우 $O(N^2 2^N)$의 계산복잡도를 보이는 NP-Hard 문제입니다 [[4]](#ref-4). 보통 많은 개수의 노드에 대한 솔루션을 구해야 할 때는 적절한 휴리스틱(Heuristic)을 사용하여 탐색공간을 줄이는 방식으로 계산의 효율을 높이곤 합니다 [[5]](#ref-5). 하지만 휴리스틱을 사용하는 경우 문제의 세부사항이 변경되면 휴리스틱 또한 적절히 수정해야 하는 번거로움이 있습니다. 2016년 말, 이 문제의식에 의거한 연구의 성과가 Google Brain의 연구진들로부터 공개됩니다.

## Neural Combinatorial Optimization with Reinforcement Learning (2016)

Neural Combinatorial Optimization with Reinforcement Learning[[1]](#ref-1)의 저자들은 별도의 heuristic의 정의 없이도 2D Euclidean graphs로 표현된 (최대 100개 노드의) TSP를 푸는 새로운 방법을 제안합니다. 딥러닝을 사용하여 TSP 문제의 학습이 가능함을 보였던 Pointer Network[[8]](#ref-8)가 지닌 지도학습(supervised learning)의 한계점을 강화학습을 통해 개선하려는 것이 주요한 아이디어라 할 수 있습니다. 이러한 접근은 강화학습으로 Neural Architecture Search라는 이산문제를 풀었던 이전 연구경험[[6]](#ref-6)에서 기인한 것으로 보여집니다.

> "We empirically demonstrate that, even when using optimal solutions as labeled data to optimize a supervised mapping, the generalization is rather poor compared to an RL agent that explores different tours and observes their corresponding rewards."

#### Pointer Network

이 논문에서는 Pointer Network[[8]](#ref-8)의 기본구조를 그대로 따릅니다. Sequence-to-Sequence 모델이 정해진 정해진 N개 노드에 대한 문제에서만 동작할 수 있는 것에 비해 Pointer Network는 임의 개수의 노드에 대해서도 동작할 수 있는 것이 특징입니다. 즉, 5개~20개 노드의 TSP를 학습한 뒤에 학습데이터에 존재하지 않는 25~50개 노드의 TSP에 대해서도 동작 가능한 구조입니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-11-Neural-Combinatorial-Optimization/pointer_network_1.png" alt="Squence-to-Sequence vs Pointer Network">
</p>
</figure>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-11-Neural-Combinatorial-Optimization/pointer_network_2.png" alt="Squence-to-Sequence vs Pointer Network">
  <figcaption style="text-align: center;">[그림2] Sequence-to-Sequence vs Pointer Network [8]</figcaption>
</p>
</figure>

먼저 Sequence-to-Sequence를 살펴보겠습니다. Sequence-to-Sequence는 전체 입력(각 노드의 이차원 좌표)에 대한 attention mask($a^i$)로 가중평균(weighted average)한 벡터($d_i^{\prime}$)를 디코더의 hidden state($d_i$)와 결합(concatenation)하여 예측에 사용하고 또한 다음 스텝의 입력으로 넣어주는 구조입니다. 이 구조는 고정된 크기의 출력을 내보내기 때문에 출력(예측해야 하는 카테고리)의 크기가 가변적인 경우에는 사용하기에 적합하지 않습니다.

반면, Pointer Network는 Sequence-to-Sequence의 attention mask를 예측에 바로 사용합니다. Attention mask의 차원이 입력의 개수에 따른다는 속성을 이용해 같은 학습파라미터의 차원을 가지고도 가변적인 개수의 TSP에 대해 동작하게 할 수 있습니다.

#### Policy Gradient (Reinforce)

Tbd

## Device placement Optimization with Reinforcement Learning (2017)

tbd

## Coming Up Next..

tbd

## References

<a name="ref-1">[1]</a>  [I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio, “Neural combinatorial optimization with reinforcement learning,” 2016.](https://arxiv.org/abs/1611.09940)

<a name="ref-2">[2]</a>  [Azalia Mirhoseini, Hieu Pham, Quoc V. Le, Benoit Steiner, Rasmus Larsen, Yuefeng Zhou, Naveen Kumar, Mohammad Norouzi, Samy Bengio, Jeff Dean,  “Device Placement Optimization with Reinforcement Learning,’ Proceedings of the 34th International Conference on Machine Learning, PMLR 70:2430-2439, 2017.](http://proceedings.mlr.press/v70/mirhoseini17a.html)

<a name="ref-3">[3]</a>  [DocP’s Channel, “Travelling Salesman Problem (TSP): Direct sampling vs simulated annealing in Python,” 2017.](https://youtu.be/2iBR8v2i0pM)

<a name="ref-4">[4]</a>  [WikiPedia, “Travelling Salesman Problem,” 12 Feb. 2021.](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Exact_algorithms)

<a name="ref-5">[5]</a>  [David L Applegate, Robert E Bixby, Vasek Chvatal, and William J Cook. "The traveling salesman problem: a computational study," Princeton university press, 2011.](https://www.jstor.org/stable/j.ctt7s8xg)

<a name="ref-6">[6]</a>  [Barret Zoph and Quoc Le. "Neural architecture search with reinforcement learning," arXiv preprint arXiv:1611.01578, 2016.](https://arxiv.org/abs/1611.01578)

<a name="ref-7">[7]</a>  [Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks," In Advances in Neural Information Processing Systems, pp. 3104–3112, 2014.](https://dl.acm.org/doi/10.5555/2969033.2969173)

<a name="ref-8">[8]</a>  [Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. "Pointer networks," In Advances in Neural Information Processing Systems, pp. 2692–2700, 2015b.](https://proceedings.neurips.cc/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Paper.pdf)