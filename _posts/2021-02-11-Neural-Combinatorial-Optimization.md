---
layout: post
title: Neural Combinatorial Optimization with Reinforcement Learning
author: jinwoo park
categories: [combinatorial_optimization, reinforcement_learning]
image: assets/images/2021-02-11-Neural-Combinatorial-Optimization/cover.png
---

Neural Combinatorial Optimization은 딥러닝을 사용하여 조합최적화문제(Combinatorial Optimization Problem)를 풀고자 하는 연구분야입니다. 이번 포스팅에서는 그 중에서도 조합최적화문제의 풀이에 강화학습을 사용한 방법론을 제안한 대표적인 연구[[1]](#ref-1)를 소개하려고 합니다.

## Combinatorial Optimization Problem

조합최적화문제란 유한한 탐색공간(search space)에서 최적의 해를 찾는 문제이며, 그 탐색공간은 보통 이산적(discrete)으로 표현할 수 있습니다. 대표적인 문제로는 [순회 세일즈맨 문제 (Traveling Salesman Problem)](https://en.wikipedia.org/wiki/Travelling_salesman_problem), [작업공정 스케줄링 (Job Ship Scheduling)](https://en.wikipedia.org/wiki/Job_shop_scheduling), [배낭 문제 (Knapsack Problem)](https://en.wikipedia.org/wiki/Knapsack_problem) 등이 여기에 해당하며, 많은 조합최적화문제들이 [NP-Hard](https://en.wikipedia.org/wiki/NP-hardness) 군에 속하는 것으로 알려져 있습니다. 이 중 "순회 세일즈맨 문제"에 대해 좀 더 자세히 살펴보도록 하겠습니다.

#### Traveling Saleman Problem (TSP)

순회 세일즈맨 문제(이하 TSP)는 여행거리의 총합이 최소화되도록 전체 노드의 순회순서를 결정하는 문제입니다. 아래 그림처럼 노드의 순회순서를 결정함에 따라 전체 여행거리의 총합은 천차만별로 달라질 수 있습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-11-Neural-Combinatorial-Optimization/tsp.png" alt="tsp solutions comparison">
  <figcaption style="text-align: center;">[그림1] 주어진 노드에 대한 두 개의 솔루션 비교 [2]</figcaption>
</p>
</figure>

이는 N개 지점에 대한 모든 순열(permutations)을 탐색하는 문제로, [brute-force search](https://en.wikipedia.org/wiki/Brute-force_search)의 경우 $O(N!)$, [dynamic programming](https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm)의 경우 $O(N^2 2^N)$의 계산복잡도를 보이는 NP-Hard 문제입니다 [[3]](#ref-3). 보통 많은 개수의 노드에 대한 솔루션을 구해야 할 때는 적절한 휴리스틱(Heuristic)을 사용하여 탐색공간을 줄이는 방식으로 계산 효율을 높이곤 합니다 [[4]](#ref-4). 하지만 휴리스틱을 사용하는 경우 문제의 세부사항이 변경되면 휴리스틱 또한 적절히 수정해야 하는 번거로움이 발생합니다. 2016년 말, 이 문제의식에 의거한 연구의 성과가 Google Brain의 연구진들로부터 공개되었습니다.

## Neural Combinatorial Optimization with Reinforcement Learning (2016)

Neural Combinatorial Optimization with Reinforcement Learning[[1]](#ref-1)의 저자들은 별도의 heuristic의 정의 없이도 2D Euclidean graphs로 표현된 (최대 100개 노드의) TSP를 푸는 새로운 방법을 제안합니다. 딥러닝을 사용하여 TSP 문제의 학습이 가능함을 보였던 Pointer Network[[7]](#ref-7)가 지닌 지도학습(supervised learning)의 한계점을 강화학습을 통해 개선하려는 것이 주요한 아이디어라 할 수 있습니다. 이러한 접근은 강화학습으로 Neural Architecture Search라는 이산문제를 풀었던 이전 연구경험[[5]](#ref-5)에서 기인한 것으로 보여집니다.

> "We empirically demonstrate that, even when using optimal solutions as labeled data to optimize a supervised mapping, the generalization is rather poor compared to an RL agent that explores different tours and observes their corresponding rewards."

#### Pointer Network

이 논문에서는 Pointer Network[[7]](#ref-7)의 기본구조를 그대로 따릅니다. Sequence-to-Sequence 모델이 정해진 정해진 N개 노드에 대한 문제에서만 동작할 수 있는 것에 비해 Pointer Network는 임의 개수의 노드에 대해서도 동작할 수 있는 것이 특징입니다. 즉, 5개~20개 노드의 TSP를 학습한 뒤에 학습데이터에 존재하지 않는 25~50개 노드의 TSP에 대해서도 동작 가능한 구조입니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-11-Neural-Combinatorial-Optimization/pointer_network_1.png" alt="Squence-to-Sequence vs Pointer Network">
  <figcaption style="text-align: center;">[그림2] Sequence-to-Sequence vs Pointer Network [8]</figcaption>
</p>
</figure>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-11-Neural-Combinatorial-Optimization/pointer_network_2.png" alt="Squence-to-Sequence vs Pointer Network">
  <figcaption style="text-align: center;">[수식1] Sequence-to-Sequence vs Pointer Network [8]</figcaption>
</p>
</figure>

먼저 Sequence-to-Sequence를 살펴보겠습니다. Sequence-to-Sequence는 전체 입력(각 노드의 이차원 좌표)에 대한 attention mask($a^i$)로 가중평균(weighted average)한 벡터($d_i^{\prime}$)를 디코더의 hidden state($d_i$)와 결합(concatenation)하여 예측에 사용하고 또한 다음 스텝의 입력으로 넣어주는 구조입니다. 이 구조는 고정된 크기의 출력을 내보내기 때문에 출력(예측해야 하는 카테고리)의 크기가 가변적인 경우에는 사용하기에 적합하지 않습니다.

반면, Pointer Network는 Sequence-to-Sequence의 attention mask를 예측에 바로 사용합니다. Attention mask의 차원이 입력의 개수에 따른다는 속성을 이용해 같은 학습파라미터의 차원을 가지고도 가변적인 개수의 TSP에 대해 동작하게 할 수 있습니다.

#### Policy Gradient (REINFORCE)

Policy는 강화학습 에이전트의 행동방식을 정의합니다. 이는 주어진 상태로부터 어떤 행동(action)을 결정하는 상태-행동의 매핑함수라고도 할 수 있습니다. Policy Gradient는 주어진 문제에서 에이전트가 받는 보상의 기댓값을 최대화 하도록 policy를 직접적으로 업데이트하는 방법들을 통칭하며, 이 중에서도 REINFORCE는 Monte-Carlo method를 통해 얻은 샘플 에피소드로 추정한 리턴값을 이용해 policy를 업데이트하는 방법입니다. (Policy Grandient와 REINFORCE에 대한 자세한 설명은 [Lilian Weng의 블로그 포스팅](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)을 참고해주세요.)

본 논문에서는 모든 여행거리의 총 합에 대한 기댓값을 목적함수로 정의하고 이를 최소화시키는 것으로 문제를 정의합니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-11-Neural-Combinatorial-Optimization/nco_objective.png" alt="l2 distance for objective function">
</p>
</figure>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-11-Neural-Combinatorial-Optimization/nco_form_2.png" alt="l2 distance for objective function">
</p>
</figure>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-11-Neural-Combinatorial-Optimization/nco_form_3.png" alt="l2 distance for objective function">
  <figcaption style="text-align: center;">[수식2] 여행거리 총합의 기댓값을 J로 정의 [7]</figcaption>
</p>
</figure>

또한 REINFORCE 알고리즘으로 목적함수의 gradient를 표현하고 이를 Monte Carlo sampling 형태로 근사합니다. 더불어 총 여행거리의 기댓값을 예측하는 critic으로 baseline 함수 $b(s)$를 정의합니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-11-Neural-Combinatorial-Optimization/nco_form_4.png" alt="l2 distance for objective function">
</p>
</figure>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-11-Neural-Combinatorial-Optimization/nco_form_4_2.png" alt="l2 distance for objective function">
</p>
</figure>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-11-Neural-Combinatorial-Optimization/nco_form_5.png" alt="l2 distance for objective function">
  <figcaption style="text-align: center;">[수식3] 목적함수의 gradient 정의 [7]</figcaption>
</p>
</figure>

#### Experimental Results

저자는 다음과 같은 네 가지의 실험 설정을 제안합니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-11-Neural-Combinatorial-Optimization/experiment_config.png" alt="4 experimental configs">
</p>
  <figcaption style="text-align: center;">[그림3] 4 가지 실험 설정 [7]</figcaption>
</figure>

각각이 의미하는 바는 다음과 같습니다.

* RL pretraining-Greedy: RL 에이전트가 추정하는 가장 좋은 action을 선택
* Sampling: Inference에서 의도적으로 다양한 샘플 경로를 획득하고 그 중 가장 좋은 경로를 고르는 방법
* Active Search (AS): Test 시에 얻은 여러 샘플 경로들에 대해 더 작은 loss를 갖게끔 policy를 개선

결과적으로 20~100개 노드의 TSP에 대해 Optimal과 유사한 경로를 획득할 수 있었습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-11-Neural-Combinatorial-Optimization/nco_result.png" alt="TSP50 / TSP100 experimental results">
</p>
  <figcaption style="text-align: center;">[그림4] TSP50 (위) / TSP100 (아래) [7]</figcaption>
</figure>

## Coming Up Next..

다음 포스팅에서는 강화학습을 사용한 Neural Combinatorial Optimization 방법을 실제 산업 문제(Chip Placement Problem)에 적용한 사례에 대해 알아보도록 하겠습니다.


## References

<a name="ref-1">[1]</a>  [I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio, “Neural combinatorial optimization with reinforcement learning,” 2016.](https://arxiv.org/abs/1611.09940)

<a name="ref-2">[2]</a>  [DocP’s Channel, “Travelling Salesman Problem (TSP): Direct sampling vs simulated annealing in Python,” 2017.](https://youtu.be/2iBR8v2i0pM)

<a name="ref-3">[3]</a>  [WikiPedia, “Travelling Salesman Problem,” 12 Feb. 2021.](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Exact_algorithms)

<a name="ref-4">[4]</a>  [David L Applegate, Robert E Bixby, Vasek Chvatal, and William J Cook. "The traveling salesman problem: a computational study," Princeton university press, 2011.](https://www.jstor.org/stable/j.ctt7s8xg)

<a name="ref-5">[5]</a>  [Barret Zoph and Quoc Le. "Neural architecture search with reinforcement learning," arXiv preprint arXiv:1611.01578, 2016.](https://arxiv.org/abs/1611.01578)

<a name="ref-6">[6]</a>  [Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks," In Advances in Neural Information Processing Systems, pp. 3104–3112, 2014.](https://dl.acm.org/doi/10.5555/2969033.2969173)

<a name="ref-7">[7]</a>  [Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. "Pointer networks," In Advances in Neural Information Processing Systems, pp. 2692–2700, 2015b.](https://proceedings.neurips.cc/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Paper.pdf)