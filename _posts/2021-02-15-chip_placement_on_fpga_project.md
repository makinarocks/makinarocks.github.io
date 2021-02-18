---
layout: post
title: Chip Placement on FPGA 프로젝트를 소개합니다!
author: kyeongmin woo
categories: [combinatorial_optimization, reinforcement_learning]
---

* 펩리스 하이퍼링크 추가
* ASIC에 대한 용어 설명, 하이퍼 링크

안녕하세요. MakinaRocks의 우경민입니다.

MakinaRocks의 COP(Combinatorial Optimization Problem) 팀에서는 지난 2020년 9월부터 2021년 1월까지 반도체 설계 공정 중 하나인 Placement & Routing에 강화학습을 적용하는 프로젝트를 진행했습니다. AI Chip 설계를 전문으로 하는 펩리스(fabless) 스타트업 Furiosa AI와의 협업으로 진행되었으며, Furiosa AI가 가지고 있는 반도체 설계 기술과 Makinarocks의 산업 AI 역량을 결합하여 상용 FPGA 자동화 도구와 비교해 효율적인 문제 해결의 가능성을 확인할 수 있었습니다. 

본 프로젝트는 지난 2020년 4월 Google에서 발표한 Chip Placement with Deep Reinforcement Learning[[1](#ref-1)] 논문(이하 Google의 Chip Placement 논문)에 기초를 두고 있으며, 논문에서 제시하는 문제 정의를 참고하였습니다. 다만 논문에서는 ASIC(Application-Specific Integrated Circuit)을 대상으로 하는 반면 COP 팀에서 진행한 프로젝트는 FPGA(Field-Programmable Gate Array)를 대상으로 하였다는 점에서 큰 차이가 있습니다.

## 문제 정의부터 살펴보기

우리가 일상적으로 사용하는 반도체는 모두 수천만 개 이상의 소자들로 구성되어 있으며, 각각의 소자들이 서로 전기적 신호를 주고 받도록 설계되어 있습니다. 이러한 반도체를 실제로 구현하기 위해서는 모든 소자들을 손톱 크기의 Chip Canvas에 배치하고 각각을 연결해주어야 합니다. 반도체 산업에서는 어떤 소자를 어디에 배치하고(Placement) 어떻게 연결할 것인지(Routing) 결정하는 공정을 Placement & Routing 공정, 줄여서 P&R 이라고 부릅니다.

COP 팀에서는 논리적으로 정의된 소자들의 연결 그래프를 입력으로 받아 개별 소자들의 물리적인 위치를 결정해주는 강화학습 에이전트를 만들어 P&R 공정, 그 중에서도 Placement 작업을 해결하고자 했습니다. Furiosa AI 측의 제안으로 FPGA를 사용하여 진행하게 되었고, 관련 프로젝트를 처음 진행하는 만큼 작은 크기의 반도체 설계부터 시작했습니다.

문제 정의부터 반도체 산업의 특성이 많이 녹아져 있는 만큼, 문제를 이해하고 해결하는 모든 과정에서 관련 도메인 지식을 많이 요구하는 프로젝트였습니다. 프로젝트 문제 정의에 대한 구체적인 설명에 앞서 FPGA는 바둑판이고 Netlist는 바둑돌이라는 느낌을 가지고 시작하시면 보다 쉽게 이해하실 수 있을 것 같습니다.

### FPGA란?

시스템 반도체의 일종인 FPGA는 Field Programmable Gate Array의 약자로, Programmable이라는 표현에서도 알 수 있듯이 사용자가 목적에 맞게 내부 로직을 변경할 수 있는 반도체를 말합니다. ASIC의 경우 내부 구조가 모두 결정되어 있기 때문에 정해진 목적대로만 사용할 수 있습니다. 반면 FPGA는 사용자가 자신이 원하는 대로 내부 로직을 조작하는 것이 가능합니다.

따라서 FPGA를 사용하게 되면 강화학습 에이전트가 배치한 결과를 실제 FPGA Board 상에서 구동이 가능한지 빠르게 확인할 수 있습니다. 반면 ASIC을 사용하면 실제 반도체를 제작하기 전까지는 실물을 대상으로 테스트를 진행하는 것이 불가능하다는 점에서 FPGA를 사용하게 되면 보다 효율적으로 검증 작업을 수행할 수 있습니다. 참고로 프로젝트에서 사용한 FPGA는 Xilinx 사의 U250 Board이고, 배치 결과를 분석하는 데에 사용한 프로그램은 동사에서 제작한 [Vivado](https://www.xilinx.com/support/university/vivado.html)를 사용했습니다.

#### Programmable의 의미는?

FPGA 배치 문제의 이해를 돕기위해 FPGA의 동작 방식에 대해 간략히 설명하도록 하겠습니다. FPGA가 Programmable 한 이유는 모든 소자가 미리 배치되어 있고, 전선들 또한 개별 소자들을 서로 연결할 수 있도록 미리 배치되어 있기 때문입니다. 즉 FPGA에서 프로그래밍한다는 것은 어떤 소자를 활성화시킬지 결정하는 작업이라고 할 수 있습니다. 아래 그림은 Vivado 상에서 FPGA Board의 일부분을 캡쳐한 것인데, 각각의 작은 네모 박스들이 색깔별로 서로 다른 소자(TILE)를 나타냅니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-15-chip-placement-on-fpga-project/chip_placement_on_fpga_chip_canvas.png" alt="normal gradient" width="80%">
  <figcaption style="text-align: center;">[그림] - FPGA Board</figcaption>
</p>
</figure>

이때 활성화의 단위 소자를 BEL이라고 부릅니다. 참고로 BEL은 FPGA를 구성하는 최소 단위이기도 한데, Xilinx FPGA는 다음과 같은 계층 구조로 되어 있습니다. 아래 이미지는 위의 이미지를 매우 크게 확대한 것으로, 자세한 내용은 Xilinx의 Rapid Wright 홈페이지[[3](#ref-3)]를 참고하시기 바랍니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-15-chip-placement-on-fpga-project/chip_placement_on_fpga_tile_site_bel.png" alt="normal gradient" width="80%">
  <figcaption style="text-align: center;">[그림] - TILE > SITE > BEL</figcaption>
</p>
</figure>

### 반도체 설계도, Netlist

Netlist는 반도체의 논리적인 설계도로서, 여기에는 반도체가 동작하려면 어떤 소자들이 필요하고, 각각의 소자들은 어떻게 연결되어 있는지 정의되어 있습니다. 이러한 Netlist에는 개별 소자들의 연결 관계만 담겨 있을 뿐 각각의 소자들의 위치 정보나 어떤 소자가 다른 소자와 얼마나 가까워야 하는지에 대한 정보는 포함되어 있지 않습니다. Chip Placement란 Netlist 설계도에 정의되어 있는 내용을 최적 동작이 가능하도록 Netlist를 구성하는 소자들의 위치를 결정하는 과정이라고 할 수 있습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-15-chip-placement-on-fpga-project/chip_placement_placement_and_routing.png" alt="normal gradient" width="=100%">
  <figcaption style="text-align: center;">[그림] - Placement & Routing</figcaption>
</p>
</figure>

#### Macro & Standard Cell

Netlist를 구성하는 소자들은 그 크기 및 기능에 따라 Macro와 Standard Cell로 구분합니다. 상대적으로 크기가 큰 Macro는 RAM[[4](#ref-4)], DSP[[5](#ref-5)] 등과 같이 그 자체만으로도 복잡한 기능을 수행할 수 있는 소자들입니다. 반면 Standard Cell은 LUT, Flip-Flop[[6](#ref-6)]과 같이 단순한 연산이나 순간적인 데이터 저장 등의 기능만을 가지고 있습니다. Google의 Chip Placement 논문[[1](#ref-1)]을 기준으로 하면 Macro만 강화학습 알고리즘으로 배치하고 그 이외의 소자들은 Clustering 하여 갯수를 줄이고 Cluster 단위로 전통적인 알고리즘(Force-Directed Method)[[7](#ref-7)]을 사용하여 배치합니다.


* 72개가 작은가? 왜 작은가? Google 논문 이야기가 전혀 없음 + 몇백 개 (few hundreds of macros) 이런 식으로 표현하기

#### 작은 문제 정의하기

COP 팀에서 첫 번째 문제로 확보한 Netlist는 소자의 개수가 72개로 Macro가 2개, Standard Cell이 70개 였습니다. 하지만 Google의 Chip Placement 논문[[1](#ref-1)]에 따라 진행한다면 강화학습 Agent로 단 2개의 Macro만을 배치하는 것이었기 때문에 Agent의 성능을 평가하기에는 너무 적다고 판단하였습니다. 따라서 본 프로젝트에서는 총 72개의 Macro와 Standard Cell 모두를 강화학습 Agent가 배치하도록 문제를 정의하여 실험을 진행했습니다.

## 강화학습 환경 만들기

프로젝트에서 다룬 문제에 대한 간략한 소개에 이어 개발 과정에 대해서도 소개해보려 합니다. 어떤 문제에 강화학습을 적용하기 위해 가장 먼저 해야 하는 작업은 에이전트가 학습할 수 있도록 적절한 강화학습 환경을 만드는 것입니다. 이때 환경으로 사용 가능한 시뮬레이터가 있다면 에이전트와 시뮬레이터를 연결하는 작업만 수행하면 되지만, 그렇지 못한 상황이라면 주어진 문제에 맞게 동작하는 환경을 직접 개발해야 합니다. 

FPGA와 관련해서는 Xilinx 사의 Vivado를 시뮬레이터로 사용할 수 있었습니다. 하지만 동작 속도가 느리고 에이전트와의 연결 과정에서 어려움이 예상되어 사용하지 않는 것으로 결론 내렸습니다. 대신 FPGA와 유사하게 동작하는 Python 프로그램을 개발하여 학습 환경으로 사용했습니다.

환경을 개발하는 과정에서 시작 단계에서는 예상치 못한 다양한 문제들을 경험할 수 있었습니다. Google의 Chip Placement 논문에서 모호하게 기술하고 있는 부분들을 실험과 추론을 통해 구체화하기도 하고, 안정적인 학습을 위한 속도 개선과 디버깅 작업에도 많은 시간을 소요했었습니다. 구체적으로 환경을 개발하면서 많이 고민한 이슈들로는 다음과 같은 것들이 있었습니다.

### (1) 어디에 배치할지 어떻게 결정할까

ASIC을 사용하는 Google의 Chip Placement 논문에서는 전체 Chip Canvas를 일정한 간격의 Grid로 나누고, 에이전트가 Action으로서 그 중 하나를 선택하도록 하고 있습니다. FPGA 또한 사용하고자 하는 Board에 동일한 방법으로 Grid를 적용할 수 있습니다. 다만 Chip Canvas 상에 다른 소자들과 겹치지만 않는다면 소자를 자유롭게 배치할 수 있는 ASIC과는 달리 FPGA는 각 소자의 타입에 따라 배치 가능한 위치가 미리 정해져 있다는 문제가 있었습니다.

이러한 FPGA의 본질적인 특성 때문에 Google의 Chip Placement 논문보다는 다소 복잡하게 환경을 구성하게 되었습니다. 우선 에이전트가 소자를 배치할 Grid Cell을 선택하면 해당 Grid Cell 내에 포함된 BEL 중 임의로 하나를 추출하여 소자와 매핑하도록 하였습니다. Action에 적용되는 Masking 또한 소자의 타입에 따라 다르게 적용했다는 점에서도 소자 간 구분이 없어 하나의 Mask만 사용하는 논문과는 차이가 있습니다.


* Grid를 어떻게 잡았는지, 한 셀 안에 몇 개 정도가 들어가는지

Grid의 Row, Column 갯수 또한 중요한 문제인데, Google의 Chip Placement 논문에서는 많게는 $$128 \times 128$$까지 실험을 진행했고, 결과적으로는 $$30 \times 30$$으로 설정했을 때에 성능이 가장 좋았다고 언급합니다. 그러나 본 프로젝트에서는 Netlist의 크기가 논문보다 작기 때문에 Grid의 크기 또한 작게하기로 결정했습니다. 최종적인 실험에는 $$6 \times 6$$ 크기의 Grid를 사용하였습니다.

### (2) Reward는 어떻게 계산할까

Reward Function은 강화학습에서 가장 중요한 것 요소 중 하나로, 학습의 방향을 결정합니다. Google의 Chip Placement 논문에서는 아래와 같이 Reward Function을 제안하고 있습니다.

* p, g 가 뭘까요? placement, graph

>$$
R_{p,q} = -\text{WireLength}(p, g) - \lambda \text{Congestion}(p, g) \\
\text{S.t. } \text{density}(p,g) \leq \text{max}_{\text{density}}
$$

반도체의 소자들을 연결하며 신호를 주고 받을 수 있게 하는 전선을 Wire라고 부릅니다. 이 Wire의 길이에 따라 반도체의 성능이 달라지고 경우에 따라서는 반도체가 정상적으로 동작하지 못하게 되기도 합니다. Wire length가 짧을수록 이점을 가지므로 Reward Function에서 Wire Length에 따라 페널티를 부여하고 있습니다.

Wire Length를 구하는 방법은 여러가지[[7](#ref-7)]가 있는데, Chip Placement 논문에서는 배치된 소자들의 2차원 위치 정보를 통해 HPWL(Half Perimeter Wire Length)[[7](#ref-7)] 방식에 따라 구하고 있습니다. 예시를 통해 확인하면 구현 내용을 보다 쉽게 이해할 수 있을 것 같아 네 개의 소자가 네 개의 Grid Cell에 나누어 배치된 예시 이미지를 준비했습니다. 

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-15-chip-placement-on-fpga-project/chip_placement_on_fpga_wirelength_calculation.png" alt="normal gradient" width="90%">
  <figcaption style="text-align: center;">[그림] - Wire Length Example</figcaption>
</p>
</figure>

하나씩 확인해보면 동일한 Grid Cell 내에 배치된 소자들을 서로 연결하는 빨간 Wire(1번-2번)의 HPWL는 0으로 계산되며, Grid Cell 두 개에 걸쳐 연결되는 녹색 Wire(1번-3번)는 1로 계산됩니다. 경우에 따라서는 노란 Wire(1,2,3,4번)처럼 하나의 Wire가 복수의 소자들과 연결되어 있기도 합니다. HPWL은 Wire와 연결된 지점을 모두 포함하는 최소 사각형을 먼저 그리고, 그 둘레의 절반으로 Wire Length를 추정하기 때문에 노란 Wire는 길이가 2로 계산됩니다.

Routing Congestion에 대해서는 반도체 설계와 관련된 지식이 요구되는 만큼 Furiosa AI의 도움을 받아 구현하게 되었습니다. 구체적으로는 각각의 Grid Cell를 통과하는 Wire의 갯수를 사용하여 수평 방향과 수직 방향 각각에 대한 congestion을 계산하고, 이 중 상위 10%에 대한 평균 값을 사용하는 방식입니다. 


* Placement Density 설명 추가하기

마지막으로 Placement Density는 전체 FPGA Board 상에서 배치 영역을 적절하게 조절하는 것으로 제약 조건을 만족하도록 했습니다.

### (3) 환경에서 어떤 정보를 주어야 할까

Google의 Chip Placement 논문에서 제시하는 Observation의 유형으로는 Macro Feature, Netlist Graph, Current Macro id, Netlist Metadata, Mask 등이 있습니다. 그런데 논문에서는 각각의 정보들이 어떻게 구성되어 있는지에 대해서는 간략하게 예시 수준으로만 기술하고 있습니다. 이러한 모호성을 해결하기 위해 Reward를 예측하는 데에 도움이 되는지에 따라 정보를 추가해나가는 방식으로 다양한 실험을 진행했습니다. 최종적으로는 다음과 같은 정보들을 Observation으로 에이전트에 전달하도록 했습니다.

- Macro Feature: 타입(One-Hot), 위치(Row & Col), 포트 갯수
- Netlist Graph: 인접 매트릭스
- Current Macro id: Macro Feature에서의 Index
- Netlist Metadata: 사용하는 영역의 크기, Grid의 크기, Congestion Map, 총 Macro/Wire의 갯수
- Mask: 타입별 마스킹 정보


* 포트 갯수에 대한 설명 추가하기 / Google에서는 안 쓰고 있음

FPGA이므로 사용하는 영역의 크기는 TILE의 갯수를 활용했습니다. 또한 Congestion Map은 Google의 Chip Placement 논문에 나오지 않는 요소로, 쉽게 말해 각 Grid Cell 단위로 배치되어 있는 Macro들의 총 Port 갯수에 대한 정보를 가지고 있습니다. 포트 갯수가 많을수록 해당 영역과 연결되어 있는 Wire의 갯수가 많을 것이라고 추정할 수 있는 만큼 Routing Congestion을 추정하는 데에 도움이 될 것이라 판단하여 추가하게 되었습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-15-chip-placement-on-fpga-project/chip_placement_on_fpga_congestion_map.png" alt="normal gradient" width="90%">
  <figcaption style="text-align: center;">[그림] - Congestion Map</figcaption>
</p>
</figure>

## 강화학습 에이전트 만들기

강화학습 에이전트는 환경으로부터 전달받은 Observation를 처리하는 State Encoder와 이렇게 처리된 정보를 바탕으로 Action을 결정하는 Policy 두 부분으로 나누어 개발했습니다.

### State Encoder

에이전트에서 가장 핵심적인 부분은 State를 적절하게 표현하여 강화학습 Policy가 쉽게 이해할 수 있도록 표현하는 부분이라고 생각합니다. 이를 위해서는 환경으로부터 받은 Observation을 적절하게 처리하여 State Representation으로 만들어주어야 합니다.

배치 대상이 되는 Netlist는 Node와 Edge로 구성되는 Graph 형태로 되어 있습니다. 따라서 좋은 State Representation을 확보하기 위해서는 Graph 데이터를 잘 처리할 수 있는 모델이 필요합니다. Google의 Chip Placement 논문에서 제시하는 구조는 다음과 같으며, 이를 수렴할 때까지 반복적으로 각 Embedding을 업데이트 했다고 말합니다.

>$$
\eqalign{
&e_{ij} = f c_1 (\text{concat}( f c_0(v_i)  \lvert f c_0(v_j) \lvert w^e_{ij}))\\
&v_i = \text{mean}_{j \in N(v_i)}(e_{ij})
}
$$

그런데 위의 수식을 기반으로 State Encoder를 구현하며 COP 팀에서는 다음 두 가지 의문들이 제기되었습니다.

#### 1. $$fc$$에 Activation Function이 필요한가

위의 수식에는 두 개의 $$fc$$가 나옵니다. 그런데 이와 관련하여 Google의 Chip Placement 논문에서는 Activation Function 유무에 대해서는 언급하지 않고 있습니다. 처음에 Fully Connected Network에 

* Fully Connected Network - 구글 논문 확인해보기 - 처음에는 가장 단순하게 Fully Connected Layer로 보았다. Affine 연산만으로 가정 -> 그런데 안 됨 -> Activation function을 추가하니 되었다.


단순히 Affine 연산을 의미하는 것으로 해석하여 Activation Function 없이 구현하였으나 Embedding이 수렴하지 않았고, 모든 $$fc$$의 출력에 Activation Function을 추가한 후에야 Embedding이 수렴을 확인할 수 있었습니다.

#### 2. 수렴 여부는 어떻게 결정할 것인가

Embedding이 수렴한다는 것은 결국 이전 Embedding과 현재 Embedding 간의 차이가 점차 줄어든다는 것을 의미합니다. 그런데 그 차이가 무한히 작아질 때까지 계속 연산을 반복하는 것은 비효율적이므로 그 차이가 일정 수준 이하로 떨어지면 수렴한 것으로 판단합니다.

* wiki frobenius norm 링크 추가하기

Google의 Chip Placement 논문에서 구체적인 방안은 제시하지 않고 있습니다. COP 팀에서는 두 Embedding의 차이를 계산하는 방법으로 **Frobenius Norm**을 사용했습니다. $$n \times m$$ 행렬 $$A$$의 Frobenius Norm은 아래와 같이 계산됩니다.

$$
\| A \|_\text{F} = \root \of {\Sigma_{i=1}^n \Sigma_{j=1}^m \vert a_{ij} \vert^2}
$$

업데이트 중단의 기준이 되는 Threshold의 크기를 결정하는 것 또한 중요한 문제였습니다. 이때 Threshold를 너무 높게 잡으면 충분히 수렴되지 않은 것이므로 Embedding의 정확성이 떨어지게 되는 반면, 너무 낮게 잡으면 연산량이 과도하게 많아지게 됩니다. 특히 PyTorch에서는 반복적으로 Network를 forwarding하면 계산 그래프가 누적되어 Memory를 과도하게 차지하도록 되어 있어, Threshold를 낮게 설정하고 실험하는 경우에는 Out of Memory Issue도 빈번하게 발생했습니다.

이러한 문제에 대처하기 위해 Max iteration의 크기를 hyper parameter로 추가하여 일정 횟수 이상 반복적으로 Embedding이 업데이트되지는 않도록 했습니다. 참고로 최종 실험은 Max iteration은 10으로, Threshold는 1-e7로 설정하고 진행했습니다.


* PPO 레퍼런스 추가하기

### 강화학습 알고리즘

강화학습 알고리즘으로는 Google의 Chip Placement 논문과 동일하게 PPO를 사용했습니다.

* Chip Placement 이미지 추가하기

아키텍쳐 구조는 원 논문을 따랏다.


## 평가는 무엇을 기준으로 하나?

반도체 산업에서는 P&R의 결과로 반도체 성능의 척도인 PPA(Performance, Power, Area)가 결정된다고 말합니다. 즉 개별 소자들을 어떻게 배치하느냐에 따라 각 소자들을 연결하는 Wire의 길이와 필요한 영역의 크기가 달라진다는 것입니다. 이러한 점에서 P&R 결과의 평가 척도로 PPA를 보여주는 수치들을 주로 사용합니다. COP 팀 또한 이러한 수치들을 기준으로 모델의 최종 성능을 평가했습니다. 프로젝트에서 사용한 모델 평가 지표들은 다음과 같습니다.

- WNS(Worst Negative Setup-time Slack): Clock Frequency와 관련된 지표
- WHS(Worst Negative Hold-time Slack): Clock Frequency와 관련된 지표
- DP(Dynamic Power): Power와 관련된 지표
- RU(Routing Utilization Ratio): 사용하는 Wire 길이와 관련된 지표

WNS와 WHS의 Time-Slack 이라는 것은 Clock Frequency를 지키는 데에 얼마나 많은 여유 시간이 있는지 나타내는 것입니다. 따라서 이것이 크면 클수록 보다 여유롭게 Clock Frequency를 유지할 수 있으며, 동시에 더 높은 Clock Frequency 또한 가능하다는 것을 의미합니다. 반대로 이것이 음수가 되면 현재 배치 결과로는 주어진 Clock Frequency 대로 구현하는 것이 불가능하다는 뜻입니다.


* 문장 흐름 다듬기

Dynamic Power는 전체 전력 소비량 중 배치된 결과로 인해 사용되는 전력량을 의미합니다. FPGA Board 자체가 반도체이기 때문에 배치가 전혀 이뤄지지 않은 상태에서도 전력을 일정량 소모하게 되는데, 이는 Static Power라고 합니다. 마지막으로 Routing Utilization은 전체 사용 가능한 Wire 중에서 얼마나 많은 Wire를 사용하는지를 나타내는 수치입니다. 정확하게는 Horizontal / Vertical 나누어서 구해지며, 실험에서는 두 값의 합으로 정의했습니다.

* Magnitude 설명 부분 살려내기

정리하자면 WNS, WHS는 크면 클수록 좋은 값, DP와 RU는 작으면 작을수록 좋은 값 입니다. PPA는 이러한 값들을 모두 종합적으로 반영한 것인 만큼 COP 팀에서는 Overall Score를 계산하여 각 배치들을 비교했습니다. 정확한 계산식은 아래와 같습니다.

> Overall Score = WNS + WHS - 100 * Routing Utilization - 100 * Dynamic Power


## 그래서 얼마나 잘했나?


COP 팀에서 개발한 강화학습 알고리즘의 성능을 평가하기 위해서는 Vivado 내에서 강화학습 알고리즘이 결정한 대로 배치하고, Routing을 수행한 후 FPGA에서 사용되는 평가 척도들을 계산해야 합니다. COP 팀에서는 강화학습 알고리즘의 배치 결과를 아래와 같은 txt 파일로 저장하고 이를 Vivado에 입력 제약 조건으로 전달하여 이러한 과정이 이뤄질 수 있도록 했습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-15-chip-placement-on-fpga-project/chip_placement_on_fpga_txt_result.png" alt="normal gradient" width="90%">
  <figcaption style="text-align: center;">[그림] - txt Result Example</figcaption>
</p>
</figure>

위 txt 파일에서 각각의 Row는 Netlist에서 배치 대상이 되는 소자의 이름과 FPGA Board 상에서 소자가 배치될 BEL를 매핑한 것이라고 할 수 있습니다.

> [NETLIST MACRO NAME] [FPGA SITE NAME] [FPGA BEL NAME]

최종적인 배치 결과는 다음과 같습니다.

| Place-name | Dyn-power | WNS | WHS | Route-V | Route-H | Overall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random Placement | 0.014 | -0.494 | 0.124 | 0.006291 | 0.003450 | -2.744234 |
| Vivado Placement | **0.01** | 0.079 | 0.026 | **0.001188** | 0.001569 | -1.1707 |
| RL Placement | **0.01** | **0.163** | **0.046** | 0.001301 | **0.001324** | -1.053514 |
ㄴ
Random Placement는 모든 소자의 위치를 임의로 선택한 배치 결과를 말하고, Vivado Placement는 Vivado에서 찾은 최적 배치 결과를 말합니다. RL Placement가 COP 팀에서 개발한 에이전트의 배치 결과입니다.

Vivado Placement의 Overall Score가 -1.1707인 반면 강화학습 에이전트로 배치한 결과는 -1.053514로 나왔습니다. 이는 Random Placement 결과를 0으로, Vivado Placement 결과를 1로 보았을 때 약 1.03에 해당하는 수치로 Vivado의 최적 배치와 비교해 볼 때 3% 정도 더 나은 배치 결과를 얻었다고 할 수 있습니다.

각 배치를 시각화하면 다음과 같습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-15-chip-placement-on-fpga-project/chip_placement_on_fpga_random_result.png" alt="normal gradient" width="90%">
  <figcaption style="text-align: center;">[그림] - Random result</figcaption>
</p>
</figure>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-15-chip-placement-on-fpga-project/chip_placement_on_fpga_vivado_result.png" alt="normal gradient" width="90%">
  <figcaption style="text-align: center;">[그림] - Vivado result</figcaption>
</p>
</figure>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-15-chip-placement-on-fpga-project/chip_placement_on_fpga_rl_result.png" alt="normal gradient" width="90%">
  <figcaption style="text-align: center;">[그림] - RL result</figcaption>
</p>
</figure>


* 가운데 초록색에 대한 설명 / 라우팅 선에 대한 설명 추가

RL Placement와 Random Placement의 배치 결과는 모두 제약조건으로 Vivado에 전달되므로 모두 주황색 사각형으로 표현되고 있습니다(일부 파란색 사각형은 입력 포트 등 에이전트의 배치 범위를 벗어난 것입니다). 참고로 파란색 사각형은 Vivado 내부 알고리즘에 따라 배치한 소자를, 주황색 사각형은 제약 조건에 따라 배치된 소자를 의미합니다. Placement 이후의 Routing 작업은 모두 Vivado의 Solution을 따랐습니다.


* 실험의 의의

- FPGA에서도 잘 되더라.
- 앞으로 더 잘할거다.
- 이산 최적화 문제를 앞으로 더 많이 풀어볼 것이다.

* 비즈니스 벨류 이야기 추가하기

* 마지막에 세 포스팅 링크 연결하기

## References

<a name="ref-1">[1]</a>  [Azalia Mirhoseini, Anna Goldie, Mustafa Yazgan, Joe Jiang, Ebrahim Songhori, Shen Wang, Young-Joon Lee, Eric Johnson, Omkar Pathak, Sungmin Bae, Azade Nazi, Jiwoo Pak, Andy Tong, Kavya Srinivasa, William Hang, Emre Tuncer, Anand Babu, Quoc V. Le, James Laudon, Richard Ho, Roger Carpenter, Jeff Dean, 2020, Chip Placement with Deep Reinforcement Learning.](https://arxiv.org/abs/2004.10746)

<a name="ref-2">[2]</a>  [Google AI, 2020, Chip Placement with Deep Reinforcement Learning, Google AI Blog.](https://ai.googleblog.com/2020/04/chip-design-with-deep-reinforcement.html)

<a name="ref-3">[3]</a>  [Rapid Wright, Xilinx Architecture Terminology.](https://www.rapidwright.io/docs/Xilinx_Architecture.html)

<a name="ref-4">[4]</a>  [Xilinx, 2020, UltraScale Architecture Memory Resources.](https://www.xilinx.com/support/documentation/user_guides/ug573-ultrascale-memory-resources.pdf)

<a name="ref-5">[5]</a>  [Xilinx, 2020, UltraScale Architecture DSP Slice.](https://www.xilinx.com/support/documentation/user_guides/ug579-ultrascale-dsp.pdf)

<a name="ref-6">[6]</a>  [Xilinx, 2020, UltraScale Architecture Configurable Logic Block.](https://www.xilinx.com/support/documentation/user_guides/ug574-ultrascale-clb.pdf)

<a name="ref-7">[7]</a>  [K. Shahookar & P. Mazumder, 1991, VLSI Cell Placement Techniques, ACM Computing Surveys.](http://users.eecs.northwestern.edu/~haizhou/357/p143-shahookar.pdf)


막스필드 Guide
<a name="ref-7">[7]</a>  [K. Shahookar & P. Mazumder, 1991, VLSI Cell Placement Techniques, ACM Computing Surveys.](http://users.eecs.northwestern.edu/~haizhou/357/p143-shahookar.pdf)