---
layout: post
title: 머신러닝에서 Software Testing이 필요한 이유
author: kyeongmin woo
categories: [software-testing]
image: 
---

Software Testing이란 작성한 코드가 원하는대로 정확히 동작하는지 검증하는 작업을 말합니다. Python에서는 PyTest와 같은 Test Tool을 사용하여 Test를 자동화할 수 있습니다. Test를 자동화하면 어떤 함수에 무슨 문제가 있는지 한 번에 확인 가능하면서, Testing 작업 자체가 간단해져 자주 확인하게 된다는 장점이 있습니다. 또한 Pre-Commit hook[[2](#ref-2)]으로 추가하여 Test를 통과한 코드만 [GitHub](<https://github.com/>)을 통해 공유하도록 하는 것도 가능한데, 이를 통해 전체적인 코드의 안정성이 높아지는 효과도 기대할 수 있습니다. 특히 이러한 장점들은 코드가 복잡하고 방대하거나 배포를 염두해두고 있는 상황이라면 더욱 명확해집니다.

## Software Testing이 필요한 이유

앞서 언급한 Test 자동화의 장점들은 코딩이 필요한 영역이라면 분야에 구애받지 않고 누릴 수 있는 장점들입니다. 머신러닝 분야 또한 코딩이 필요한 분야라는 점에서 Test Code를 도입하여 얻을 수 있는 이점들이 많다고 생각합니다. 구체적으로 사내에서 머신러닝 프로젝트를 진행하며 느낀 **Test Code 작성 및 Test 자동화의 장점**으로는 다음과 같은 것들이 있습니다.

- 디버깅이 쉬워진다.
- 리펙토링이 쉬워진다.
- 코드 이해가 쉬워진다.

각각에 대해 하나씩 확인해보도록 하겠습니다.

#### 디버깅이 쉬워진다!

머신러닝에서 Test Code가 중요하다고 생각하는 첫 번째 이유는 디버깅의 시간적 공간적 Search Space를 줄여준다는 것입니다. 코딩을 하다보면 예상치 못한 반작용(Side Effect)이 종종 발생합니다. 

- 함수 A를 A'으로 고쳤을 뿐인데, 그로 인해 다른 함수 B가 기존과 다르게 동작하는 경우

가 대표적입니다. 여기서 문제는 A'으로 고친 이후에 해당 함수의 동작만을 확인하고 넘어가기 쉽다는 점입니다. A에 대한 업데이트인 만큼 B, C, D 처럼 다른 함수를 하나씩 확인하지는 않게 되기 때문입니다. 특히 전체 프로그램을 동작시켰을 때 B가 정상 동작하는 것처럼 보인다면 문제가 있다는 것을 모르고 넘어가는 경우가 생길 수 있습니다.

예를 들어 아래 그림에서와 같이 `main` 브랜치에서 새로운 브랜치 `feature/refactor_step`를 따서 작업을 진행하는 상황을 가정해보려 합니다. 만약 Commit `C5`까지 작성을 완료한 후에 무언가 잘못되었다는 것을 알았다면 문제의 원인을 찾기 위해 `C1`부터 `C5`까지 브랜치에서 작업한 모든 내용들을 일일이 확인하고 그로 인한 반작용을 찾아내야 합니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-25-unittest_for_ml/testcode_search_space.png" alt="normal gradient" width="90%">
  <figcaption style="text-align: center;">[그림] - Branch Example</figcaption>
</p>
</figure>

만약 Test가 Pre-Commit hook[[2](#ref-2)]으로 설정되어 있어, 매 Commit 마다 전체 프로그램의 부분 부분이 정상 동작하는지 자동으로 검증했다면 `C2`가 생성되기 이전에 어떤 부분에 문제가 있는지 바로 알 수 있었을 것입니다. 이러한 점에서 Test를 자동화하면 시간적인 디버깅 Search Space가 줄어들게 됩니다.

또한 Test Code를 잘 작성해두면 디버깅의 공간적 Search Space가 줄어들게 됩니다. Test Code를 잘 작성해두었다면 적어도 Test가 적용된 부분들에 대해서는 검증하지 않거나, 적어도 검증의 우선순위를 낮춰 확인할 수 있기 때문입니다.

머신러닝 프로젝트를 진행하다보면 모델의 학습 성능이 기대 이하인 경우가 자주 발생하고, 그로인해 학습이 되지 않는 이유를 찾기 위한 디버깅 과정이 빈번하게 일어납니다. 예를 들어 강화학습 프로젝트를 진행하며 환경과 에이전트를 모두 직접 구현한 상황을 가정해 보겠습니다. 모든 것을 정확히 구현했다고 생각했지만 모델의 성능이 기대 수준에 크게 미치지 못한다면 다음과 같은 문제들을 생각해 볼 수 있습니다.

- Hyper Parameter Tuning이 필요하다.
- State Representation, Reward Function이 적절치 못하다.
- 구현이 잘못되어있다.

여기에 문제가 있다면 한 번 학습하고 평가하는 데에 오랜 시간이 소요되기 때문에 각각의 가정들에 대해 검증하려면 시간이 많이 필요하다는 점입니다. 이러한 점에서 Test를 자동화하여 구현에 대한 신뢰성을 높일 수 있다면 머신러닝 모델 자체에만 집중할 수 있게 됩니다.

#### 리펙토링이 쉬워진다!

Refactoring은 기존 코드의 기능은 그대로 유지하면서 코드의 Maintinability 혹은 Readability를 높이는 작업을 말합니다. 기능은 동일하더라도 코드만 두고 본다면 변경이 생긴 것이므로 Refactoring를 진행한 후에 기존과 동일하게 동작하는지 검증해야하는데, 모든 함수를 이렇게 검증하는 것은 쉽지 않습니다. 만약 Test Code를 미리 만들어두었다면 코드 변경 후에도 쉽고 빠르게 함수 단위 검증이 가능해집니다. 결과적으로 각각의 기능을 검증하는 Test Code를 잘 작성해두면 Refactoring에 소요되는 비용이 줄어들게 되어, 코드의 품질이 높아지는 효과를 기대할 수 있습니다.

#### 코드 이해가 쉬워진다!

마지막으로 Test Code를 잘 작성해두게 되면 다른 사람이 내가 작성한 코드를 이해하는 데에도 많은 도움이 됩니다. Test의 목적은 코드가 의도대로 동작하는지 확인하는 것인데, 이를 뒤집어 생각해보면 Test Code만 잘 확인하면 어떤 의도를 가지고 작성된 코드인지, 전체 소스 코드에서 어떤 기능을 담당하고 있는지 알 수 있습니다. 이러한 점에서 잘 쓰인 Test Code는 코드의 모호성을 줄여주고, 코드와 관련된 팀원들 간의 커뮤니케이션 비용을 줄여주는 효과를 만들어 냅니다.

#### 무엇이든 적당하게

여러 장점을 언급한 만큼 Test Code를 작성하고 이를 자동화했을 때 단점도 짚고 넘어가려 합니다. 대표적인 단점은 Test Code를 작성하는 것 자체가 부담스럽다는 것입니다. Test Code까지 생각하게 되면 사실상 작성하는 코드의 양이 2배가 되고, 그에 따라 체감하는 개발 속도 또한 느려져 답답하게 느껴지기도 합니다. 이러한 점 때문에 어느 정도 수준까지 Test를 진행할 것인가에 대해서는 개발자들 사이에서도 갑론을박이 있고 진행하고 있는 프로젝트의 요구사항, 작업 크기, 참여하는 개발자의 특성 등을 고려하여 적절한 수준으로 도입할 필요가 있습니다.

## 가장 기본적인 Software Testing, Unit Test

**Unit Test**란 Software Testing 방법론 중 하나로, 프로그램을 구성하는 개별 코드 조각들이 목적에 맞게 동작하는지 검증하는 것을 말합니다. 단순하게 생각하면 프로그램을 구성하는 함수들이 특정 입력에 대해 기대되는 값을 출력하는지 일일이 확인하는 작업입니다. 이러한 점에서 Unit Test는 런타임에 제 함수의 동작을 검증하다는 점에서 동적(Dynamic) 분석에 속하고, 모듈을 구성하는 함수를 단위로 한다는 점에서 가장 기초적인 수준의 Software Testing 중 하나라고 할 수 있습니다.

### 머신러닝에서 Unit Test

머신러닝에서는 구현한 모델에 대한 성능 검증이 필수적으로 요구되고, 기존의 방법론과 비교하여 약간이라도 높은 성능을 보이는 것이 중요하게 여겨집니다. 그런데 모델을 구현하고 실험 환경을 구축하는 과정에서 생각보다 Human Error가 빈번하게 발생합니다. 대표적으로 다음과 같은 경우들을 생각해 볼 수 있습니다.

- Test Set에 Train Set이 일부 섞여 들어가는 경우
- Train Set에 포함되어선 안 되는 레이블의 데이터가 포함되는 경우
- MinMax Scaling에서 Max 값이 잘못 설정되어 1 이상의 값이 나오는 경우
- 강화학습 환경을 업데이트해야 하는데, 일부 변수에 대해서는 처리가 이뤄지지 않는 경우

위의 예시들이 가지는 공통점 중 하나는 모델의 학습 과정 상에서는 어떠한 에러도 발생시키지 않는다는 것입니다. 이렇게 되면 성능 재현이 이뤄지지 않는다거나, 학습이 이뤄지지 못할 때에 그  원인을 찾는 것이 더욱 어려워지게 됩니다. Unit Test는 이와 같은 문제들을 관리하는 도구로서 머신러닝 실험 환경의 신뢰성을 높여줄 수 있습니다.

## Test Code를 작성하며 놓치기 쉬운 것들

효율적인 Unit Test를 위해서는 유연하게 접근할 필요가 있습니다. 여기서 말하는 유연함이란 상황에 따라 적절한 Test Code를 작성해야 한다는 것을 의미하며, 이는 모든 상황에 적용 가능한 정답은 없다는 뜻이기도 합니다. 하지만 좋은 Unit Test를 위해 Test Code 작성 과정에서 확인해봐야 하는 지점들은 존재한다고 생각합니다. 이와 관련하여 회사에서 프로젝트를 진행하며 받은 동료들의 피드백을 모아 Test Code를 작성할 때 고려하면 좋은 점들을 정리해보았습니다.

#### 1. Test의 답지는 직접 만들자

Unit Test는 함수가 어떤 입력에 대해 의도한대로 출력을 내보내는지 확인하는 방식으로 이뤄집니다. 따라서 입력 값과 매칭되는 정답이 존재하게 되는데, 이때 정답을 검증 대상 중 하나인 다른 함수로 계산하게 되면 정확한 Test가 이뤄지기 어렵습니다. 아래 예시는 환경 `env`를 `reset` 했을 때 특정한 `initial_obs`를 반환하는지 확인하는 Test Case입니다. 그런데 `env.reset()`의 반환값과 비교할 정답을 구하기 위해 동일한 객체의 `env.get_initial_observation()` Method를 사용하고 있습니다. 이렇게 되면 `env.get_initial_observation()`의 반환값이 틀리지 않으리라는 보장이 없으므로 정확한 검증이 이뤄질 수 없습니다. 이는 시험 결과를 채점하려고 하는데 문제를 푼 사람과 문제의 답지를 작성한 사람이 동일해지는 것과 같습니다.

```python
class TestEnv:

    def test_initial_obs_with_reset():
        """Initial observation which is the return of reset must always be 1."""
        env = CustomEnv()

        obs = env.reset()
        initial_obs = env.get_intial_observation()

        assert initial_obs == obs
```

대신 Method의 Output이나, Attribute값은 Hard Coding을 통해 매번 적절한 값이 나오는지 직접 비교하는 것을 추천합니다. 여기서 Hard Coding으로 모든 경우의 수를 작성하는 것이 부담되기도 하고 거부감이 드는 것도 사실입니다. 하지만 Hard Coding으로 답지를 구성하게 되면 앞서 언급한 것처럼 문제를 푸는 사람(`env` object)과 문제의 답지를 작성한 사람(test case)을 분리할 수 있어 검증의 정확도가 높아지는 효과가 있습니다. 뿐만 아니라 Test Code를 통해 데이터의 형태를 명시적으로 확인할 수 있어 이후 코드를 이해하는 데에 도움이 된다는 장점도 있습니다.

```python
class TestEnv:

    def test_initial_obs_with_reset():
        """Initial observation which is the return of reset must always be 1."""
        env = CustomEnv()

        obs = env.reset()
        initial_obs = self.initial_observation

        assert initial_obs == obs

    @property
    def initial_observation():
        """Define correct initial observation."""
        return 1
```

#### 2. 무엇이 문제인지 빠르게 확인할 수 있도록 해야 한다

Test의 목적은 현재 코드에서 작성자가 원하는 대로 동작하지 않는 부분을 찾아내는 것입니다. 따라서 Test 결과가 나왔을 때 어떤 부분이 문제인지 빠르고 정확하게 확인할 수 있도록 해야 합니다. 그런데 아래 예시와 같이 하나의 Test Case에서 여러 내용을 검증하도록 구성하면 Test에 실패했을 때 무엇이 문제인지 확인하려면 Error Message를 일일이 확인해야 합니다.

```python

def test_step_method():
    """Check next observation, reward, done which are the returns of step method."""
    env = CustomEnv()
    obs = env.reset()

    for step_idx in range(100):
        obs, reward, done, info = env.step(obs)

        assert obs == self.valid_obs[step_idx]
        assert reward == self.valid_reward[step_idx]
        assert done == self.valid_done[step_idx]
        assert info == self.valid_info[step_idx]
```

여기서 한 가지 더 비효율적인 부분이 있다면 obs에 대한 검증에 실패하면 reward, done, info 등에 대한 검증은 하지도 않고 Test Case가 종료된다는 점입니다. 이 경우 obs가 정확히 구해지도록 개선한 뒤 다시 reward, done, info를 검증해야 합니다. 만약 한 번에 네 경우에 대해 모두 검증이 이뤄지도록 한다면 한 번의 Test 만으로도 모든 문제를 알 수 있어 디버깅의 피로도가 줄어들게 됩니다.

구체적으로 Test Case를 작성히먄서 다음과 같은 점들을 고려하는 것이 이러한 비효율을 줄이는 데에 도움이 되었습니다.

- Test Case는 최대한 잘게 쪼개어주는 것이 좋습니다.
- Test Case의 이름에서 검증 대상이 잘 드러나도록 해야 합니다.
- Test Case의 특성에 따라 그룹화하는 것이 좋습니다.

위의 예시는 다음과 같이 네 개로 나눌 수 있습니다.
```python

def test_obs_of_step_method():
    """Check next observation which is the return of step method."""
    env = CustomEnv()
    obs = env.reset()

    for step_idx in range(100):
        obs, _, _, _ = env.step(obs)

        assert obs == self.valid_obs[step_idx]

def test_reward_of_step_method():
    """Check reward which is the return of step method."""
    env = CustomEnv()
    obs = env.reset()

    for step_idx in range(100):
        _, reward, _, _ = env.step(obs)

        assert reward == self.valid_reward[step_idx]

def test_done_of_step_method():
    """Check done which is the return of step method."""
    env = CustomEnv()
    obs = env.reset()

    for step_idx in range(100):
        _, _, done, _ = env.step(obs)

        assert done == self.valid_done[step_idx]

def test_info_of_step_method():
    """Check info which is the return of step method."""
    env = CustomEnv()
    obs = env.reset()

    for step_idx in range(100):
        _, _, _, info = env.step(obs)

        assert info == self.valid_info[step_idx]
```

#### 3. 가능한 경우의 수를 모두 생각해 보아야 한다

함수는 어떤 입력이 주어졌을 때 원하는 출력 값을 반환해야 합니다. 따라서 입출력을 검증하는 것이 필수적이라 할 수 있는데, 이와 관련하여 다음과 같은 점들을 확인해 볼 필요가 있습니다.

- 입력의 범위, 입력의 타입이 정해져 있다면 이에 대한 check가 필요합니다.
- 가능한 Edge Case를 list-up 하고 어디까지 검증할 것인지 결정해야 합니다.
- Class의 Method는 Attribute를 변경하기도 하는데, 이에 대해서도 반환값과 마찬가지로 검증이 필요합니다.

Edge Case란 입력값으로 특별한 처리가 필요한 값이 들어오는 경우를 의미합니다[[1](#ref-1)]. Edge Case를 검증한다는 것은 입력으로 0 부터 1 까지의 실수가 들어올 것으로 가정하는 함수가 있을 때, 0과 1처럼 경계에 있는 입력이나, -1, 100 처럼 경계 밖의 입력이 들어올 때에도 적절히 처리하는지 확인하는 것을 의미합니다. 

이러한 Edge를 선정할 때에는 함수의 가능한 State를 모두 나열해보는 것이 중요합니다. 예를 들어 다음과 같이 하나의 Method 내에 if 문에 총 3개 있다면 8개의 State가 존재하므로, 개별 State에 대한 Edge Case를 고려해 Test Code를 작성하는 것이 좋습니다.

```python
def it_has_eight_state(inp1: int, inp2: int, inp3: int) -> int:
    number = 0
    if inp1 > 0:
        number += 1
    if inp2 > 0:
        number += 1
    if inp3 > 0:
        number += 1
    return number
```

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-25-unittest_for_ml/eight_states.png" alt="normal gradient" width="50%">
  <figcaption style="text-align: center;">[그림] - States of it_has_eight_state Method</figcaption>
</p>
</figure>

모든 State, 모든 Edge Case를 검증하는 것이 Unit Test의 입장에서는 바람직한 일이지만 전체 프로젝트를 진행하는 입장에서는 비효율적일 수도 있습니다. 상황에 따라서는 Edge Case를 모두 나열해보고 발생하지 않을 것이라는 믿음이 있는 Edge Case에 대해서는 제외할 필요도 있습니다.

#### 4. Test Case마다 그 목적이 무엇인지 Docstring을 명확히 작성해야 한다

Test Case마다 Docstring으로 무엇을 검증하기 위한 것이며, 입력과 출력은 어떠해야 한다는 것을 명시해주어야 합니다. 대표적인 Activation Function인 ReLU를 직접 구현하고 이를 검증한다고 하면 다음과 같이 Docstring을 작성할 수 있습니다.

```python
def test_activation_relu(self, mean: float, std: float) -> None:
    """Check actiovation function ReLU is correct.

    CheckList
        - The return value of the relu(x) is 0 when x < 0
        - The return value of the relu(x) is x when x >= 0
    """
```

#### 5. Test Code도 유지 보수의 대상이다

Test Code 또한 유지 보수의 대상이며, 새로운 기능을 추가하거나 기존 기능에 변경이 생기면 그에 맞추어 업데이트 되어야 합니다. 따라서 항상 100% Coverage를 가지는 Test Code를 작성하는 것은 비효율적일 수 있으며 상황에 맞게 Test Code의 작성을 미루거나 작성하지 않는 것도 필요하다고 생각합니다. 경우에 따라서는 앞서 언급한 내용들에 대해서도 타협하는 것이 보다 효율적일 수도 있습니다.

## PyTest로 Unit Test 자동화하기

**PyTest**는 이름에서도 알 수 있듯이 Python에서 Unit Test를 위해 자주 사용되는 프레임워크 입니다. 설치 방법을 비롯한 기본적인 사용 방법은 [PyTest 홈페이지](<https://docs.pytest.org/en/stable/getting-started.html>)에서 확인하실 수 있습니다.

### PyTest가 Test Code를 인식하는 방법

PyTest가 사용자의 Test Code를 인식하도록 하기 위해서는 다음과 같은 Directory Convention에 따라 Test Code를 작성해야 합니다. 

- Test File들은 모두 testpaths 내에 위치해야 한다. 일반적으로 Test Paths는 `tests/`로 해야 합니다.
- testpaths 내에서는 디렉토리를 생성하여 Test File들을 분류할 수 있습니다. PyTest는 Test Paths에서 Recursive하게 탐색합니다.
- testpaths 내의 Test File의 이름은 모두 `test_*.py` 또는 `*_test.py` 꼴로 해야 합니다.

PyTest에서 제공하는 [디렉토리 구조의 예시](<https://docs.pytest.org/en/stable/example/pythoncollection.html>)는 다음과 같습니다.

>```
tests/
|-- example
|   |-- test_example_01.py
|   '-- test_example_02.py
|-- foobar
|   |-- test_foobar_01.py
|   '-- test_foobar_02.py
'-- hello
    '-- world
        |-- test_world_01.py
        '-- test_world_02.py
```

### Test Case를 작성하는 방법

개별 Test Case들은 Class의 Method 혹은 Function의 형태로 Test File 내에 정의됩니다. 이때 Class의 형태로 묶어서 정의하면 Class 이름에 따라 그루핑(Grouping)되는데, 구체적으로 다음 두 가지 장점을 가지게 됩니다.

- 설정값, initializing 등을 공유할 수 있어 Test Code 작성이 편리해집니다.
- Test 결과가 Class 이름으로 묶여서 표시되기 때문에 결과 확인이 편리해집니다.

Test Case들에 대해서도 Convention을 따라야 PyTest가 Test Case로 인식할 수 있습니다. 대표적으로 다음과 같은 규칙들이 있습니다.

- Test File 내에 정의된 모든 Test Case들은 Class의 Method 혹은 Function 형태로 정의되어야 하며, 이때 Class와 Method, Function의 이름은 모두 `test_*` 꼴이어야 합니다.
- Class의 이름은 `Test*.py`여야 합니다.
- Class를 사용한다면 생성자 `__init__()` Method를 만들지 않습니다.

관련하여 예시 코드는 TDD를 공부하며 작성한 Repository [PyTest 예시 Code](<https://github.com/enfow/test-driven-dev-python/blob/main/tests/test_ch1.py>)에서 확인하실 수 있습니다.

```python
# tests/test_ch1.py
class TestDollar:
    """Test Dollar"""

    def test_multiplication(self):
        """test code"""
        five = Dollar(5)
        five.times(2)
        assert 10 == five.amount
```

### Test를 조금 더 쉽게

Unit Test를 비롯하여 모든 Software Testing의 효과를 높이려면 빈번하게 Test를 진행할 수 있어야 합니다. PyTest를 사용하면 한 줄의 명령어로 원하는 모든 Test를 실행할 수 있다는 점에서 Unit Test에 대한 접근성을 높여주게 됩니다. 그런데 PyTest 또한 option 값을 넣어야 한다면 다소 복잡해질 수 있습니다. 이때 `Makefile`을 사용하면 보다 편리하게 Test를 수행하는 것이 가능해집니다.

```bash
# Makefile
utest:
	env PYTHONPATH=. pytest ./tests/ -s --verbose --ignore tests/test_example_01.py
```

위와 같은 내용을 담아 Project Directory에 Makefile을 정의해두면 `$ make utest`로 PyTest를 실행할 수 있습니다. 참고로 위 명령어는 `./tests` 디렉토리에서 Test Case들을 찾되 test_example_01.py에 정의된 Test Case는 포함하지 않는다는 의미를 가지고 있습니다. 추가적인 option 값들은 [PyTest 홈페이지](<https://docs.pytest.org/en/stable/reference.html#command-line-flags>)에서 확인할 수 있습니다. `env PYTHONPATH=.`는 때때로 PyTest가 디렉토리를 잘못 잡는 경우가 있어 방어적으로 추가한 것입니다.

## Reference

<a name="ref-1">[1]</a>  [Wikipedia, 2021, Edge case.](https://en.wikipedia.org/wiki/Edge_case#Software_engineering)

<a name="ref-2">[2]</a>  [pre-commit.](https://pre-commit.com/)