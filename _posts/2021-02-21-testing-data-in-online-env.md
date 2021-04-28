---
layout: post
title: 실시간 데이터 검증하기
author: minjoo lee
categories: [deeplearning]
image: assets/images/2021-02-21-data_is_tested/total.gif
---

마키나락스는 제조업에서 실시간으로 생산 장비와 공정의 고장 및 이상을 사전에 예측하는 이상탐지 시스템을 제공하고 있습니다.
이상탐지 시스템을 적용하고자 하는 제조 생산 현장은 제품 및 공정의 변화가 잦은 곳이 많습니다.
공정이 변하면 데이터 분포가 달라지기 때문에, 사전에 학습된 모델이 적용하려는 시점에는 정상 작동하기 어렵습니다.
이런 문제를 해결하기 위해 학습과 추론이 동시에 가능한 코드 형태로 모델을 배포하고, 배포된 동안 새롭게 수집한 데이터를 이용해 모델을 학습합니다.
학습된 모델이 등록되면 실시간으로 새롭게 입력되는 데이터에 대해 추론합니다.

실시간 추론 서비스를 제공하는 모델은 배포된 코드와 함께 현장에서 수집한 학습 데이터로 완성됩니다 [[1]](#ref-1).
모델이 안정적으로 학습되고 추론하기 위해서는 코드뿐만 아니라 데이터에 대해서 유효성 테스트가 필요합니다.
이번 포스트에서는 실시간으로 입력돼 학습과 추론에 사용되는 데이터의 유효성을 확인할 수 있는 방법에 대해 소개드리겠습니다.

## Why data need test?

예상과 다르게 입력된 데이터로 모델을 학습, 추론한 경우 의도와 다른 결과를 출력할 수 있습니다.
예를 들어, Numeric 데이터 입력을 받는 연산 코드에서 Boolean 데이터 `False`가 입력됐을 경우를 생각해보겠습니다.
입력된 데이터의 Type이 다르지만 Python은 `False`를 0으로 변환해 연산한 결과를 출력합니다.
이 경우 코드가 작동하는데 문제 없기 때문에, 나중에 출력 결과를 통해 디버깅하는 것은 어렵습니다.
이와 같은 문제는 데이터 유효성 테스트를 통해 미리 방지할 수 있습니다.

이번 포스트에서는 Dataset과 함께 구성하는 Sample과 Feature에 적용할 수 있는 유효성 검증 방법을 소개하겠습니다.
1. Input Sample
2. Input Feature
3. Input Dataset

{% assign i = 1 %}
<div class="row">
    <div style="width:45%; float:left; margin-right:10px;">
        <figure class="image" style="align: center;">
            <p align="center">
                <img src="/assets/images/2021-02-21-data_is_tested/valid.gif" alt="valid-data" width="120%">
                <figcaption style="text-align: center;">[그림{{ i }}] Input Valid Data</figcaption>
            </p>
        </figure>
    </div>
    {% assign i = i | plus: 1 %}
    <div style="width:45%; float:right;">
        <figure class="image" style="align: center;">
            <p align="center">
                <img src="/assets/images/2021-02-21-data_is_tested/invalid.gif" alt="invalid-data" width="120%">
                <figcaption style="text-align: center;">[그림{{ i }}] Input Invalid Data</figcaption>
            </p>
        </figure>
    </div>
    {% assign i = i | plus: 1 %}
</div>

## 1. Input Sample

현실적인 예시를 추가, 더 재밋을 것 같은 예를 들어 사람이 개입되어 어떻게 되는

실시간으로 입력되는 데이터 1개를 Sample(Row)이라 하고, Sample이 모여 Dataset이 됩니다.
약속한 구조와 자료형의 데이터 Sample이 입력되는지 지속적으로 확인하는 과정이 필요합니다.

<figure class="image" style="align: center;">
    <p align="center">
        <img src="/assets/images/2021-02-21-data_is_tested/sample.png" alt="" width="50%">
        <figcaption style="text-align: center;">[그림{{ i }}] Sample</figcaption>
    </p>
</figure>
{% assign i = i | plus: 1 %}

Sample의 `Feature A`와 `Feature B` 속성을 이용해 새로운 `Feature C`를 만들어 내는 Feature Engineering 코드가 있습니다. Numeric 데이터 `Feature A`와 `Feature B`에 대해 곱하기 연산을 하여 `Feature C`를 만든다고 가정해봅시다.

```python
def make_feature_c(sample):
    feature_a = sample["feature_a"]
    feature_b = sample["feature_b"]
    return feature_a * feature_b
```

실시간으로 데이터가 입력되는 상황에서 `Feature A` 또는 `Feature B`가 정상적으로 들어오지 않는 경우를 생각해 보겠습니다.
예를 들어, Feature에 Numeric 대신 Boolean 값이 들어온 상황입니다.

```python
>>> data = {
    "feature_a": False,
    "feature_b": 4,
}
>>> feature_c = make_feature_c(data)
>>> feature_c
0
```

위의 예시에서는 Engineering 할 Feature에 Boolean 값이 들어오면서 예상치 못한 `Feature C`가 만들어지게 됩니다.
잘 못 입력된 `Feature A`가 0이라는 의도하지 않은 결과를 만들고 다음 프로세스까지 영향을 미치게 됩니다.
기본적으로 데이터에 의도한 Type이 올바르게 들어왔는지, Feature가 모두 있는지 등 미리 정해놓은 구조대로 구성되어 있는지 최소한의 검증을 미리하는 것이 모델의 안정성에 많은 도움이 됩니다.

<div class="row">
    <div style="width:45%; float:left; margin-right:10px;">
        <figure class="image" style="align: center;">
            <p align="center">
                <img src="/assets/images/2021-02-21-data_is_tested/test-normal-case.png" alt="" width="120%">
                <figcaption style="text-align: center;">[그림{{ i }}] Test Normal Case</figcaption>
            </p>
        </figure>
    </div>
    {% assign i = i | plus: 1 %}
    <div style="width:45%; float:right;">
        <figure class="image" style="align: center;">
            <p align="center">
                <img src="/assets/images/2021-02-21-data_is_tested/test-input-sample.png" alt="" width="120%">
                <figcaption style="text-align: center;">[그림{{ i }}] Test Input Sample</figcaption>
            </p>
        </figure>
    </div>
    {% assign i = i | plus: 1 %}
</div>

### 1.1 About Json Schema

**Json Schema**를 활용해 위 예시를 포함해 데이터의 약속된 형태를 표현할 수 있습니다.
Json Schema란 `JSON`형식으로 작성된 다른 데이터의 구조를 설명하는 하나의 데이터입니다 [[2]](#ref-2).
의도하는 데이터의 형식을 표현하고, 새로 들어오는 데이터가 Schema에 맞는지 검증합니다.

Json Schema의 주요 요소를 소개해 드리겠습니다.

- "type"
  - Schema에서 데이터 형식을 지정합니다.
  - `string`, `number`, `object`, `array`, `boolean`, `null`
  - [공식 페이지 - type](https://json-schema.org/understanding-json-schema/reference/type.html#type) 참고
 
- "properties"
  - `object` 데이터 내 속성을 구체적으로 정의힙니다.
  - [공식 페이지 - properties](https://json-schema.org/understanding-json-schema/reference/object.html?highlight=required#properties) 참고
 
- "required"
  - `object` 데이터 내에 필수로 가져야하는 속성을 지정합니다.
  - [공식 페이지 - required properties](https://json-schema.org/understanding-json-schema/reference/object.html?highlight=required#required-properties) 참고
 

`make_feature_c` 입력 상황에 적용할 수 있는 Json Schema를 보여드리겠습니다.

서비스에 입력되는 데이터가 Feature 이름과 값이 맵핑되어 있는 Python `dict` 자료형이라고 가정해봅시다.
여기서 `dict`는 JSON 형식 중 "object"와 호환되는 자료 구조로 `type` Field의 값은 object로 합니다.
데이터는 `Feature A`와 `Feature B`를 필수로 가져야하므로 `required` Field에 추가합니다.
마지막으로 두 Feature 모두 Numeric 데이터를 가짐을 아래와 같이 `properties` Field를 작성합니다.

```json
{
    "type": "object",
    "required": ["feature_a", "feature_b"],
    "properties": {
        "feature_a": {"type": "number"},
        "feature_b": {"type": "number"},
    }
}
```

위에 설정한 검증 조건 외에 Feature마다 값의 범위, null 검증 조건을 표현할 수 있고,
`if-then-else` 구조를 사용해 복잡한 조건부 구조까지 표현할 수 있습니다.

### 1.2 Json schema validator

Python [jsonschema](https://pypi.org/project/jsonschema/) 패키지를 활용해 입력된 `dict` 데이터가 유효한 구조로 정의되어 있는지 검증하는 예시를 보여드리겠습니다.

```python
>>> from jsonschema import validate
>>> # Schema를 선언합니다.
>>> schema = {
    "type": "object",
    "required": ["feature_a", "feature_b"],
    "properties": {
        "feature_a": {"type": "number"},
        "feature_b": {"type": "number"},
    }
}

>>> # 유효한 Sample을 생성합니다.
>>> sample = {
    "feature_a" : 3,
    "feature_b" : 4,
}
>>> # 유효성 확인을 통과한 경우, 오류없이 통화합니다.
>>> validate(instance=sample, schema=schema)

>>> # 유효하지 않은 Sample을 생성합니다.
>>> sample = {
    "feature_a" : False,
    "feature_b" : 4,
}
>>> # 유효성 확인을 통과하지 못 한 경우, 오류와 함께 이유를 반환합니다.
>>> validate(instance=sample, schema=schema)
ValidationError: False is not of type 'number'

Failed validating 'type' in schema['properties']['feature_a']:
    {'type': 'number'}

On instance['feature_a']:
    False
```

추론 시점마다 1개 이상의 데이터가 들어오는 환경에서 사용하는 경우 아래와 같이 For Loop을 이용해 확인할 수 있습니다.
>>>> 권장되는 방식 

```python
import jsonschema
from jsonschema import validate

def json_schema_validator(samples):
    for sample in samples:
        validate(instance=sample, schema=schema)
```

## 2. Input Feature

실시간으로 입력되는 데이터 Sample은 Feature(Column)로 구성되어 있습니다.
지속적으로 동알한 형태의 Feature가 입력되는지 확인하는 과정이 필요합니다.

<figure class="image" style="align: center;">
    <p align="center">
        <img src="/assets/images/2021-02-21-data_is_tested/feature.png" alt="" width="50%">
        <figcaption style="text-align: center;">[그림{{ i }}] Feature</figcaption>
    </p>
</figure>
{% assign i = i | plus: 1 %}

딥러닝 모델은 입력 데이터의 Feature Shape만 동일하다면 추론을 통해 결과를 얻을 수 있습니다.
`Feature A`, `Feature B`, `Feature C` 순서로 들어오던 데이터가 `Feature B`, `Feature C`, `Feature A` 순서로 입력돼도 모델은 문제없이 추론합니다.
이런 경우 모델의 추론 결과를 통해 입력 오류를 확인하는 것은 어렵습니다.
오류를 방지하기 위해서는 Input Feature에 대한 검증이 필요합니다.

### 2.1 Preprocessing for Validity

데이터가 `pandas.DataFrame` 형태로 변경한다면, Feature의 순서를 고정하는 것으로 검증 과정을 대신할 수 있습니다.
전처리 과정 중 아래와 같은 클래스를 이용해 Feature의 순서를 고정할 수 있습니다.

```python
import pandas as pd

class ColumnAligner():
    """동일한 DataFrame의 column 순서를 보장합니다."""

    def __init__(self):
        self.column_alignment = None

    def fit(self, df: pd.DataFrame):
        """fit하는 DataFrame의 컬럼 순서를 저장합니다.

        Parameters
        ----------
        df : pandas.DataFrame
            표준 컬럼 순서를 갖는 데이터.
        """
        self.column_alignment = df.columns.tolist()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """저장된 컬럼 순서로 순서를 변경합니다.

        모든 컬럼을 포함하고 있을 때를 가정합니다.

        Parameters
        ----------
        df : pandas.DataFrame
            컬럼 재배치 대상이 되는 데이터.

        Returns
        -------
        pandas.DataFrame
            컬럼 재배치된 데이터.
        """
        return df.loc[:, self.column_alignment]
```

학습 데이터로 전처리 클래스를 fit을 하고, 전처리 과정에 transform을 추가해 준다면 항상 학습 데이터와 같은 형태의 데이터를 얻을 수 있습니다.


```python
>>> # train_data를 기준으로 test_data의 col 순서를 맞춥니다.
>>> column_aligner = ColumnAligner()
>>> column_aligner.fit(train_data)
>>> test_data = column_aligner.transform(test_data)
```

### 2.2 Unit Test Preprocessing

`ColumnAligner`  클래스에 대한 Unit Test를 추가해 안정성을 높일 수 있습니다.

```python
def test_column_aligner_fit():
    # `ColumnAligner`의 fit 함수를 테스트 합니다.
    column_aligner = ColumnAligner()

    df = pd.DataFrame([{
        "feature_a": 1,
        "feature_b": 3,
        "feature_c": 5,
    }])
    """
    >>> df
       feature_a  feature_b  feature_c
    0          1          3          5
    """

    # `df` 컬럼 순서를 저장합니다.
    column_aligner.fit(df)

    # `column_aligner`에 저장된 컬럼 순서를 확인합니다.
    assert column_aligner.column_alignment \
        == ["feature_a", "feature_b", "feature_c"]


def test_column_aligner_transform():
    # `ColumnAligner`의 transform 함수를 테스트 합니다.
    column_aligner = ColumnAligner()

    # transform 함수만 확인하기 위해 fit 함수의 결과를 직접 저장합니다.
    stored_column_alignment = ["feature_a", "feature_b", "feature_c"]
    column_aligner.column_alignment = stored_column_alignment

    # 컬럼이 역순으로 존재하는 `test_df`를 생성합니다.
    df = pd.DataFrame([{
        "feature_c": 5,
        "feature_b": 3,
        "feature_a": 1,
    }])
    """
    >>> test_df
       feature_c  feature_b  feature_a
    0          5          3          1
    """

    # `column_aligner`를 이용해 `test_df` 컬럼 순서를 재배치한
    # `preprocessed_df`를 생성합니다.
    preprocessed_df = column_aligner.transform(test_df)

    # 재배치된 데이터`preprocessed_df`가 동일한 순서의 컬럼을 갖는지 확인합니다.
    assert preprocessed_df.columns.tolist() == stored_column_alignment

```

전처리 과정의 코드는 단순해서 Unit Test가 필요없어 보일 수 있습니다.
하지만 데이터가 전처리 함수를 지나면서 변해가는 과정에 생기는 문제는 쉽게 파악하기 어렵습니다 [[1]](#ref-1).
전처리 과정이 제대로 동작하는지 계속 확인 하는 것은 매우 중요합니다.

## 3. Input Dataset

여러 Sample과 Feature로 구성된 Dataset에 대해 유효성 검증이 필요합니다.

<figure class="image" style="align: center;">
    <p align="center">
        <img src="/assets/images/2021-02-21-data_is_tested/dataset.png" alt="" width="50%">
        <figcaption style="text-align: center;">[그림{{ i }}] Dataset</figcaption>
    </p>
</figure>
{% assign i = i | plus: 1 %}

Validation Dataset을 중심으로 소개해 드리겠습니다.
Validation Dataset은 학습에 사용되지 않은 데이터로서 주로 학습된 모델을 평가하는 데 사용됩니다.
마키나락스 이상탐지 시스템에서는 Validation Dataset의 Anomaly Score를 이용해 알람의 Threshold를 결정합니다.

Validation Dataset이 모델을 평가하는데 적절하지 않은 데이터 셋이었다면 어떻게 될까요?
모델에 대한 평가도 왜곡되고, 마키나락스 이상탐지 시스템에서 중요한 Threshold가 잘 못 계산될 수 있습니다.
Validation Dataset은 시스템의 전체적인 성능 안정성을 위해 검증되어야 합니다.

### 3.1 Example of Invalid Validation Dataset
>>>>> 흐름 다시
시계열 데이터는 [그림{{ i }}]과 같이 데이터 중 가장 오래된 부분을 Train Dataset으로, 나머지 뒷 부분을 Validation Dataset으로 분할해 사용합니다.
월요일부터 일요일까지 일주일 데이터를 이용해 모델을 학습하는 상황을 가정하겠습니다.
Train Dataset : Validation Dataset 비율을 5 : 2로 할 경우, [그림{{ i | plus: 1}}]과 같이 월요일부터 금요일까지 데이터를 Train Dataset으로,
토요일과 일요일 데이터를 Validation Dataset으로 사용하게 됩니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-21-data_is_tested/train_valid.png" alt="train-valid" width="120%">
  <figcaption style="text-align: center;">[그림{{ i }}] Train / Validation split</figcaption>
</p>
</figure>
{% assign i = i | plus: 1 %}

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-21-data_is_tested/week_train_valid.png" alt="train-valid-a-week" width="120%">
  <figcaption style="text-align: center;">[그림{{ i }}] Train / Validation split - A week</figcaption>
</p>
</figure>
{% assign i = i | plus: 1 %}

**하지만 이때 일요일이 휴일이라면 적합한 분할일까요?**
일요일 데이터는 작업이 이뤄지는 월요일부터 토요일까지의 데이터와 다른 분포를 갖게되므로 토요일까지의 데이터만 사용하는 것이 적합합니다.
모델을 학습하기 전 이런 상황을 몰랐다면, 마키나락스 이상탐지 시스템에서는 의도와 다른 Threshold 결과를 출력하고 비정상적인 작동을 하게 될 것입니다.

주중에 주말로 변화하는 것과 같이 데이터의 성격이 중간에 달라지는 상황을 Dataset Shift라고 합니다 [[4]](#ref-1).
온라인 환경의 상황을 모를 때 Validation Dataset의 Dataset Shift 여부를 확인하는 과정에 대해 소개드리겠습니다.

### 3.2 Test Dataset Shift in Validation Dataset
>>>>> 조금 더 쉽게
Output은 k차원의 Input Dataset의 Joint Distribution으로 만들어진 1차원의 데이터로 
Input Dataset이 변경은 모델의 Output 분포를 변화시킵니다.
$$P(y | x_0, x_1, ..., x_{k-1})$$
역으로 Output 분포의 변화를 이용해 Dataset Shift 여부를 확인할 수 있습니다 [[4]](#ref-1).

Output의 분포 변화를 확인하기 위해 두 집단 간의 평균을 비교하는 통계적 검정방법인 T-test를 이용합니다.
Python scipy 패키지를 이용해 임의로 생성한 두 집단을 비교해 보겠습니다.

```python
>>> import numpy as np
>>> from scipy import stats

>>> # 평균 0, 분산 1인 Normal Distribution에서 샘플이 100개인 집단을 생성합니다.
>>> group_a = np.random.randn(100) 

>>> # 평균 3, 분산 4인 Normal Distribution에서 샘플이 100개인 집단을 생성합니다.
>>> group_b = 3 + 2 * np.random.randn(100) 

>>> # 두 집단을 비교합니다.
>>> t_statistic, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

>>> # p_value가 0.05보다 작은 경우 평균이 다른 집단으로 판단합니다.
>>> if p_value < 0.05:
        print("Difference between the means of two groups")
```

위 과정을 확장해서 온라인 환경의 상황을 모를 때 Validation Dataset의 Dataset Shift 여부를 확인하는 과정에 대해 소개드리겠습니다.
시간이 지남에 따라 Dataset Shift 여부를 확인하기 위함으로 한 시점을 기준으로 이전 시점 데이터를 Group-pre, 이후 시점 데이터를 Group-post로 구분하여 T-test를 진행합니다.
단 Train Dataset과 Validation Dataset 사이에는 Dataset Shift가 없어 Group-pre는 Train Dataset과 같은 분포일 것을 가정하는 것이 필요합니다.

이때 집단을 구분하는 시점을 하나로 고정할 경우 여러 상황에 대응하기 어렵습니다.
예를 들어, 데이터의 50%를 기준으로 한다면 [그림{{ i }}] 상황에서는 Dataset Shift를 확인할 수 있지만, [그림{{ i | plus: 1 }}] 상황에서는 불가능합니다.

```python
>>> # 평균 0, 분산 1인 Normal Distribution에서 샘플이 100개인 집단을 생성합니다.
>>> group_a = np.random.randn(100) 

>>> # 평균 30, 분산 4인 Normal Distribution에서 샘플이 100개인 집단을 생성합니다.
>>> group_b = 30 + 2 * np.random.randn(100) 

>>> # [그림{{ i }}] 데이터는 group_a와 group_b를 연결한 데이터 입니다.
>>> graph_11 = np.append(group_a, group_b)

>>> # [그림{{ i | plus: 1 }}] 데이터는 [그림{{ i }}] 데이터에 group_a를 추가로 연결한 데이터 입니다.
>>> graph_12 = np.append(graph_11, group_a)
```

<div class="row">
    <div style="width:45%; float:left; margin-right:10px;">
        <figure class="image" style="align: center;">
            <p align="center">
                <img src="/assets/images/2021-02-21-data_is_tested/group-around-5-1.png" alt="" width="120%">
                <figcaption style="text-align: center;">[그림{{ i }}] 1/2를 기준으로 변경된 경우</figcaption>
            </p>
        </figure>
    </div>
    {% assign i = i | plus: 1 %}
    <div style="width:45%; float:right;">
        <figure class="image" style="align: center;">
            <p align="center">
                <img src="/assets/images/2021-02-21-data_is_tested/group-around-5-2.png" alt="" width="120%">
                <figcaption style="text-align: center;">[그림{{ i }}] 1/3와 2/3를 기준으로 변경된 경우</figcaption>
            </p>
        </figure>
    </div>
    {% assign i = i | plus: 1 %}
</div>

여러 시점에 대해 Dataset Shift를 판단하는 것이 필요합니다.
시점을 정하는 방법으로 [그림{{ i }}]과 같이 균등하게 구간을 나누는 방법을 사용할 수 있습니다.
[그림{{ i }}]은 Dataset Shift 후보로 균등하게 분포한 4개 구간을 정의한 상황입니다.
더 많은 구간을 구분할 경우 정확도가 올라갈 수 있지만, 확인 과정이 오래걸릴 수 있다는 Trade-off가 있습니다.
추가로 각 시점에 대해 T-test 검증 후 P-value가 가장 낮은 시점을 중심으로 Dataset Shift가 일어났음을 예상할 수 있습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-21-data_is_tested/group-split-5.png" alt="train-valid-a-week" width="120%">
  <figcaption style="text-align: center;">[그림{{ i }}] 1/5, 2/5, 3/5 ,4/5를 기준으로 그룹화</figcaption>
</p>
</figure>
{% assign i = i | plus: 1 %}

```python
# Dataset Shift가 예상된 경우 ValueError를 raise합니다.
def check_dataset_shift(
    data: np.ndarray,          # 평가 대상이 되는 Output
    n_test_points: int = 5,      # 구간 분할 횟수
):
    # 구간 하나의 크기를 정합니다.
    split_size = len(data) // n_test_points

    min_p_value = float("inf")
    shift_point = None
    for test_point in range(1, n_test_points):

        # Group-pre는 판단 시점 이전 데이터를 할당합니다.
        # Group-post는 판단 시점 이후 데이터를 할당합니다.
        group_pre = data[: split_size * test_point]
        group_post = data[split_size * test_point :]

        t_statistic, p_value = stats.ttest_ind(
            group_pre,
            group_post,
            equal_var=False,
        )

        # 가장 작은 P-value와 해당 시점을 저장합니다.
        if min_p_value > p_value:
            min_p_value = p_value
            shift_point = test_point

    # 최소 P-value가 0.05보다 작을 때 Dataset Shift를 판단하고 ValueError를 raise합니다.
    if min_p_value < 0.05:
        raise ValueError(f"Check dataset shift around {shift_point}/{n_test_points}")

```

[그림{{ i | minus: 2 }}]과 [그림{{ i | minus: 1 }}]에 사용된 데이터를 적용하면 아래와 같은 결과를 얻을 수 있습니다.

```python
>>> # [그림{{ i | minus: 2 }}] 데이터를 6개 구간으로 나눠 Dataset Shift를 확인합니다.
>>> # 3/6 지점에서 Dataset Shift가 일어났음을 예상할 수 있습니다.
>>> check_dataset_shift(graph_11, n_test_points=6)
ValueError: Check dataset shift around 3/6

>>> # [그림{{ i | minus: 1 }}] 데이터를 6개 구간으로 나눠 Dataset Shift를 확인합니다.
>>> # 2/6 지점에서 Dataset Shift가 일어났음을 예상할 수 있습니다.
>>> check_dataset_shift(graph_12 n_test_points=6)
ValueError: Check dataset shift around 2/6
```

P-value가 가장 작은 한 지점을 출력하므로, 여러 지점에서의 변화를 확인하는데는 한계가 있습니다.
그리고 Group-pre는 Train Dataset과 같은 분포라는 가정이 틀릴 수 있습니다.
하지만 최소한의 검증으로 사용될 수 있음을 기대할 수 있습니다.
## 4. Conclusion

이번 포스트에서 데이터 유효성 검증이 필요한 이유와 검증 방법을 다뤄보았습니다.
데이터를 이용해 모델을 학습하는 머신러닝 특성상 코드뿐만 아니라 데이터에 대해 유효성을 판단하는 과정이 필요합니다.
이번 포스트를 통해서 비슷한 문제를 고민하는 분들께 도움이 되었으면 좋겠습니다.

<a name="ref-1">[1]</a> [Eric Breck Shanqing Cai Eric Nielsen Michael Salib D. Sculley, 2017, The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction.](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf)

<a name="ref-2">[2]</a> [https://json-schema.org](https://json-schema.org/)

<a name="ref-3">[3]</a> [H. Khizou. "Unit Testing Data: What Is It and How Do You Do It?" winderresearch.com (accessed Apr. 13, 2021)](https://winderresearch.com/unit-testing-data-what-is-it-and-how-do-you-do-it/)


<a name="ref-4">[4]</a> [J. Quiñonero Candela, M. Sugiyama, A. Schwaighofer, and N. D. Lawrence. Dataset Shift in Machine Learning Shift in Machine Learning. The MIT Press 2009 The MIT Press, 2009.](http://www.acad.bg/ebook/ml/The.MIT.Press.Dataset.Shift.in.Machine.Learning.Feb.2009.eBook-DDU.pdf)
