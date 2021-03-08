---
layout: post
title: 온라인 환경에서 Data 검증하기
author: minjoo lee
categories: [deeplearning]
image: assets/images/2021-02-21-data_is_tested/total.gif
---

마키나락스가 이상탐지 시스템을 적용하고자 하는 제조 생산 현장은 제품 및 공정의 변화가 잦은 곳이 많습니다.
공정이 변화하면 데이터 분포가 달라지기 때문에, 사전에 학습된 모델이 적용하려는 시점에는 정상 작동하기 어렵습니다.
이런 문제를 해결하기 위해 온라인 환경에서 학습과 추론이 동시에 가능한 코드 형태로 모델을 배포합니다.
온라인 환경에서 사용되는 모델은 배포된 코드와 함께 현장에서 수집한 학습 데이터로 완성됩니다.[[1]](#ref-1)

온라인 환경에서 모델이 안정적으로 학습되고 추론하기 위해서는 코드뿐만 아니라 데이터에 대해서 유효성 테스트가 필요합니다.
앞선 포스트에서 코드의 안정성을 보장하기 위한 Software Test와 Regression Test에 대해서 소개해 드렸습니다.
이번 포스트에서는 온라인 환경에서 데이터의 유효성을 확인할 수 있는 방법에 대해 소개드리겠습니다. 

## Why data need test?

온라인 환경에서 예상과 다르게 입력된 데이터로 모델을 학습, 추론한 경우 의도와 다른 결과를 출력할 수 있습니다.
예를 들어, Numeric 데이터 입력을 받는 연산 코드에서 Boolean 데이터 `False`가 입력됐을 경우를 생각해보겠습니다.
입력된 데이터의 Type이 다르지만 `False`를 0으로 변환해 연산한 결과가 출력될 수 있습니다.
이 경우 코드가 작동하는데 문제 없기 때문에, 나중에 출력 결과를 디버깅하는 것은 불가능하다고 볼 수 있습니다.
이와 같은 문제는 데이터 유효성 테스트를 통해 미리 방지할 수 있습니다.

데이터의 유효성을 확인하는 방법 3가지를 소개하겠습니다.
- Input Sample Test
- Input Feature Test
- Dataset Test

<div class="row">
    <div style="width:45%; float:left; margin-right:10px;">
        <figure class="image" style="align: center;">
            <p align="center">
                <img src="/assets/images/2021-02-21-data_is_tested/valid.gif" alt="valid-data" width="120%">
                <figcaption style="text-align: center;">Input Valid Data</figcaption>
            </p>
        </figure>
    </div>
    <div style="width:45%; float:right;">
        <figure class="image" style="align: center;">
            <p align="center">
                <img src="/assets/images/2021-02-21-data_is_tested/invalid.gif" alt="invalid-data" width="120%">
                <figcaption style="text-align: center;">Input Invalid Data</figcaption>
            </p>
        </figure>
    </div>
</div>

## Input Sample Test

Input Sample Test 란 입력으로 받은 데이터의 각 Sample(Row)의 유효성에 대해서 판단하는 Test입니다.

<figure class="image" style="align: center;">
    <p align="center">
        <img src="/assets/images/2021-02-21-data_is_tested/sample.png" alt="" width="50%">
        <!-- <figcaption style="text-align: center;">Sample</figcaption> -->
    </p>
</figure>

예를 들어, `Feature A`와 `Feature B`를 이용해 새로운 `Feature C`를 만들어 내는 Feature Engineering 코드가 있습니다. 

```python
def make_feature_c(feature_a, feature_b):
    return feature_a * feature_b
```
<figure class="image" style="align: center;">
    <p align="center">
        <img src="/assets/images/2021-02-21-data_is_tested/make-feature-c.png" alt="" width="50%">
        <!-- <figcaption style="text-align: center;"></figcaption> -->
    </p>
</figure>

Unit Test는 `Feature A`와 `Feature B`의 예상된 입력을 가정하고, `Feature C`가 생성되는 로직을 확인합니다. 

```python 
def test_make_feature_c():
    feature_a = 3
    feature_b = 4
    feature_c = make_feature_c(feature_a, feature_b)
    assert feature_c == 12
```

그런데 만약 온라인 환경에서 `Feature A` 또는 `Feature B`가 정상적으로 들어오지 않는 상황을 생각해 보겠습니다.

```python
>>> feature_a = False
>>> feature_b = 4
>>> feature_c = make_feature_c(feature_a, feature_b)

>>> feature_c
0
```

이때는 Engineering 할 Feature에 Boolean 값이 들어오면서 예상치 못한 `Feature C`가 만들어지게 됩니다. 
`Feature A`가 잘못 들어온 것으로 Error가 나는 것이 기대되지만, 이 경우 0이 나와 의도하지 않은 결과로 다음 프로세스까지 영향을 미치게 됩니다.

`make_feature_c` 함수 내에 입력된 값이 Numeric 아닌 경우 Error를 Raise하도록 구현할 수 있습니다.
하지만 앞의 예시 외에도 데이터가 입력되었을 때, 기본적으로 데이터에 의도한 Feature가 모두 있는지, Type이 올바르게 들어왔는지 등 미리 정해놓은 구조대로 구성되어 있는지 최소한의 검증을 미리하는 것이 모델의 안정성에 많은 도움이 됩니다.

<div class="row">
    <div style="width:45%; float:left; margin-right:10px;">
        <figure class="image" style="align: center;">
            <p align="center">
                <img src="/assets/images/2021-02-21-data_is_tested/test-normal-case.png" alt="" width="120%">
                <figcaption style="text-align: center;">Test Normal Case</figcaption>
            </p>
        </figure>
    </div>
    <div style="width:45%; float:right;">
        <figure class="image" style="align: center;">
            <p align="center">
                <img src="/assets/images/2021-02-21-data_is_tested/test-input-sample.png" alt="" width="120%">
                <figcaption style="text-align: center;">Test Input Sample</figcaption>
            </p>
        </figure>
    </div>
</div>


**Json Schema**를 활용해 약속한 대로 데이터가 들어오는지 확인할 수 있습니다. Json Schema란 `JSON`형식으로 작성된 다른 데이터의 구조를 설명하는 하나의 데이터 자체입니다.[[2]](#ref-2) 의도하는 데이터의 형식을 표현하고, 새로 들어오는 데이터가 Schema에 맞는지 검증합니다.

### Json Schema

Json Schema를 주요 요소를 소개해 드리겠습니다.

- "type"
  - Schema에서 데이터 형식을 지정합니다.
  - `string`, `number`, `object`, `array`, `boolean`, `numm`
  - [공식 페이지 - type](https://json-schema.org/understanding-json-schema/reference/type.html#type) 참고
  
- "properties" 
  - `object` 데이터 내 속성을 구체적으로 정의힙니다. 
  - [공식 페이지 - properties](https://json-schema.org/understanding-json-schema/reference/object.html?highlight=required#properties) 참고
  
- "required"
  - `object` 데이터 내에 필수로 가져야하는 속성을 지정합니다.
  - [공식 페이지 - required properties](https://json-schema.org/understanding-json-schema/reference/object.html?highlight=required#required-properties) 참고
  

앞의 예시에 적용할 수 있는 Json Schema를 보여드리겠습니다.

현재 마키나락스에서 구현한 모델의 입력 데이터는 대부분 Feature 이름과 값이 맵핑되어 있는 Python `dict` 자료형 입니다. 여기서 `dict`는 JSON 형식 중 "object"와 유사하므로 type Field의 값은 object로 했습니다. 데이터는 `Feature A`와 `Feature B`를 필수로 가져야하므로 required Field에 추가했습니다. 마지막으로 두 feature 모두 수치형 데이터이어야 하므로 아래와 같이 properties Field를 작성했습니다. (추가 사용법은 Json Schema[[2]](#ref-2) 참고)

```json
{
    "type": "object",
    "required": ["feature A", "feature B"],
    "properties": {
        "feature A": {"type": "number"},
        "feature B": {"type": "number"},
    }
}
```

### Json schema validator

입력된 `dict` 데이터가 구조와 맞는지 검증하는 예시를 보여드리겠습니다. Python [jsonschema](https://pypi.org/project/jsonschema/) 라이브러리를 활용해 데이터가 선언한 Json schema에 맞는지 검증할 수 있습니다.

```python
>>> from jsonschema import validate

>>> schema = {
    "type": "object",
    "required": ["feature A", "feature B"],
    "properties": {
        "feature A": {"type": "number"},
        "feature B": {"type": "number"},
    }
}

>>> data = {"feature A" : 3., "feature B" : 4.}
>>> print(validate(instance=data, schema=schema))

>>> data = {"feature A" : False, "feature B" : 4.}
>>> print(validate(instance=data, schema=schema))
ValidationError: False is not of type 'number'

Failed validating 'type' in schema['properties']['feature A']:
    {'type': 'number'}

On instance['feature A']:
    False
```

### Applications

추론 시점마다 데이터가 100개 들어오는 환경에서 사용하는 경우를 소개해 드리겠습니다. 
이때, 추론 주기가 짧다면 들어오는 100개의 데이터에 대해 모두 확인하는 경우 결과를 출력하는데
시간이 오래걸릴 수 있습니다. 이럴 경우 random으로 10% 데이터만 추출해 검증하고 넘어갈 수 있습니다.

```python
import jsonschema
from jsonschema import validate
import random

def json_schema_validator(datapoints, sample_ratio=0.1):
    n_total = len(datapoints)
    n_samples = max(int(n_total * sample_ratio), 1)
    sample_indices = random.sample(range(n_total), n_samples)
    for sample_index in sample_indices:
        sample = datapoints[sample_index]
        try:
            validate(instance=sample, schema=schema)
        except jsonschema.exceptions.ValidationError as ve:
            return False
    return True
```

## Input Feature Test

다음으로 Input Feature Test 란 입력으로 받은 데이터의 각 Feature(Column)의 유효성에 대해서 판단하는 Test입니다.

<figure class="image" style="align: center;">
    <p align="center">
        <img src="/assets/images/2021-02-21-data_is_tested/feature.png" alt="" width="50%">
        <!-- <figcaption style="text-align: center;">Feature</figcaption> -->
    </p>
</figure>

### Preprocessing for Validity

예를 들어서 딥러닝 모델은 Input의 Shape만 동일하다면 추론을 통해 결과를 얻을 수 있습니다.
`Feature A`, `Feature B`, `Feature C` 순서로 들어오던 데이터가 `Feature B`, `Feature C`, `Feature A` 순서로 입력될 때도 모델은 문제없이 추론합니다.
이런 경우 결과의 오류를 확인하는 것은 거의 불가능하다고 볼 수 있습니다. 
이러한 오류를 방지하기 위해서는 Input Feature의 순서에 대한 테스트를 해야합니다.

만약 `pandas.DataFrame` 형태라면 Feature의 순서를 고정하는 것으로 해결 할 수 있습니다. 
전처리 과정 중 아래와 같은 클래스를 이용해 Feature의 순서를 고정할 수 있습니다.

```python
import pandas as pd

class ColumnAligner():
    def __init__(self):
        self.column_alignment = None

    def fit(self, df: pd.DataFrame):
        self.column_alignment = df.columns.tolist()

    def transform(self, df: pd.DataFrame):
        return df.loc[:, self.column_alignment]
```

학습 데이터로 전처리 클래스를 fit을 하고, 전처리 과정에 transform을 추가해 준다면 항상 학습 데이터와 같은 형태의 데이터를 얻을 수 있습니다.


```python
>>> column_aligner = ColumnAligner()
>>> column_aligner.fit(train_data)
>>> test_data = column_aligner.transform(test_data)
```

### Unit Test Preprocessing

하지만 전처리 코드에 오류가 있다면 유효성이 보장되지 않을 것입니다.
해당 클래스 구현 코드에 대해 Unit Test를 추가해준다면, 코드의 안정성 보장을 통해 데이터의 안정성 보장까지 가능합니다.

```python
def test_column_aligner():
    column_aligner = ColumnAligner()

    df = pd.DataFrame({
        "feature A": [1],
        "feature B": [3],
        "feature C": [5],
    })
    '''
    >>> df
       feature A  feature B  feature C
    0          1          3          5
    '''
    test_df = df.copy()
    test_df = test_df.iloc[:, ::-1] 
    '''
    >>> test_df
       feature C  feature B  feature A
    0          5          3          1
    '''
    column_aligner.fit(df)
    preprocessed_df = column_aligner.transform(test_df)
    assert df.columns.equals(preprocessed_df.columns)

```

전처리 과정의 코드는 단순해서 Unit Test가 필요없어 보일 수 있습니다. 
하지만 데이터가 전처리 함수를 지나면서 변해가는 과정에 생기는 문제는 쉽게 파악하기 어렵습니다.[[1]](#ref-1) 
전처리 과정이 제대로 동작하는지 계속 확인 하는 것이 매우 중요합니다.

## Input Dataset Test

다음으로 입력 Dataset에 대해 유효성 Test입니다.

<figure class="image" style="align: center;">
    <p align="center">
        <img src="/assets/images/2021-02-21-data_is_tested/dataset.png" alt="" width="50%">
        <!-- <figcaption style="text-align: center;">Dataset</figcaption> -->
    </p>
</figure>

### Example of Invalid Validation Dataset

Validation Dataset 검증이 필요한 경우에 대해 예를 들어보겠습니다.

Validation Dataset은 학습에 사용되지 않은 데이터로서 주로 모델을 평가하는 데 사용됩니다. 
마키나락스에서는 Validation Dataset의 Anomaly Score를 이용해 알람의 Threshold를 결정하는데 사용합니다. 
그런데 Validation Dataset이 모델을 평가하는데 적절하지 않은 데이터 셋이었다면 어떻게 될까요?
모델에 대한 평가도 왜곡되고, 이상탐지 시스템에서 중요한 Threshold 값 또한 잘못 계산될 수 있습니다. 

시계얼 데이터에서는 데이터 중 가장 오래된 데이터부터 Train Set으로, 나머지 뒷 부분을 Validation Dataset으로 사용합니다.[그림1]
월요일 부터 일요일까지 일주일 데이터를 이용해 모델을 학습하는 상황을 예를 들어보겠습니다.
Train Set : Validation Dataset 비율을 5 : 2로 할 경우, 월요일부터 금요일까지 데이터를 Train Set으로, 토요일과 일요일 데이터를 Validation Dataset으로 사용하게 됩니다.[그림2]

그런데 일요일이 현장 휴일이라면 월요일 ~ 토요일과 다른 분포의 데이터를 갖게 될 것 입니다. 
이런 경우 토요일과 일요일로 구성된 Validation Dataset은 적절하지 않다고 할 수 있습니다.

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-21-data_is_tested/train_valid.png" alt="train-valid" width="120%">
  <figcaption style="text-align: center;">[그림1] - Train / Validation split</figcaption>
</p>
</figure>

<figure class="image" style="align: center;">
<p align="center">
  <img src="/assets/images/2021-02-21-data_is_tested/week_train_valid.png" alt="train-valid-a-week" width="120%">
  <figcaption style="text-align: center;">[그림2] - Train / Validation split - A week</figcaption>
</p>
</figure>

모델을 학습하기 전 이런 현장의 상황을 몰랐다면, 모델은 예상하지 못한 방향으로 작동하게 됩니다. (예시에서는 의도와 다른 Threshold 결과를 출력합니다.)
이처럼 현장에서는 실험 환경과 다른 경우를 예상해 전체적인 데이터셋의 결과에 대해 테스트 코드를 작성하는 것이 필요합니다. 

### Example of Validation Dataset Verification

유효한 데이터로 이루어진 Validation Dataset은 Train Set에 근접한 부분의 Anomaly Score와 먼 부분의 Anomaly Score가 비슷한 분포를 가집니다.
이 경우 확인 방법으로 AUROC를 적용할 수 있습니다. ([AUROC-위키피디아](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) 참고) 
데이터 분포가 변화하지 않는 경우라면 0.4 ~ 0.6 사이의 AUROC가 나올 것입니다. 
계산한 AUROC를 바탕으로 Validation Dataset의 유효성을 검증 할 수 있습니다.

```python
from sklearn.metrics import roc_auc_score

def check_validation_set(scores):
    near_label = len(scores) // 2
    far_label = len(scores) - near_label
    labels = [0] * near_label + [1] * far_label
    auroc = roc_auc_score(scores, labels)
    if abs(auroc - 0.5) > 0.1 :
        return False 
    return True
```

## Conclusion

이번 포스트에서 데이터 유효성 검증이 필요한 이유와 검증 방법을 다뤄보았습니다.
데이터를 이용해 모델을 학습하는 머신러닝 특성상 코드뿐만 아니라 데이터에 대해 유효성을 판단하는 과정이 필요합니다.
이번 포스트를 통해서 비슷한 문제를 고민하는 분들께 작은 도움이 되었으면 좋겠습니다.

<a name="ref-1">[1]</a> [Eric Breck Shanqing Cai Eric Nielsen Michael Salib D. Sculley, 2017, The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction.](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf)

<a name="ref-2">[2]</a> [https://json-schema.org](https://json-schema.org/)

<a name="ref-3">[3]</a> [https://winderresearch.com/unit-testing-data-what-is-it-and-how-do-you-do-it](https://winderresearch.com/unit-testing-data-what-is-it-and-how-do-you-do-it/)
