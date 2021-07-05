
## 1. Directory Convention

- 주요 컨텐츠는 `_posts`로 시작하는 디렉토리에 포함되어 있습니다. 또한 컨텐츠에 필요한 이미지는 `assets/images` 디렉토리에 들어 있습니다.

- Jekyll에서는 `_posts` 디렉토리 내에 있는 Markdown 또는 html 파일을 블로그의 Posting으로 인식합니다. 따라서 새로운 포스팅을 작성하고자 한다면 각 디렉토리의 `_posts`에 새로 파일을 추가하시면 됩니다.
  
- Jekyll의 Posting 파일들은 모두 아래와 같은 Naming Convention을 따라야 합니다.
    - `yy-mm-dd-new-posting-name.md`


## 2. Posting Convention

### 2.1. Header Field

- 모든 Posting 파일들은 다음 예시와 같은 Header를 가지고 있어야 합니다.

```
---
layout: post
title: Regression Test, Are you sure?
author: wontak ryu
categories: [test]
image: assets/images/2020-02-10-Regression-Test/13_.gif
---
```

- **layout**은 `post`여야 합니다.
- **title**은 내용에 맞게 임의의 String으로 설정할 수 있습니다.
- **author**는 작성자를 의미하며 추가하려면 `_config.yml`에 추가해야합니다.
- **categories**는 해당 포스팅의 Tag를 의미합니다.
- **image**는 썸네일이미지 경로를 적어주시면 됩니다.

### 2.2. Latex

- 수식은 Latex 문법에 따라 표기합니다.
- $$ 와 같이 double dollar sign을 사용하여 수식임을 나타냅니다.

```
$$\theta x_1 + (1-\theta)x_2 \in C$$
```

위 수식은 다음과 같이 표기됩니다.

$$\theta x_1 + (1-\theta)x_2 \in C$$

### 2.3. Image Convention

- Posting 파일에서 이미지를 삽일할 때 아래의 Convention을 따라야합니다.

```
<figure class="image" style="align: center;">
<p align="center">
  <img src="{image_path}" alt="{description of image}" width="{scale_ratio}%" height="{scale_ratio}%">
  <figcaption style="text-align: center;">{figcaption}</figcaption>
</p>
</figure>
```

- figure class는 `image`여야합니다.
- {}에 들어갈 내용을 적절히 넣어야합니다.


## 3. GitHub Convnention

작성 내용에 질문이 있거나 수정 사항을 발견하신 경우 다음 두 방법 중 하나로 남겨주시면 됩니다.

- 댓글 작성하기
- Repogitory에 이슈 생성하기

새로운 내용을 추가하거나 직접 편집하시고 싶으신 경우에는 새로운 Branch를 생성하여 먼저 수정하신 후 `Pull Request`를 생성해주시면 됩니다. 신규 작성 및 기존 내용 수정은 누구나 가능합니다.

### 3.1 Branch Naming Convention

TODO

## 4. Reference 표기법

TODO
