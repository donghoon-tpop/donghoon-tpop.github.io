---
layout: post

title: Implementing Variational Autoencoders
short_title: VAE implementation

category_name: AI
category_url: /blogs/AI/
post_category: Implementation

lead: >-
  VAE를 colab 환경에서 구현해보고, 생성된 이미지에 대한 분석을 수행합니다.

updated: August 2026
series: AI Study Notes

permalink: /blogs/AI/vae-implementation/

math: true

toc:
  - id: code analysis
    label: Code Analysis

  - id: Training Algorithm
    label: Training Algorithm

back_url: /blogs/AI/
back_label: Back to AI

next_url: /blogs/AI/vae/
next_label: Understanding the VAE
---

<section id="code analysis" markdown="1">

## 1. Code Analysis

지난 포스트에서는 Variational Autoencoder가 등장하게 된 배경을 살펴보고, 그 원리와 학습 방법에 대해 수학적, 직관적으로 이해해 보았다. 이번 포스트에서는 https://wikidocs.net/271490 를 참고하여 pytorch를 활용한 variational autoencoder를 구현해보고, MNIST dataset를 학습시켜 보도록 하자.


Tensorflow가 아닌 Pytorch로 implementation을 직접 수행해 보는 것이 처음인 만큼 코드를 다소 세세하게 뜯어보는 과정을 거쳐볼 것이다.



**1\) Probabilistic Latent Variable Model**
    
오래전부터 연구자들은 관측 데이터 $x$가 직접 만들어진 것이 아니라, **눈에 보이지 않는 어떤 원인 $z$로부터 간접적으로 생성**된 것이라고 생각했다.

*Calvin Luo*는 이를 플라톤의 *Allegory of the Cave*에서의 비유로 설명하였다. 동굴 안에 갇혀 거주하는 인간들은 외부의 3차원 물체들이 동굴 내부의 불에 의해 투영한 2차원 그림자만으로 해당 물체들을 인식할 것이다.

마찬가지로 우리가 관찰하는 데이터 $x$는 보다 본질적인 원인 $z$로부터 형성된 것이라는 heuristic한 추론이 가능하다. (물론 보통 latent variable은 관측 데이터보다 훨씬 적은 차원의 수로 표현되므로, 플라톤의 비유와는 다소 차이가 존재한다.)

이를 위한 probabilistic latent variable model은 데이터가 어떻게 생성되는지 보다 근거 있게 설명할 수 있었다. 그러나 이 방법에는 학습과 추론이 어렵다는 한계가 존재했다.

생성형 모델에서는 $z$를 입력하여 $x$를 만드는 방향성으로 생성이 진행된다. 그러나 실제로 학습을 진행할 때 필요한 것은 $x$가 주어졌을 때 $z$의 분포를 추론하는 것이다.

이를 **posterior inference**라고 한다. 

간단한 모델에서는 이 방향성으로의 추론이 가능하지만, 여러 층의 random variable과 nonlinear neural network 구조에서는 $p_\theta(z \mid x)$를 정확히 계산하기 매우 어려워지는 것이다. 따라서 표현력이 좋은 생성 모델을 만들고자 할수록 그 모델의 posterior를 계산하기 어려워진다는 trade-off로 인해 bottleneck이 생기게 되었다. 

이러한 문제를 해결하기 위해 몇 가지 흐름이 탄생하였다. 1990년도 연구자들이 제안한 wake-sleep algorithm (generative network와 recognition network를 번갈아 학습시키는 알고리즘)과, 통계학과 머신러닝에서 등장한 variational inference가 그것이다. 

우리가 특히 주목해야 할 것은 이 variational inference이다. 진짜 posterior를 계산하기 어려우므로, 이를 근사하는 $q(z)$를 학습하는 것이다. 그러나 이 variational inference에서는 일반적인 probabilistic model에 적용하기 어렵다는 치명적인 단점이 존재하였다. 

**2\) Autoencoders**

앞선 흐름과는 별개로 Autoencoder가 발전하고 있었다. 일반적인 autoencoder는 
$$
x \rightarrow z \rightarrow \tilde{x}
$$
의 구조를 가지고 있다. 즉, encoder가 데이터를 짧은 code로 압축하고 decoder가 그 code에서 원래 데이터를 복원하는 것이다.

이는 back-propagation으로 쉽게 학습할 수 있다는 큰 장점이 존재한다. 그러나 일반 autoencoder의 latent vector를 생성형 모델을 위한 그것으로 사용하기에는 어려움이 있다.
각 이미지를 latent space의 특정 위치에 잘 배치했다고 해도, 데이터가 존재하지 않는 빈 latent space는 어디인지, latent vector는 실제로 어떻게 분포되어 있는지, 그리고 무엇보다 **임의로 뽑은 latent vector가 정상적인 출력을 생성할지** 알 수 없다는 것이다. 

즉, 엄밀한 의미에서 autoencoder는 확률분포를 학습하는 생성모델은 아니었던 것이다. 
<aside class="note-box">
  <p class="box-label">Intuition</p>

  <p>
  VAE 탄생 이전 통계학에서의 variational inference와 머신러닝에서 autoencoder라는 별개의 두 흐름이 존재했다.
  </p>
</aside>

</section>

<p>
# 생성 모델에서 Latent Noise가 출력에 미치는 영향

## 이론, 실험적 근거, 그리고 재현 가능한 평가 방법

## 1. 핵심 요약

이 보고서가 다루는 문제는 다음과 같다.

학습이 끝난 생성 모델의 latent vector (z)에 Gaussian noise를 더했을 때,

$$
\tilde{z}=z+\varepsilon,
\qquad
\varepsilon\sim\mathcal{N}(0,\Sigma),
$$

decoder 또는 generator의 출력이

$$
x=g(z)
$$

에서

$$
\tilde{x}=g(z+\varepsilon)
$$

로 얼마나 변하는가?

여기서는 channel noise를 포함하여 네트워크를 새로 학습하는 연구보다는, **이미 학습된 decoder에 latent perturbation을 가했을 때 출력이 어떻게 변하는지를 분석한 연구**에 초점을 맞춘다.

이 문제에 대한 가장 중요한 이론적 결과는 다음과 같다.

Noise가 충분히 작다면 decoder를 현재 latent point (z) 근처에서 선형 함수처럼 근사할 수 있다.

$$
g(z+\varepsilon)-g(z)
\approx
J_g(z)\varepsilon,
$$

여기서

$$
J_g(z)=\frac{\partial g(z)}{\partial z}
$$

는 decoder의 Jacobian이다.

따라서 출력 변화의 평균 제곱 크기는 대략

$$
\mathbb{E}_{\varepsilon}
\left[
|g(z+\varepsilon)-g(z)|_2^2
\mid z
\right]
\approx
\operatorname{tr}
\left(
J_g(z)^\top J_g(z)\Sigma
\right)
$$

로 표현된다.

AWGN처럼 모든 latent 방향에 같은 분산의 noise가 들어간다면

$$
\Sigma=\sigma^2 I
$$

이므로,

$$
\boxed{
\mathbb{E}_{\varepsilon}
\left[
|g(z+\varepsilon)-g(z)|_2^2
\mid z
\right]
\approx
\sigma^2|J_g(z)|_F^2
}
$$

가 된다.

즉, 출력 변화는 단순히 noise variance (\sigma^2)만으로 결정되지 않는다.

> 동일한 크기의 noise라도 decoder의 Jacobian이 큰 위치에서는 출력이 크게 변하고, Jacobian이 작은 위치에서는 출력이 거의 변하지 않는다.

또한 Jacobian의 singular value를 (s_i(z))라고 하면,

$$
|J_g(z)|_F^2
============

\sum_i s_i(z)^2
$$

이므로 평균적인 AWGN 민감도는 모든 singular value의 제곱합으로 결정된다.

반면 크기가 (r) 이하인 perturbation 중 출력에 가장 큰 변화를 주는 방향은 최대 singular value에 의해 결정된다.

$$
\boxed{
\sup_{|\delta|_2\leq r}
|g(z+\delta)-g(z)|*2
\approx
r,s*{\max}\bigl(J_g(z)\bigr)
}
$$

따라서 다음 두 값은 서로 다른 의미를 갖는다.

* (|J_g(z)|_F^2): 무작위 Gaussian noise에 대한 평균 민감도
* (s_{\max}(J_g(z))): 가장 위험한 방향에 대한 최악의 국소 민감도

---

# 2. 문제 설정

Decoder 또는 generator를

$$
g:\mathbb{R}^{d_z}\rightarrow\mathbb{R}^{d_x}
$$

라고 하자.

* (d_z): latent dimension
* (d_x): 출력 차원
* VAE에서는 (g(z))를 decoder가 출력하는 평균 이미지로 생각할 수 있다.
* GAN에서는 (g(z))가 generator의 출력 자체다.

Latent vector에 additive Gaussian noise가 들어간다고 하자.

$$
\varepsilon\sim\mathcal{N}(0,\Sigma)
$$

$$
\tilde{z}=z+\varepsilon
$$

그리고 출력 변화를

$$
\Delta x
========

g(\tilde{z})-g(z)
$$

로 정의한다.

AWGN의 경우에는 보통

$$
\Sigma=\sigma^2I
$$

를 사용한다.

다만 서로 다른 모델을 비교할 때 주의할 점이 있다. 모델마다 latent scale이 다르기 때문에 동일한 (\sigma)를 그대로 사용하는 것은 공정하지 않을 수 있다.

예를 들어 한 모델의 latent 값이 대체로 ([-0.1,0.1])에 있고, 다른 모델은 ([-10,10])에 있다면 동일한 variance의 noise가 두 모델에 미치는 상대적인 영향은 매우 다르다.

따라서 비교 실험에서는 다음 중 하나를 사용해야 한다.

1. 각 latent dimension을 표준화한 뒤 noise를 추가한다.
2. Latent power를 기준으로 SNR을 정의한다.

Latent SNR은 다음처럼 정의할 수 있다.

$$
\mathrm{SNR}_{\mathrm{dB}}
==========================

10\log_{10}
\left(
\frac{\mathbb{E}|z|_2^2}
{\mathbb{E}|\varepsilon|_2^2}
\right)
$$

---

# 3. 작은 noise에 대한 1차 근사

Decoder를 (z) 근처에서 Taylor expansion하면

$$
g(z+\varepsilon)
================

g(z)
+
J_g(z)\varepsilon
+
\text{2차 이상의 항}
$$

이 된다.

Noise가 충분히 작다면 2차 이상의 항을 무시할 수 있으므로

$$
\Delta x
\approx
J_g(z)\varepsilon
$$

이다.

Gaussian noise의 평균이 0이므로 1차 근사에서는

$$
\mathbb{E}[\Delta x\mid z]
\approx
0
$$

이다.

출력 변화의 covariance는

$$
\operatorname{Cov}(\Delta x\mid z)
\approx
J_g(z)\Sigma J_g(z)^\top
$$

가 된다.

이 식은 latent noise가 출력 공간에서 어떤 형태의 noise로 변하는지를 보여준다.

Latent에서는 모든 방향에 동일한 AWGN이 들어가더라도, 출력에서는 일반적으로 isotropic noise가 되지 않는다.

Decoder가 어떤 방향은 크게 늘리고 어떤 방향은 압축하기 때문이다.

---

# 4. 평균 출력 distortion

출력 변화의 평균 제곱 크기는

$$
\mathbb{E}
\left[
|\Delta x|_2^2
\mid z
\right]
\approx
\operatorname{tr}
\left(
J_g(z)^\top J_g(z)\Sigma
\right)
$$

이다.

AWGN에서는

$$
\mathbb{E}
\left[
|\Delta x|_2^2
\mid z
\right]
\approx
\sigma^2
\operatorname{tr}
\left(
J_g(z)^\top J_g(z)
\right)
$$

이고,

$$
\operatorname{tr}
\left(
J_g(z)^\top J_g(z)
\right)
=======

|J_g(z)|_F^2
$$

이므로

$$
\boxed{
D_{\mathrm{local}}(z)
\approx
\sigma^2|J_g(z)|_F^2
}
$$

가 된다.

이 관계는 작은 noise 영역에서 latent AWGN이 만드는 출력 MSE를 예측하는 가장 직접적인 공식이다.

중요한 점은 latent dimension이 같다고 해서 noise 민감도가 같은 것이 아니라는 사실이다.

두 decoder가 모두 20차원 latent를 사용하더라도,

$$
|J_{g_1}(z)|*F^2
\gg
|J*{g_2}(z)|_F^2
$$

라면 첫 번째 모델의 출력이 훨씬 크게 변할 수 있다.

---

# 5. 평균 distortion만으로 충분하지 않은 이유

두 decoder의 평균 distortion이 같더라도, noise에 대한 위험 분포는 다를 수 있다.

Jacobian의 singular value를 (s_i)라고 하면 AWGN에 의한 출력 에너지의 분산은 작은 noise 영역에서 대략

$$
\operatorname{Var}
\left(
|\Delta x|_2^2
\mid z
\right)
\approx
2\sigma^4\sum_i s_i^4
$$

로 나타난다.

예를 들어 다음 두 경우를 생각해보자.

### 모델 A

$$
s_1^2+s_2^2+\cdots+s_d^2=100
$$

이지만 대부분의 sensitivity가 하나의 방향에 집중되어 있다.

$$
s_1^2\approx100
$$

### 모델 B

동일한 총 sensitivity를 여러 방향이 균등하게 나눈다.

$$
s_i^2\approx\frac{100}{d}
$$

두 모델의 평균 AWGN distortion은 비슷할 수 있다.

하지만 모델 A에서는 noise가 민감한 단 하나의 방향에 많이 투영되는 경우 출력이 갑자기 크게 변할 수 있다.

즉 다음 두 값을 함께 봐야 한다.

$$
\sum_i s_i^2
$$

와

$$
s_{\max}^2
$$

첫 번째는 평균적인 민감도이고, 두 번째는 가장 위험한 방향의 민감도다.

민감도가 소수 방향에 얼마나 집중되어 있는지를 나타내는 간단한 지표로

$$
\frac{s_{\max}^2}
{\sum_i s_i^2}
$$

를 사용할 수 있다.

이 값이 1에 가까우면 하나의 방향이 전체 민감도를 지배한다.

---

# 6. 대칭적인 noise도 평균 출력에 편향을 만들 수 있다

Noise가 0을 중심으로 대칭적이므로 출력 변화의 평균도 항상 0일 것처럼 보일 수 있다.

하지만 이는 decoder가 국소적으로 선형일 때만 그렇다.

2차 Taylor term까지 고려하면 출력의 (k)번째 성분에 대해

$$
\mathbb{E}[\Delta x_k\mid z]
\approx
\frac{1}{2}
\operatorname{tr}
\left(
H_k(z)\Sigma
\right)
$$

가 된다.

여기서 (H_k(z))는 출력의 (k)번째 성분에 대한 Hessian이다.

즉 대칭적인 Gaussian noise라 하더라도 decoder가 휘어져 있으면 평균 출력이 한쪽 방향으로 이동할 수 있다.

이 효과로 인해 noise가 커질수록 단순히 이미지에 무작위 흔들림이 생기는 것이 아니라,

* 평균 색상이 달라지거나
* 얼굴의 평균적인 표정이 이동하거나
* 물체의 자세가 변하거나
* 특정 class 쪽으로 출력이 편향되는

현상이 나타날 수 있다.

---

# 7. Riemannian geometry 관점

다음 행렬을 생각하자.

$$
M(z)=J_g(z)^\top J_g(z)
$$

작은 latent perturbation (\delta)가 만들 출력 거리의 제곱은

$$
|g(z+\delta)-g(z)|_2^2
\approx
\delta^\top M(z)\delta
$$

로 근사된다.

따라서 (M(z))는 latent 공간에서 출력 변화량을 정의하는 국소적인 거리 행렬이다.

이 행렬은 decoder가 출력 공간의 Euclidean geometry를 latent 공간으로 끌어온 **pullback Riemannian metric**으로 해석할 수 있다.

이 관점을 체계적으로 제시한 대표적인 논문이 다음 연구다.

## Latent Space Oddity: On the Curvature of Deep Generative Models

Arvanitidis, Hansen, Hauberg, 2018

이 논문의 핵심 문제의식은 다음과 같다.

> Latent 공간에서 동일한 거리만큼 움직였다고 해서 출력도 동일한 정도로 변하는가?

일반적으로 답은 아니오다.

Decoder가 어떤 영역은 크게 늘리고, 다른 영역은 압축하기 때문에 latent 공간은 출력 관점에서 휘어진 공간처럼 작동한다.

따라서 단순한 Euclidean 거리

$$
|z_1-z_2|_2
$$

보다 decoder metric을 고려한 거리가 실제 출력 변화를 더 잘 반영한다.

이 논문은 명시적인 AWGN 대 PSNR 실험을 주요 목적으로 하지는 않았지만, latent noise 문제에 필요한 가장 중요한 이론적 토대를 제공한다.

Gaussian perturbation에 대해 이 quadratic form의 평균을 계산하면 바로

$$
\operatorname{tr}(M(z)\Sigma)
$$

가 나오기 때문이다.

---

# 8. 방향별 민감도

Jacobian의 singular value decomposition을

$$
J_g(z)
======

U
\operatorname{diag}(s_1,\ldots,s_r)
V^\top
$$

라고 하자.

Latent perturbation을 오른쪽 singular vector (v_i) 방향으로 가하면

$$
\delta=rv_i
$$

이고,

$$
|g(z+\delta)-g(z)|_2
\approx
rs_i
$$

가 된다.

따라서

* (s_i)가 큰 방향: 작은 latent 변화가 큰 출력 변화로 이어짐
* (s_i)가 작은 방향: latent가 변해도 출력은 거의 변하지 않음

이다.

또한 (v_i)는 단일 latent coordinate가 아니라 여러 coordinate의 조합일 수 있다.

따라서 단순히 각 latent dimension에 하나씩 noise를 넣어보는 coordinate-wise sensitivity 실험만으로는 가장 민감한 방향을 발견하지 못할 수 있다.

진짜 민감한 방향은

$$
J_g(z)^\top J_g(z)
$$

의 eigenvector다.

---

# 9. Generator conditioning 연구

## Is Generator Conditioning Causally Related to GAN Performance?

Odena et al., 2018

이 연구는 generator Jacobian의 singular value 분포가 모델의 품질과 어떤 관계가 있는지를 분석했다.

저자들은 latent 공간의 작은 perturbation에 대해 다음 finite-difference 비율을 측정했다.

$$
Q(z,\delta)
===========

\frac{
|G(z+\delta)-G(z)|_2
}{
|\delta|_2
}
$$

이 값은 국소적인 Jacobian amplification을 근사한다.

연구 결과는 다음과 같다.

* GAN generator의 Jacobian은 학습 중 심하게 ill-conditioned되는 경우가 많았다.
* 소수 방향에서는 작은 변화가 매우 크게 증폭되었다.
* 다른 방향에서는 출력이 거의 변하지 않았다.
* Jacobian conditioning이 나쁜 모델은 FID나 class coverage도 좋지 않은 경향을 보였다.
* Jacobian amplification이 지나치게 커지거나 작아지지 않도록 제한하면 학습 안정성과 품질이 개선되었다.

이 논문은 GAN이 중심이지만 MNIST VAE decoder와도 비교했다.

저자들은 VAE decoder의 Jacobian spectrum이 GAN보다 run 사이에서 더 안정적이고, 전반적인 sensitivity도 더 작게 나타났다고 보고했다.

이 결과는 VAE의 stochastic latent sampling이 decoder 주변을 어느 정도 부드럽게 만들 수 있다는 해석과 연결된다.

---

# 10. VAE가 PCA 방향을 따르는 현상

## Variational Autoencoders Pursue PCA Directions by Accident

Rolinek, Zietlow, Martius, 2019

이 연구는 일반적인 VAE가 왜 latent 공간에서 비교적 정돈된 방향을 학습하는지를 분석했다.

VAE에서는 encoder posterior covariance를 보통 diagonal로 둔다.

$$
q(z\mid x)
==========

\mathcal{N}
\left(
\mu(x),
\operatorname{diag}(\sigma^2(x))
\right)
$$

따라서 각 latent coordinate에는 독립적인 Gaussian noise가 들어간다.

Decoder가 서로 다른 latent coordinate의 변화에 복잡하게 얽혀 반응하면, 이 독립 noise가 reconstruction을 불안정하게 만든다.

그 결과 학습 과정은 decoder의 국소적인 column들이 서로 직교하는 방향으로 움직이는 경향을 갖는다.

즉

$$
J_g(z)^\top J_g(z)
$$

가 어느 정도 diagonal에 가까워지는 경향이 생긴다.

이 연구는 AWGN robustness 자체를 직접 측정한 논문은 아니지만, VAE decoder가 무작위 latent noise에 대해 GAN보다 더 규칙적인 sensitivity를 보일 수 있는 이론적 이유를 제공한다.

---

# 11. Latent manifold를 평평하게 만드는 연구

## Learning Flat Latent Manifolds with VAEs

Chen et al., 2020

이 연구는 decoder가 만드는 metric을 다음과 같은 형태로 만들려고 한다.

$$
J_g(z)^\top J_g(z)
\approx
c(z)I
$$

이 식이 성립하면 어떤 방향으로 동일한 크기의 perturbation을 주더라도 출력 변화량이 거의 같다.

$$
|J_g(z)\delta|_2^2
\approx
c(z)|\delta|_2^2
$$

즉 latent 공간의 방향별 anisotropy가 줄어든다.

AWGN 관점에서는 다음과 같은 의미가 있다.

* 특정한 소수 방향만 극단적으로 민감한 현상이 줄어든다.
* 무작위 noise가 우연히 위험한 방향과 겹쳐 출력이 크게 변할 가능성이 줄어든다.
* Euclidean latent distance가 출력 변화량을 더 잘 반영한다.

다만 (c(z)) 자체가 매우 크다면 모든 방향에서 출력이 크게 변할 수 있다.

따라서 isotropic하다는 것과 robust하다는 것은 완전히 같은 말이 아니다.

* Isotropic: 모든 방향의 sensitivity가 비슷함
* Robust: sensitivity의 절대적인 크기가 작음

---

# 12. 최악의 경우를 보증하는 연구

## Provable Lipschitz Certification for Generative Models

Jordan, Dimakis, 2021

Monte Carlo 방식으로 Gaussian noise를 여러 번 넣어보면 평균적인 출력 변화를 측정할 수 있다.

하지만 이 방법만으로는 매우 드물지만 위험한 방향이 존재하지 않는다고 보장할 수 없다.

이 연구는 생성 모델에 대해 다음과 같은 Lipschitz bound를 계산한다.

$$
|g(z_1)-g(z_2)|*\beta
\leq
L*{\alpha\rightarrow\beta}
|z_1-z_2|_\alpha
$$

따라서 latent perturbation의 크기가 (r) 이하라면

$$
|g(z+\delta)-g(z)|*\beta
\leq
L*{\alpha\rightarrow\beta}r
$$

라는 최악의 경우 보증을 얻을 수 있다.

이 논문은 VAE와 DCGAN architecture에 해당 방법을 적용했다.

Lipschitz certificate는 대체로 보수적일 수 있지만, Monte Carlo 평균으로는 찾기 어려운 극단적인 민감도를 확인할 수 있다는 장점이 있다.

---

# 13. 국소적인 저차원 민감 방향

## Low-Rank Subspaces in GANs

Zhu et al., 2021

이 연구는 전체 이미지가 아니라 특정 관심 영역이 latent 변화에 어떻게 반응하는지를 분석했다.

이미지의 특정 영역을 추출하는 연산을 (R)이라고 하면,

$$
J_R(z)
======

\frac{\partial R(G(z))}{\partial z}
$$

를 계산할 수 있다.

이를 singular value decomposition하면

$$
J_R(z)
======

U_R\Sigma_RV_R^\top
$$

이다.

* (V_R)의 상위 singular vector: 해당 영역을 크게 변화시키는 latent 방향
* 작은 singular value에 해당하는 방향: 해당 영역을 거의 보존하는 방향

연구 결과, 생성 모델의 민감도는 모든 latent 차원에 골고루 퍼져 있기보다 비교적 낮은 차원의 subspace에 집중되는 경우가 많았다.

그리고 이 민감한 방향은 단순한 수치적 현상이 아니라

* 얼굴 표정
* 머리카락
* 배경
* 물체의 위치
* 특정 부위의 모양

등과 같은 의미적 또는 공간적 변화와 연결되었다.

---

# 14. 출력 평균이 아니라 출력 분포의 변화

일반적인 VAE decoder는 하나의 이미지가 아니라 조건부 분포를 정의한다.

$$
p_\theta(x\mid z)
$$

따라서 latent noise가 들어왔을 때 분석해야 할 대상은 decoder 평균

$$
\mu_\theta(z)
$$

의 변화뿐만이 아니다.

Decoder가 예측하는 uncertainty나 scale도 달라질 수 있다.

## Pulling Back Information Geometry

Arvanitidis et al., 2022

이 연구는 decoder output distribution 사이의 차이를 Fisher–Rao geometry로 측정한다.

즉 단순한 픽셀 거리 대신

$$
p_\theta(x\mid z)
$$

와

$$
p_\theta(x\mid z+\delta)
$$

라는 두 확률분포가 얼마나 달라지는지를 분석한다.

두 분포 사이의 작은 KL divergence는 국소적으로 Fisher information에 의한 quadratic form으로 근사된다.

따라서 이 접근은 latent noise가

* 평균 이미지
* 출력 분산
* 전체 조건부 likelihood

를 얼마나 변화시키는지를 함께 다룰 수 있다.

---

# 15. VAE의 국소적 취약성과 geometry

## Adversarial Robustness of VAEs Through the Lens of Local Geometry

Khan, Storkey, 2023

이 논문에서 perturbation은 처음에는 입력 공간에서 시작하지만, encoder를 통과한 뒤 결국 latent representation이 이동하고, 고정된 decoder가 이를 출력으로 변환한다.

따라서 latent displacement가 reconstruction에 미치는 방향별 영향에 대한 중요한 근거를 제공한다.

연구의 핵심 결과는 다음과 같다.

* VAE의 취약성은 isotropic하지 않다.
* 일부 latent 방향은 다른 방향보다 훨씬 위험하다.
* 이러한 취약성은 decoder가 만드는 pullback metric의 eigenvalue spectrum과 연관된다.
* (\beta)-VAE에서 (\beta)를 증가시키면 robustness가 좋아질 수 있지만 reconstruction quality가 나빠질 수 있다.
* 단순히 Jacobian을 작게 만드는 것은 정보를 잃고 출력이 흐릿해지는 방식의 robustness일 수도 있다.

즉 noise에 둔감하다는 사실만으로 좋은 모델이라고 할 수 없다.

---

# 16. Latent perturbation을 직접 비교한 최근 연구

## Latent Diffusion Models with Masked AutoEncoders

2025년 연구

이 연구는 여러 autoencoder 계열에 latent perturbation을 직접 적용하고 reconstruction quality가 어떻게 변하는지를 비교했다.

비교 모델에는 다음이 포함된다.

* Deterministic autoencoder
* Denoising autoencoder
* 일반 VAE
* Stable Diffusion 계열 VAE
* 제안된 masked VAE

연구에서는 여러 비율의 Gaussian latent perturbation을 가한 뒤 reconstruction FID, MSE, PSNR, SSIM, LPIPS 등을 측정했다.

주요 결과는 다음과 같다.

### Deterministic autoencoder

아주 작은 perturbation에서는 clean reconstruction이 우수하지만, noise가 커지면 품질이 빠르게 악화되었다.

### 일반 VAE

Noise가 증가해도 출력 품질 변화는 비교적 작았지만, clean reconstruction 자체의 품질이 좋지 않았다.

즉 decoder가 강건해서라기보다, 애초에 세부 정보를 많이 표현하지 못해 broad latent region이 비슷한 출력으로 mapping되는 것일 수 있다.

### Stable Diffusion VAE와 개선된 probabilistic AE

Clean reconstruction과 noise robustness 사이에서 더 좋은 균형을 보였다.

이 결과는 매우 중요한 점을 보여준다.

> Latent noise에 둔감한 것과 원본을 잘 복원하는 것은 서로 다른 성질이다.

---

# 17. 연구들을 종합했을 때 나타나는 공통 결과

## 17.1 출력 변화는 latent 위치에 따라 크게 달라진다

동일한 variance의 AWGN을 사용해도

$$
D(z)
\approx
\sigma^2|J_g(z)|_F^2
$$

이므로 (z)의 위치에 따라 출력 변화량이 달라진다.

따라서 평균 distortion 하나만 보고해서는 안 된다.

다음과 같은 통계량을 함께 봐야 한다.

* 평균
* 중앙값
* 90% percentile
* 95% percentile
* 99% percentile

평균적으로 안정적이어도 드물게 매우 민감한 latent region이 존재할 수 있다.

---

## 17.2 민감도는 coordinate-wise하지 않고 방향-wise하다

각 latent coordinate를 하나씩 변화시키면

$$
\left|
\frac{\partial g}{\partial z_i}
\right|_2^2
$$

를 측정할 수 있다.

하지만 이는

$$
J_g(z)^\top J_g(z)
$$

의 diagonal 성분만 보는 것이다.

실제 가장 민감한 방향은 여러 coordinate가 결합된 eigenvector일 수 있다.

따라서 dimension별 sensitivity만 측정해서는 충분하지 않다.

---

## 17.3 평균 robustness와 최악의 경우 robustness는 다르다

AWGN은 에너지를 여러 방향에 무작위로 나눈다.

따라서 매우 민감한 방향이 하나뿐이고 latent dimension이 크다면, 대부분의 random noise는 그 방향에 에너지를 조금만 투영할 수 있다.

이 경우 평균 distortion은 작지만 특정 방향의 perturbation은 큰 변화를 만들 수 있다.

그러므로 다음을 따로 측정해야 한다.

$$
\text{평균 AWGN 변화}
$$

$$
\text{AWGN 상위 percentile}
$$

$$
\text{최대 singular direction 변화}
$$

$$
\text{최악의 adversarial 변화}
$$

---

## 17.4 VAE의 stochasticity는 주변을 부드럽게 만들 수 있다

VAE decoder는 학습 중

$$
z
=

\mu(x)+\sigma(x)\odot\epsilon
$$

형태의 여러 주변 latent sample을 반복해서 본다.

따라서 encoder mean 하나만 보는 deterministic AE에 비해 주변 지역에서 smoother한 mapping을 학습할 가능성이 있다.

하지만 이것은 공짜로 얻어지는 robustness가 아니다.

Decoder가 세부 정보를 무시하고 넓은 latent 영역을 비슷한 출력으로 mapping해도 Jacobian은 작아질 수 있다.

따라서 robustness는 반드시 clean reconstruction quality와 함께 평가해야 한다.

---

## 17.5 픽셀 변화와 의미 변화는 다르다

Latent noise가 만들어낸 출력 변화는 여러 공간에서 측정할 수 있다.

### Pixel distortion

$$
|g(z+\varepsilon)-g(z)|_2^2
$$

MSE나 PSNR로 측정한다.

### Structural distortion

SSIM이나 MS-SSIM을 사용한다.

### Perceptual distortion

LPIPS와 같은 deep feature metric을 사용한다.

### Semantic distortion

* Class가 유지되는가?
* 얼굴 identity가 유지되는가?
* Segmentation 결과가 유지되는가?
* 물체의 pose나 속성이 유지되는가?

를 측정한다.

Decoder가 매우 강력하면 noisy latent로부터 여전히 그럴듯한 이미지를 생성할 수 있다.

하지만 그 이미지가 원본과 같은 class나 identity를 유지한다는 보장은 없다.

따라서

$$
\boxed{
\text{Realism}
\neq
\text{Semantic preservation}
\neq
\text{Instance fidelity}
}
$$

이다.

---

# 18. 재현 가능한 실험 방법

## 18.1 기본 실험

고정된 encoder와 decoder를 준비한다.

입력 (x)를 encode하여

$$
z=\mu_\phi(x)
$$

를 얻는다.

VAE sampling noise와 추가 latent noise를 분리해서 보기 위해 먼저 posterior mean을 사용하는 것이 좋다.

Noise가 없는 출력은

$$
x_0=g_\theta(z)
$$

이다.

Noise를 추가한 출력은

$$
x_\sigma
========

g_\theta(z+\varepsilon),
\qquad
\varepsilon\sim\mathcal{N}(0,\sigma^2I)
$$

이다.

---

## 18.2 두 종류의 distortion을 분리해야 한다

### Noise 자체가 만든 출력 변화

$$
D_{\mathrm{channel}}
====================

d(x_\sigma,x_0)
$$

이 값은 latent noise 때문에 출력이 얼마나 달라졌는지를 측정한다.

### 원본 입력으로부터의 최종 reconstruction error

$$
D_{\mathrm{source}}
===================

d(x_\sigma,x)
$$

이 값에는 두 가지가 함께 포함된다.

* 원래 autoencoder의 reconstruction error
* latent noise가 추가로 만든 error

네가 관심을 두는 문제에는 첫 번째 값인

$$
d(g(z+\varepsilon),g(z))
$$

가 더 직접적이다.

---

## 18.3 Noise sweep

다음과 같이 latent standard deviation에 대한 noise 비율을 변화시킬 수 있다.

$$
\frac{\sigma_\varepsilon}{\sigma_z}
\in
\left{
0,,
0.01,,
0.02,,
0.05,,
0.1,,
0.2,,
0.4,,
0.8,,
1.0
\right}
$$

또는 다음과 같은 SNR 범위를 사용할 수 있다.

$$
\mathrm{SNR}_{\mathrm{dB}}
\in
\left{
40,,
30,,
25,,
20,,
15,,
10,,
5,,
0,,
-5
\right}
$$

각 (z)와 noise level마다 최소 20개의 독립적인 noise sample을 사용하는 것이 좋다.

Tail risk를 분석하려면 50~100개의 sample을 사용할 수 있다.

---

# 19. Jacobian 예측과 실제 결과 비교

각 latent point에서 다음 값을 계산한다.

$$
P_{\mathrm{MSE}}(z,\sigma)
==========================

\sigma^2|J_g(z)|_F^2
$$

그리고 Monte Carlo로 실제 출력 distortion을 계산한다.

$$
\widehat{D}_{\mathrm{MC}}(z,\sigma)
===================================

\frac{1}{T}
\sum_{t=1}^{T}
\left|
g(z+\varepsilon_t)-g(z)
\right|_2^2
$$

그 뒤 다음을 비교한다.

* (R^2)
* Spearman correlation
* Median relative error
* 90th percentile relative error

Decoder의 출력 차원이 크면 전체 Jacobian을 직접 계산하기 어렵다.

이때 Hutchinson estimator를 사용할 수 있다.

$$
|J_g(z)|_F^2
============

\mathbb{E}_v
\left[
|J_g(z)^\top v|_2^2
\right]
$$

여기서

$$
\mathbb{E}[vv^\top]=I
$$

를 만족하도록 (v)를 Gaussian 또는 Rademacher random vector로 뽑는다.

---

# 20. 1차 근사가 언제 깨지는가?

Jacobian 예측은 noise가 작을 때만 정확하다.

선형 근사 오차를 다음처럼 정의할 수 있다.

$$
\rho(z,\varepsilon)
===================

\frac{
|g(z+\varepsilon)-g(z)-J_g(z)\varepsilon|_2
}{
|g(z+\varepsilon)-g(z)|_2+\epsilon_0
}
$$

Noise가 작으면 (\rho)가 작다.

Noise가 커지면서 (\rho)가 갑자기 증가한다면, 그 지점부터는

* activation boundary를 넘거나
* decoder curvature의 영향이 커지거나
* semantic region이 바뀌면서

1차 Jacobian 분석이 충분하지 않다는 뜻이다.

---

# 21. 방향별 실험

Jacobian의 상위 singular vector를 구해 다음과 같은 방향으로 동일한 크기의 perturbation을 가한다.

$$
v_1,\quad v_2,\quad v_5,\quad v_{10}
$$

그리고 다음과 비교한다.

* Random Gaussian direction
* Random sphere direction
* 각 coordinate axis
* 가장 작은 singular value 방향

각 방향에서 실제 출력 변화

$$
|g(z+rv_i)-g(z)|_2^2
$$

와 1차 예측

$$
r^2s_i^2
$$

를 비교한다.

그 후 pixel, perceptual, semantic metric을 모두 측정하면 다음을 알 수 있다.

> 픽셀상으로 가장 민감한 방향이 실제로도 class나 identity를 바꾸는 의미적으로 위험한 방향인가?

두 방향이 일치하지 않을 가능성이 높다.

---

# 22. Pixel Jacobian과 semantic Jacobian

일반적인 decoder geometry는

$$
M_{\mathrm{pixel}}(z)
=====================

J_g(z)^\top J_g(z)
$$

를 사용한다.

하지만 perceptual feature extractor를 (\phi(x))라고 하면 perceptual geometry는

$$
M_{\mathrm{perceptual}}(z)
==========================

J_g(z)^\top
J_\phi(g(z))^\top
J_\phi(g(z))
J_g(z)
$$

로 정의할 수 있다.

Classifier나 task network를 (h(x))라고 하면 task geometry는

$$
M_{\mathrm{task}}(z)
====================

J_g(z)^\top
J_h(g(z))^\top
J_h(g(z))
J_g(z)
$$

가 된다.

각 metric은 서로 다른 질문에 답한다.

* (M_{\mathrm{pixel}}): 픽셀이 얼마나 변하는가?
* (M_{\mathrm{perceptual}}): 사람이 느끼는 시각적 특징이 얼마나 변하는가?
* (M_{\mathrm{task}}): class, identity, segmentation 결과가 얼마나 변하는가?

---

# 23. Semantic class가 바뀔 확률

Classifier logit을 (h(g(z)))라고 하자.

현재 class가 (y)이고 경쟁 class가 (k)라면 margin을

$$
m_{y,k}(z)
==========

h_y(g(z))-h_k(g(z))
$$

로 정의한다.

국소적으로 decision boundary까지의 latent 거리는

$$
r_{\mathrm{sem}}
\approx
\frac{
m_{y,k}(z)
}{
|\nabla_zm_{y,k}(z)|_2
}
$$

로 근사할 수 있다.

Gaussian noise에 의해 두 class의 순서가 뒤집힐 확률은 1차 근사에서

$$
P(\text{class flip})
\approx
\Phi
\left(
------

\frac{
m_{y,k}(z)
}{
\sqrt{
\nabla_zm_{y,k}(z)^\top
\Sigma
\nabla_zm_{y,k}(z)
}
}
\right)
$$

로 표현할 수 있다.

여기서 (\Phi)는 표준정규분포의 누적분포함수다.

이 식은 decoder geometry와 실제 class 보존 확률을 연결한다.

---

# 24. 현재 문헌의 가장 큰 공백

관련 이론은 상당히 잘 정리되어 있지만, 다음 조건을 모두 충족하는 표준화된 실험 연구는 많지 않다.

1. 이미 학습된 pretrained model을 고정한다.
2. VAE, deterministic AE, GAN, modern image tokenizer를 같은 조건에서 비교한다.
3. Latent scale을 표준화한다.
4. 동일한 AWGN SNR sweep을 적용한다.
5. Pixel, perceptual, semantic distortion을 모두 측정한다.
6. Jacobian 예측과 실제 Monte Carlo distortion을 비교한다.
7. 평균뿐 아니라 tail risk와 최악의 방향을 측정한다.

기존 연구들은 대체로 다음 중 하나에 집중한다.

* Latent geometry의 이론적 분석
* GAN의 singular direction 분석
* VAE robustness 분석
* Noise를 포함해 모델을 재학습하는 통신 연구
* 특정 모델에서의 latent perturbation 실험

따라서 **고정된 pretrained generative decoder에 동일한 AWGN을 가하고, Jacobian geometry가 실제 출력 변화와 semantic failure를 얼마나 정확히 예측하는지 비교하는 연구**는 여전히 가치가 있다.

---

# 25. 가장 유망한 즉시 연구 주제

다음과 같은 실험이 가장 직접적이다.

고정된 네 가지 모델을 준비한다.

* Convolutional VAE
* Deterministic AE
* StyleGAN2
* Stable Diffusion VAE

각 모델에서 약 10,000개의 latent point를 준비하고, 각 point마다 20개의 AWGN sample을 적용한다.

측정할 값은 다음과 같다.

* MSE
* PSNR
* SSIM
* LPIPS
* Class 또는 identity preservation
* (|J_g(z)|_F^2)
* 상위 singular value
* Latent density

가장 중요한 plot은 다음 세 가지다.

### Jacobian 예측과 실제 distortion

$$
\text{Observed MSE}
\quad\text{vs.}\quad
\sigma^2|J_g(z)|_F^2
$$

### SNR에 따른 의미 보존

$$
\text{Semantic survival probability}
\quad\text{vs.}\quad
\mathrm{SNR}
$$

### 무작위 noise와 가장 민감한 방향 비교

$$
\text{AWGN 99th percentile}
\quad\text{vs.}\quad
r,s_{\max}(J_g(z))
$$

이 실험은 다음 질문에 직접 답할 수 있다.

1. Decoder Jacobian만으로 AWGN-induced output distortion을 얼마나 정확히 예측할 수 있는가?
2. VAE가 deterministic AE보다 본질적으로 noise에 더 강한가, 아니면 단지 세부 정보를 덜 보존하기 때문에 둔감한 것인가?
3. Pixel상 민감한 방향과 class나 identity를 바꾸는 semantic failure 방향은 일치하는가?
4. Latent density가 낮은 영역이 더 불안정한가?
5. 평균적으로 안정적인 decoder에도 소수의 매우 위험한 방향이 존재하는가?

---

# 26. 주요 논문 정리

## 직접적인 이론적 기반

1. **Latent Space Oddity: On the Curvature of Deep Generative Models**
   Decoder Jacobian이 latent 공간의 국소적인 거리와 curvature를 결정한다는 관점을 제시한다.

2. **Is Generator Conditioning Causally Related to GAN Performance?**
   Generator Jacobian의 singular spectrum과 모델 품질 및 안정성의 관계를 분석한다.

3. **Variational Autoencoders Pursue PCA Directions by Accident**
   VAE의 stochastic sampling이 decoder 방향을 국소적으로 직교시키는 경향을 설명한다.

4. **Learning Flat Latent Manifolds with VAEs**
   (J^\top J)를 scaled identity에 가깝게 만들어 방향별 sensitivity 차이를 줄인다.

5. **Provable Lipschitz Certification for Generative Models**
   Latent perturbation이 출력에서 최악의 경우 얼마나 증폭될 수 있는지를 보증한다.

## 방향별·의미적 민감도

6. **Low-Rank Subspaces in GANs**
   출력의 특정 영역이나 의미적 속성을 변화시키는 latent subspace를 Jacobian SVD로 찾는다.

7. **Do Not Escape From the Manifold**
   Latent space의 민감한 principal direction이 위치에 따라 회전하며, 하나의 전역 방향이 모든 위치에서 동일하게 작동하지 않음을 보여준다.

8. **Pulling Back Information Geometry**
   Latent perturbation이 decoder의 출력 확률분포 전체를 얼마나 변화시키는지를 Fisher–Rao geometry로 분석한다.

9. **Adversarial Robustness of VAEs Through the Lens of Local Geometry**
   VAE의 방향별 취약성을 stochastic pullback metric의 eigenvalue spectrum과 연결한다.

## 직접적인 latent perturbation 비교

10. **Latent Diffusion Models with Masked AutoEncoders**
    여러 autoencoder 계열에 Gaussian latent perturbation을 가하고 reconstruction 및 perceptual degradation을 비교한다.

---

# 27. 최종 결론

Latent vector에 AWGN이 추가되었을 때 출력이 얼마나 변하는지는 기본적으로 다음 세 요소에 의해 결정된다.

$$
\boxed{
\text{Noise 크기}
+
\text{현재 latent 위치}
+
\text{Decoder의 방향별 Jacobian}
}
$$

작은 noise 영역에서 평균 출력 MSE는

$$
\boxed{
\mathbb{E}
\left[
|g(z+\varepsilon)-g(z)|_2^2
\mid z
\right]
\approx
\sigma^2|J_g(z)|_F^2
}
$$

로 근사할 수 있다.

하지만 이 값만으로는 충분하지 않다.

* 가장 위험한 방향은 (s_{\max}(J_g(z)))가 결정한다.
* 민감한 방향은 latent 위치에 따라 달라진다.
* Pixel상 변화와 semantic 변화는 다를 수 있다.
* 큰 noise에서는 Hessian과 decoder curvature가 중요해진다.
* Noise에 둔감한 모델이 반드시 좋은 representation을 가진 것은 아니다.

따라서 이 문제를 제대로 평가하려면 다음 네 가지를 함께 사용해야 한다.

1. AWGN Monte Carlo sweep
2. Jacobian 기반 평균 distortion 예측
3. Singular-vector 방향별 perturbation
4. Pixel·perceptual·semantic metric의 분리 평가

현재 문헌에는 이론적 도구와 개별 실험은 충분히 존재하지만, **동일한 조건의 frozen decoder들을 대상으로 AWGN, Jacobian spectrum, semantic failure를 통합적으로 비교한 표준 연구는 아직 부족하다.**

바로 이 부분이 네가 생각하고 있는 연구 방향의 가장 분명한 빈틈이다.


</p>


</section>