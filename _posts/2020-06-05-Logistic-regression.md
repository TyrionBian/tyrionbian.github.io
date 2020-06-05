---
layout:            post
title:             "Logistic regression"
date:              2020-06-05
tag:               Machine Learning Basic
category:          Machine Learning Basic
author:            tianliang
math:              true
---
## Logistic regression
We will focus on binary classification problem for Logistic regression in which $$y$$ can take on only two values, 0 and 1. (Most of we say here also can generalize to the multiple-class case.) For instance, if we are trying to build a spam classifier for email, then $$x^{(i)}$$ may be some features of a piece of email, and $$y$$ may be 1 if it is a piece of spam mail, and 0 otherwise. 0 is also called the **negative class**, and 1 the **positive class**, and they are sometimes also denoted by the symbols “-” and “+.” Given $$x^{(i)}$$, the corresponding $$y^{(i)}$$ is also called the label for the training example.

We can try to use linear regression algorithm to predict $$y$$ given $$x$$. But it always performs very poorly, and also doesnot make sense for $$h_\theta(x)$$ to take values larger than 1 or smaller than 0 when we know that $$y\in\{0,1\}$$.

To fix this problem, we change our hypotheses $$h_\theta(x).
\$$
h_\theta(x) = g(\theta^{\mathrm{T}}x) = \frac{1}{1+\mathrm{e}^{-\theta^{\mathrm{T}}x}}
$$
where
\$$
g(z)=\frac{1}{1+\mathrm{e}^{-z}}
$$
is called the **logistic function** or the **sigmod function**. Here is a plot showing $$g(z)$$:

<figure>
<img src="{{ "/images/Sigmoid_function_01.png" | absolute_url }}" />
<figcaption>sigmod function</figcaption>
</figure>

$$g(z)$$, and hence also h(x), is always bounded between 0 and 1. And we are keeping the convention of letting $$x_0=1$$, so that $$\theta^{\rm{T}}x=\theta_0+\sum_{j=1}^n\theta_j x_j$$.

Before we moving on, here is a useful property of the derivative of the sigmoid function, which we write as $$g'$$:

\$$
g'(z) &= \frac{d}{dz} \frac{1}{1+e^{-z}} \newline
&= -\frac{1}{(1+e^{-z})^2} (-e^{-z}) \newline
&=\frac{e^{-z}}{(1+e^{-z})^2} \newline
&=\frac{1+e^{-z}-1}{(1+e^{-z})^2} \newline
&=\frac{1}{1+e^{-z}}(1-\frac{1}{1+e^{-z}}) \newline
&=g(z)(1-g(z))
$$

Logistic regression can be derived as the maximum likelihood estimator under a set of assumptions.Let us assume that:

\$$
P(y=1 | x;\theta) &= h_\theta(x) \newline
P(y=0 | x:\theta) &= 1 - h_\theta(x).
$$

We also can written like this:
\$$
p(y | x;\theta) = (h_\theta(x))^y(1-h_\theta(x))^{1-y}.
$$

Assuming that the $$m$$ training examples where generated independently, we can write the likelihood of the parameters like this:
\$$
L(\theta) &= p(\vec{y} | X;\theta) \newline
&=\prod_{i=1}^m p(y^{(i)} | x^{(i)};\theta) \newline
&=\prod_{i=1}^m(h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}}
$$

We can maximize the log likelihood:
\$$
\ell(\theta) = logL(\theta) = \sum_{i=1}^m y^{(i)} log(h(x^{(i)})) + (1-y^{(i)}) log(1-h(x^{(i)}))
$$

To maximize the likelihood, we can use gradient ascent.Firstly, let's start by working with just one training example $$(x,y)$$, and take derivatives to derive the stochastic gradient ascent rule:
\$$
\nabla \ell(\theta) &= \frac{\partial}{\partial \theta_j} \ell(\theta) \newline
&= \frac{\partial \ell(\theta)}{\partial g(\theta^{\mathrm{T}}x)} \cdot \frac{\partial g(\theta^{\mathrm{T}}x)}{\partial \theta^{\mathrm{T}}x} \cdot \frac{\partial \theta^{\mathrm{T}}x}{\partial \theta_j} \newline
&= \left( y \frac{1}{g(\theta^{\mathrm{T}}x)} - (1-y) \frac{1}{1-g(\theta^{\mathrm{T}}x)} \right) \cdot g(\theta^{\mathrm{T}}x)(1-g(\theta^{\mathrm{T}}x)) \cdot \frac{\partial}{\partial \theta_j}\theta^{\mathrm{T}}x \newline 
&= (y(1-g(\theta^{\mathrm{T}}x)) - (1-y)g(\theta^{\mathrm{T}}x))x_j \newline 
&= (y-h_\theta(x))x_j
$$


