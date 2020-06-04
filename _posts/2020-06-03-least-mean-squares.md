---
layout:            post
title:             "Least mean squares"
date:              2020-06-03
tag:               Machine Learning Basic
category:          Machine Learning Basic
author:            tianliang
math:              true
---
## Least mean squares

- TOC
{:toc}

We use Linear Regression as basic background algorithm. We will also use $$\mathcal{X}$$ denote the space of input values, and $$\mathcal{Y}$$ the space of output values. In this example, $$\mathcal{X} = \mathcal{Y} = \mathbb{R}$$.
As an initial choice, let's say we decide to approximate $$y$$ as a linear function of $$x$$:
\$$ h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 $$

Here, the $$\theta_i$$'s are the **parameters**(also called weights) parameterizing the space of linear functions mapping from $$\mathcal{X}$$ to $$\mathcal{Y}$$.When there is no risk of confusion, we will drop the $$\theta$$ subscript in $$h_\theta(x)$$, and write it more simply as $$h(x)$$. To simplify our notation, we also introduce the convention of letting $$x_\theta = 1$$ (this is the **intercept term**), so that
\$$ 
h(x) = \sum_{i=0}^n \theta_ix_i = \theta^\mathrm{T} x
$$

where on the right-hand side above we are viewing $$\theta$$ and $$x$$ both as vectors, and here $$n$$ is the number of input variables (not counting $$x_0$$).

Now, given a training set, how do we pick, or learn, the parameters $$\theta$$? One reasonable method seems to be to make $$h(x)$$ close to y, at least for the training examples we have. To formalize this, we will define a function that measures, for each value of the $$\theta$$’s, how close the $$h(x^{(i)})$$’s are to the corresponding $$y^{(i)}$$’s. We define the **cost function**:
\$$
J(\theta) = \frac{1}{2} \sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2
$$

If you’ve seen linear regression before, you may recognize this as the familiar least-squares cost function that gives rise to the **ordinary least squares** regression model. Whether or not you have seen it previously, let’s keep going, and we’ll eventually show this to be a special case of a much broader family of algorithms.


### 1. LMS algorithm

We want to choose θ so as to minimize $$J(θ)$$ [^1]. Let’s consider the gradient descent algorithm, which starts with some initial θ, and repeatedly performs the update:
\$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
$$

(This update is simultaneously performed for all values of $$j = 0, ..., n$$.)
Here, $$\alpha$$ is called the **learning rate**. This is a varu natural algorithm that repeatedly takes a step in the direction of steepest decrease of $$J$$.

In order to implement this algorithm, we have to work out what is the partical derivative term on the right hand side. Let's first work it out for the case of if we have only one training example $$(x,y)$$, so that we can neglect the sum in the definition of $$J$$. We have:
\$$
\begin{equation}
 \begin{aligned}
\frac{\partial}{\partial \theta_j} J(\theta) &=
\frac{\partial}{\partial \theta_j} \frac{1}{2}(h_\theta(x)-y)^2 \newline
&=2*\frac{1}{2}(h_\theta(x)-y) \cdot \frac{\partial}{\partial \theta_j}(h_\theta(x)-y) \newline
&=(h_\theta(x)-y) \cdot \frac{\partial}{\partial \theta_j}\left(\sum_{i=0}^n\theta_ix_i-y\right) \newline
&=(h_\theta(x)-y)x_j
 \end{aligned}
\end{equation}
$$

For a single training example, this gives the update rule:
\$$
\theta_j := \theta_j + \alpha \left( y^{(i)}-h_\theta(x^{(i)}) \right) x_j^{(i)}
$$

The rule is called the **LMS** update rule (LMS stands for “least mean squares”),
and is also known as the **Widrow-Hoff** learning rule.

### 2. Least squares algorithm
Given a training set, define the **design matrix** $$X$$ to be the $$m-by-n$$
matrix (actually $$m-by-n + 1$$, if we include the intercept term) that contains the training examples’ input values in its rows:
\$$
X = 
\begin{bmatrix}
 — (x^{(1)})^\mathrm{T} — \newline
 — (x^{(2)})^\mathrm{T} — \newline
 \vdots \newline
 — (x^{(m)})^\mathrm{T} — \newline
\end{bmatrix}
$$

Also, let $$\vec{y}$$ be the $$m$$-dimensional vector containing all the target values from
the training set:
\$$
\vec{y} = 
\begin{bmatrix}
 y^{(1)} \newline
 y^{(2)} \newline
 \vdots \newline
 y^{(m)} \newline
\end{bmatrix}
$$

Now, since $$h_\theta(x^{(i)}) = (x^{(i)})^\mathrm{T} \theta$$, we can easily verify that
\$$
\begin{equation}
 \begin{aligned}
X\theta-\vec{y} &=
\begin{bmatrix}
 (x^{(1)})^\mathrm{T}\theta \newline
 \vdots \newline
 (x^{(m)})^\mathrm{T}\theta \newline
\end{bmatrix} -
\begin{bmatrix}
 y^{(1)} \newline
 \vdots \newline
 y^{(m)} \newline
\end{bmatrix} 
\newline
&=
\begin{bmatrix}
 h_\theta(x^{(1)})-y^{(1)} \newline
 \vdots \newline
 h_\theta(x^{(m)})-y^{(m)} \newline
\end{bmatrix}
 \end{aligned}
\end{equation}
$$

Thus, using the fact that for a vector $$z$$, we have that $$z^\mathrm{T} z = \sum_i z_i^2$$:
\$$
\begin{equation}
 \begin{aligned}
\frac{1}{2}(X\theta-\vec{y})^\mathrm{T}(X\theta-\vec{y}) &= 
\frac{1}{2}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2 \newline 
&= J(\theta)
 \end{aligned}
\end{equation}
$$

Finally, to minimize $$J$$, let’s find its derivatives with respect to $$θ$$.We find that [^2]
\$$
\begin{equation}
 \begin{aligned}
\nabla_\theta J(\theta) &= 
\nabla_\theta \frac{1}{2} (X\theta-\vec{y})^\mathrm{T} (X\theta-\vec{y}) \newline 
&= \frac{1}{2} \nabla_\theta (\theta^\mathrm{T} X^\mathrm{T} - \vec{y}^\mathrm{T})(X\theta-\vec{y}) \newline
&= \frac{1}{2} \nabla_\theta (\theta^\mathrm{T} X^\mathrm{T} X \theta - \theta^\mathrm{T} X^\mathrm{T} \vec{y}^\mathrm{T} + \vec{y}^\mathrm{T}\vec{y} - \vec{y}^\mathrm{T} X \theta) \newline
&= \frac{1}{2} (2X^\mathrm{T} X \theta - X^\mathrm{T} \vec{y}^\mathrm{T} + 0 - X^\mathrm{T} \vec{y}) \newline
&= X^\mathrm{T} X \theta - X^\mathrm{T} \vec{y}
 \end{aligned}
\end{equation}
$$

To minimize $$J$$, we set its derivatives to zero, the value of $$θ$$ that minimizes $$J(θ)$$ is given in closed form by the equation
\$$
\theta = (X^\mathrm{T} X)^{-1} X^\mathrm{T} \vec{y}
$$

### 3. Probabilistic interpretation

Let us assume that the target variables and the inputs are related via the equation [^3]
\$$
y^{(i)}=\theta^{\mathrm{T}}x^{(i)}+\epsilon^{(i)},
$$

where $$\epsilon^{(i)}$$ is an error term that captures either unmodeled effects or random noise.

Let us further assume that the $$\epsilon^{(i)}$$ are distributed IID (independently and identically distributed) according to a Gaussian distribution (also called a Normal distribution) with mean zero and some variance $$\sigma^2$$.We can write this assumption as "$$\epsilon^{(i)} \sim \mathcal{N}(0,\sigma^2)$$".  The density of $$\epsilon^{(i)}$$ is given by
\$$
p\left( \epsilon^{(i)} \right) = 
\frac{1}{\sqrt{2\pi} \sigma} exp \left( -\frac{(\epsilon^{(i)})^2}{2\sigma^2} \right).
$$

This implies that
\$$
p\left( y^{(i)}|x^{(i)};\theta \right) = 
\frac{1}{\sqrt{2\pi} \sigma} exp \left( -\frac{(y^{(i)}-\theta^\mathrm{T}x^{(i)})^2}{2\sigma^2} \right).
$$

The notation "
$$
p(y^{(i)}|x^{(i)};\theta)
$$" indicates that this is the distribution of $$y^{(i)}$$ given $$x^{(i)}$$ and parameterized by $$\theta$$.Note that we should not condition on "
$$
\theta(p(y^{(i)}|x^{(i)};\theta))
$$", since $$\theta$$ is not a random variable. We can also write the distribution of $$y^{(i)}$$ as 
$$
y^{(i)}|x^{(i)};\theta \sim \mathcal{N}(\theta^\mathrm{T}x^{(i)}, \sigma^2)
$$.

Given $$X$$ (the design matrix, which contains all the $$x^{(i)}$$’s) and $$\theta$$, what is the distribution of the $$y^{(i)}$$’s?  The probability of the data is given by $$p(\vec{y}|X;\theta)$$. This quantity is typically viewed a function of $$\vec{y}$$ (and perhaps $$X$$), for a fixed value of $$\theta$$. When we wish to explicitly view this as a function of $$\theta$$, we will instead call it the **likelihood function**:
\$$
L(\theta) = L(\theta;X,\vec{y})=p(\vec{y}|X;\theta).
$$

Note that by the independence assumption on the $$\epsilon^{(i)}$$’s (and hence also the $$y^{(i)}$$’s given the $$x^{(i)}$$’s), this an also be written
\$$
\begin{equation}
 \begin{aligned}
L(\theta) &= \prod_{i=1}^m p(y^{(i)}|x^{(i)};\theta) \newline
&= \prod_{i=1}^m \frac{1}{\sqrt{2\pi} \sigma} exp \left( -\frac{(y^{(i)}-\theta^\mathrm{T}x^{(i)})^2}{2\sigma^2} \right).
 \end{aligned}
\end{equation}
$$

Now, given this probabilistic model ralating the $$y^{(i)}$$'s and $$x^{(i)}$$'s, what is a reasonable way of choosing our best guess of the parameters $$\theta$$? The principal of **maximum likelihood** says that we should choose $$\theta$$ so as to make the data as high probability as possible. That is to say, we should choose $$\theta$$ to maximize $$L(\theta)$$.

For the convenience of calculation, we maximize the **log likelihood** $$\ell(\theta)$$:
\$$
\begin{equation}
 \begin{aligned}
\ell(\theta) &= logL(\theta) \newline
&= log\prod_{i=1}^m \frac{1}{\sqrt{2\pi} \sigma} exp \left( -\frac{(y^{(i)}-\theta^\mathrm{T}x^{(i)})^2}{2\sigma^2} \right) \newline
&= \sum_{i=1}^m log \frac{1}{\sqrt{2\pi} \sigma} exp \left( -\frac{(y^{(i)}-\theta^\mathrm{T}x^{(i)})^2}{2\sigma^2} \right) \newline
&= m log\frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{2\sigma^2} \sum_{i=1}^m (y^{(i)}-\theta^\mathrm{T}x^{(i)})^2.
 \end{aligned}
\end{equation}
$$

We know that, maximizing $$\ell(\theta)$$ gives the same result as minimizing
\$$
\frac{1}{2}\sum_{i=1}^m (y^{(i)}-\theta^\mathrm{T}x^{(i)})^2,
$$

which we recognize to be $$J(\theta)$$, our original least-squares cost function.



## Reference
[^1]: [CS229: Machine Learning](http://cs229.stanford.edu/).
[^2]: MatrixCookBook.
[^3]: Bishop C M. Pattern recognition and machine learning[M]. springer, 2006.
