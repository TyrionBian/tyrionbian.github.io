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

We use Linear Regression as basic background algorithm. We will also use $$X$$ denote the space of input values, and $$Y$$ the space of output values. In this example, $$X = Y = \mathbb{R}$$.
As an initial choice, let's say we decide to approximate $$y$$ as a linear function of $$x$$:

$$ h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 $$

Here, the $$\theta_i$$'s are the **parameters**(also called weights) parameterizing the space of linear functions mapping from $$X$$ to $$Y$$.When there is no risk of confusion, we will drop the $$\theta$$ subscript in $$h_\theta(x)$$, and write it more simply as $$h(x)$$. To simplify our notation, we also introduce the convention of letting $$x_\theta = 1$$ (this is the **intercept term**), so that
\$$ 
h(x) = \sum_{i=0}^n \theta_ix_i = \theta^\mathrm{T} x
$$

where on the right-hand side above we are viewing $$\theta$$ and $$x$$ both as vectors, and here $$n$$ is the number of input variables (not counting $$x_0$$).

Now, given a training set, how do we pick, or learn, the parameters $$\theta$$? One reasonable method seems to be to make $$h(x)$$ close to y, at least for the training examples we have. To formalize this, we will define a function that measures, for each value of the $$\theta$$’s, how close the $$h(x^{(i)})$$’s are to the corresponding $$y^{(i)}$$’s. We define the **cost function**:
\$$
J(\theta) = \frac{1}{2} \sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2
$$

If you’ve seen linear regression before, you may recognize this as the familiar least-squares cost function that gives rise to the **ordinary least squares** regression model. Whether or not you have seen it previously, let’s keep going, and we’ll eventually show this to be a special case of a much broader family of algorithms.


### LMS algorithm

We want to choose θ so as to minimize $$J(θ)$$. Let’s consider the gradient descent algorithm, which starts with some initial θ, and repeatedly performs the update:
\$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
$$

(This update is simultaneously performed for all values of $$j = 0, ..., n$$.)
Here, $$\alpha$$ is called the **learning rate**. This is a varu natural algorithm that repeatedly takes a step in the direction of steepest decrease of $$J$$.

In order to implement this algorithm, we have to work out what is the partical derivative term on the right hand side. Let's first work it out for the case of if we have only one training example $$(x,y)$$, so that we can neglect the sum in the definition of $$J$$. We have:
\$$
\begin{equation}
 \begin{split}
\frac{\partial}{\partial \theta_j} J(\theta) &=
\frac{\partial}{\partial \theta_j} \frac{1}{2}(h_\theta(x)-y)^2  \\
&=2*\frac{1}{2}(h_\theta(x)-y) \cdot \frac{\partial}{\partial \theta_j}(h_\theta(x)-y)  \\
&=(h_\theta(x)-y) \cdot \frac{\partial}{\partial \theta_j}\Bigl(\sum_{i=0}^n\theta_ix_i-y\Bigr)  \\
&=(h_\theta(x)-y)x_j
 \end{split}
\end{equation}
$$


For a single training example, this gives the update rule:
\$$
\theta_j := \theta_j + \alpha \bigl( y^{(i)}-h_\theta(x^{(i)}) \bigr) x_j^{(i)}
$$

The rule is called the **LMS** update rule (LMS stands for “least mean squares”),
and is also known as the **Widrow-Hoff** learning rule.
\$$
\begin{equation}
 \begin{aligned}
\frac{\partial}{\partial \theta_j} J(\theta) &=
\frac{\partial}{\partial \theta_j} \frac{1}{2}(h_\theta(x)-y)^2 \newline
&=2*\frac{1}{2}(h_\theta(x)-y) \cdot \frac{\partial}{\partial \theta_j}(h_\theta(x)-y) \newline
&=(h_\theta(x)-y) \cdot \frac{\partial}{\partial \theta_j}\Bigl(\sum_{i=0}^n\theta_ix_i-y\Bigr) \newline
&=(h_\theta(x)-y)x_j
 \end{aligned}
\end{equation}
$$











