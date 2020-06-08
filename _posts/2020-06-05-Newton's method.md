---
layout:            post
title:             "Newton's method"
date:              2020-06-05
tag:               Machine Learning Basic
category:          Machine Learning Basic
author:            tianliang
math:              true
---
## Newton's method
Returning to logistic regression, Let’s now talk about a different algorithm for maximizing $$\ell(θ)$$.
To get us started, let's consider Newton's method for finding a zero of a function. Specifically, suppose we have some function $$f : \mathbb{R} \to \mathbb{R}$$, and we want to find a value of $θ$ so that $$f(θ)=0$$. Here, $$θ \in \mathbb{R}$$ is a real number. 

Newton’s method performs the following update:
\$$
θ := θ - \frac{f(θ)}{f'(θ)}
$$

<figure>
   <img src="{{ "/images/NewtonIteration_Ani.gif" | absolute_url }}" />
   <figcaption>Newton's method(from wikipedia)</figcaption>
</figure>

Newton’s method gives a way of getting to $f(θ) = 0$. What if we want to use it to maximize some function $\ell$? The maxima of $\ell$ corespond to points where its first deriative $\ell'(θ)$ is zero. So letting $f(θ)=\ell'(θ)$, we can obtain:
\$$
θ := θ - \frac{\ell'(θ)}{\ell''(θ)}
$$

We know that $θ$ is vector-valued, so we need to generalize Newton's method to this setting. The generalization of Newton’s method to this multidimensional setting (also called the Newton-Raphson method) is given by:
\$$
θ := θ - \rm{H}^{-1}\nabla_θ \ell(θ)
$$

$\nabla_θ \ell(θ)$ is the partial derivatives of $\ell(θ)$ with respect to the $θ_i$'s; $\mathrm{H}$ is an n-by-n matrix (actually, n + 1-by-n + 1, assuming that we include the intercept term) called the Hessian, whose entries are
given by:
\$$
\rm{H}_{ij} = \frac{\partial^2 \ell{θ}}{\partial θ_i \partial θ_j}
$$

Newton’s method typically enjoys faster convergence than (batch) gradient descent, and requires many fewer iterations to get very close to the minimum. 

Since Newton's method needs to find and invert an $n-by-n$ Hessian, so one iteration of Newton's can be more expensive than on iteration of gradient descent. But in practical applications, it is usually much faster overall,, because $n$ is too large.

When Newton’s method is applied to maximize the logistic regression log likelihood function $\ell(θ)$, the resulting method is also called **Fisher scoring**.
