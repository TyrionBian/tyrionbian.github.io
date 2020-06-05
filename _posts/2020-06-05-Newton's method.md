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
Returning to logistic regression, Let’s now talk about a different algorithm for maximizing $$\ell(\theta)$$.
To get us started, let's consider Newton's method for finding a zero of a function. Specifically, suppose we have some function $$f : \mathbb{R} \to \mathbb{R}$$, and we want to find a value of $\theta$ so that $$f(\theta)=0$$. Here, $$\theta \in \mathbb{R}$$ is a real number. 

<figure>
   <img src="{{ "/images/NewtonIteration_Ani.gif" | absolute_url }}" />
   <figcaption>Newton's method</figcaption>
</figure>


