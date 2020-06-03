---
layout:            post
title:             "Pytorch search image"
date:              2020-04-12
tag:               Pytorch
category:          Pytorch
author:            tianliang
math:              true
---
## Pytorch search image

We use Linear Regression as basic background algorithm. We will also use $$\mathcal{X}$$ denote the space of input values, and $$\mathcal{Y}$$ the space of output values. In this example, $$\mathcal{X} = \mathcal{Y} = \mathbb{R}$$.
As an initial choice, let's say we decide to approximate $$y$$ as a linear function of $$x$$:
\$$ h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 $$
