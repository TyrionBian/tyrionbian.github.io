---
layout:            post
title:             "Support Vector Machine"
date:              2020-06-09
tag:               Machine Learning Basic
category:          Machine Learning Basic
author:            tianliang
math:              true
---
## Support Vector Machine

To tell the SVM story, we’ll need to first talk about margins and the idea of separating data with a large “gap.” Next, we’ll talk about the optimal margin classifier, which will lead us into a digression on Lagrange duality. We’ll also see kernels, which give a way to apply SVMs efficiently in very high dimensional (such as infinitedimensional) feature spaces, and finally, we’ll close off the story with the SMO algorithm, which gives an efficient implementation of SVMs.
### 1. Margins
We begin our discussion of support vector machines by returning to the two-class classification problem using linear models of the form:
\$$
y({\bf x}) = {\bf w}^{\rm{T}}\phi({\bf x}) + b .
$$

where $\phi({\bf x})$ denotes a fixed feature-space transformation, and we have made the bias parameter b explicit. The training dataset comprises N input vectors ${\bf x}_1, ..., {\bf x}_N$, with corresponding target values $y_1, ..., y_N$ where $y_n \in \{-1,1\}$, and new data points x are classifed according to the sign of $y({\bf x})$.

$y({\bf x}_n) > 0$ for points having $t_n = +1$; 
$y({\bf x}_n) < 0$ for points having $t_n = −1$; 
so that $t_ny({\bf x}_n) > 0$ for all training data points. 

The support vector machine approaches this problem through the concept of the margin, which is defined to be the smallest distance between the decision boundary and any of the samples.

If the training data is linearly separable, we can select two parallel hyperplanes that separate the two classes of data, so that the distance between them is as large as possible. The region bounded by these two hyperplanes is called the "margin", and the maximum-margin hyperplane is the hyperplane that lies halfway between them. 

<figure>
   <img src="{{ "/images/SVM_margin.png" | absolute_url }}" />
   <figcaption>SVM_margin(from wikipedia)</figcaption>
</figure>

Maximum-margin hyperplane and margins for an SVM trained with samples from two classes. Samples on the margin are called the support vectors.

the distance of a point ${\bf x}_n$ to the hyperplane defined by $y({\bf x}) = 0$ is given by:
\$$
\frac{t_n y({\bf x})}{\| {\bf w} \|} = \frac{t_n ({\bf w}^{\rm{T}}\phi({\bf x}) + b)}{\| {\bf w} \|}
$$

We note that if we make the rescaling $w → \kappa w$ and $b → \kappa b$,
then the distance from any point xn to the decision surface, given by $t_n y({\bf x})/\| {\bf w} \|$ is unchanged. We can use this freedom to set:
\$$
t_n ({\bf w}^{\rm{T}}\phi({\bf x}) + b) = 1
$$

for the point that is closest to the surface. In this case, all data points will satisfy the constraints
\$$
t_n ({\bf w}^{\rm{T}}\phi({\bf x}) + b) >= 1,\quad\quad n = 1, ..., N.
$$

This is known as the canonical representation of the decision hyperplane. These hyperplanes can be described by the equations:
${\bf w}^{\rm{T}}\phi({\bf x}) + b = 1$ (anything on or above this boundary is of one class, with label 1)
and
${\bf w}^{\rm{T}}\phi({\bf x}) + b = -1$ (anything on or below this boundary is of the other class, with label −1).

Geometrically, the distance between these two hyperplanes is computed using the distance from a point to a plane equation. It's $\frac {2}{\|{\vec {w}}\|}$, so to maximize the distance between the planes we want to minimize $\| \vec{w} \|$. The distance is computed using the distance from a point to a plane equation.

#### 2. Why is the SVM margin equal to $\frac {2}{\| {\bf w} \|}$?

###### method1:
Let ${\bf x}_0$ be a point in the hyperplane ${\bf w}{\bf x} - b = -1$. So we get ${\bf w}{\bf x}_0 - b = -1$. The distance from ${\bf x}_0$ to hyperplanes ${\bf w}{\bf x} - b = 1$:
\$$
\begin{equation}
 \begin{aligned}
& \gamma = \frac{\| {\bf w}{\bf x}_0 - b - 1 \|}{\sqrt{\| w\|^2}} \newline
\Longrightarrow &\gamma = \frac{\| -1 - 1 \|}{\| w\|} \newline
\Longrightarrow &\gamma = \frac{2}{\| w\|} \newline
 \end{aligned}
\end{equation}
$$

###### method2:
To measure the distance between hyperplanes ${\bf w}{\bf x} - b = -1$ and ${\bf w}{\bf x} - b = 1$, we only need to compute the perpendicular distance from ${\bf x}_0$ to plane ${\bf w}{\bf x} - b = 1$, denoted as $\gamma$.

Note that $\frac{\bf w}{\| {\bf w} \|}$ is a unit normal vector of the hyperplane ${\bf w}{\bf x} - b = 1$. So ${\bf x}_0 + \gamma \frac{\bf w}{\| {\bf w} \|}$ should be a point in hyperplane ${\bf w}{\bf x} - b = 1$.
\$$
{\bf w}({\bf x}_0 + \gamma \frac{{\bf w}}{\| {\bf w} \|}) - b = 1
$$

Expanding this equation, we have
\$$
\begin{equation}
 \begin{aligned}
&{\bf w}{\bf x}_0 + \gamma \frac{{\bf w}{\bf w}}{\| {\bf w} \|} - b = 1 \newline
\Longrightarrow &{\bf w}{\bf x}_0 + \gamma \frac{\| {\bf w} \|^2}{\| {\bf w} \|} - b = 1 \newline
\Longrightarrow &{\bf w}{\bf x}_0 + \gamma {\| {\bf w} \|} - b = 1 \newline
\Longrightarrow &{\bf w} {\bf x}_0 - b = 1 - \gamma {\| {\bf w} \|} \newline
\Longrightarrow &-1 = 1 - \gamma {\| {\bf w} \|} \newline
\Longrightarrow &\gamma = \frac{2}{\| {\bf w} \|} \newline
 \end{aligned}
\end{equation}
$$

The optimization problem then simply requires that we maximize ${\bf w}^{-1}$, which is equivalent to minimizing ${\bf w}^{2}$, and so we have to solve the optimization problem:
\$$
\begin{equation}
 \begin{aligned}
\underset{{\bf w},b}{\arg\min} &\quad \frac{1}{2}\| {\bf w}\| ^{2} \newline
\text{s.t.}  &\quad t_i ({\bf w}^{\rm{T}}\phi({\bf x}^{(i)}) + b) \geqslant 1, \quad i = 1, ..., m
 \end{aligned}
\end{equation}
$$

The factor of $1/2$ is included for later convenience. This is an example of a **quadratic programming** problem in which we are tring to minimize a quadratic function subject to a set of linear inequality constraints. 
