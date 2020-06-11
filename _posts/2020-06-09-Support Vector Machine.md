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
\frac{t_n y({\bf x})}{\Vert {\bf w} \Vert} = \frac{t_n ({\bf w}^{\rm{T}}\phi({\bf x}) + b)}{\Vert {\bf w} \Vert}
$$

We note that if we make the rescaling $w → \kappa w$ and $b → \kappa b$,
then the distance from any point xn to the decision surface, given by $t_n y({\bf x})/\Vert {\bf w} \Vert$ is unchanged. We can use this freedom to set:
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

Geometrically, the distance between these two hyperplanes is computed using the distance from a point to a plane equation. It's $\frac {2}{\Vert{\vec {w}}\Vert}$, so to maximize the distance between the planes we want to minimize $\Vert \vec{w} \Vert$. The distance is computed using the distance from a point to a plane equation.

#### 2. Why is the SVM margin equal to $\frac {2}{\Vert \vec{w} \Vert}$?

###### method1:
Let ${\bf x}_0$ be a point in the hyperplane ${\bf w}{\bf x} - b = -1$. So we get ${\bf w}{\bf x}_0 - b = -1$. The distance from ${\bf x}_0$ to hyperplanes ${\bf w}{\bf x} - b = 1$:
\$$
\begin{equation}
 \begin{aligned}
& \gamma = \frac{\Vert {\bf w}{\bf x}_0 - b - 1 \Vert}{\sqrt{\Vert w\Vert^2}} \newline
\Longrightarrow &\gamma = \frac{\Vert -1 - 1 \Vert}{\Vert w\Vert} \newline
\Longrightarrow &\gamma = \frac{2}{\Vert w\Vert} \newline
 \end{aligned}
\end{equation}
$$

###### method2:
To measure the distance between hyperplanes ${\bf w}{\bf x} - b = -1$ and ${\bf w}{\bf x} - b = 1$, we only need to compute the perpendicular distance from ${\bf x}_0$ to plane ${\bf w}{\bf x} - b = 1$, denoted as $\gamma$.

Note that $\frac{w}{\Vert w \Vert}$ is a unit normal vector of the hyperplane ${\bf w}{\bf x} - b = 1$. So ${\bf x}_0 + \gamma \frac{w}{\Vert w \Vert}$ should be a point in hyperplane ${\bf w}{\bf x} - b = 1$.
\$$
\begin{equation}
 \begin{aligned}
{\bf w}({\bf x}_0 + \gamma \frac{\bf w}{\Vert {\bf w} \Vert}) - b = 1
 \end{aligned}
\end{equation}
$$

Expanding this equation, we have
\$$
\begin{equation}
 \begin{aligned}
&{\bf w}{\bf x}_0 + \gamma \frac{\bf ww}{\Vert {\bf w} \Vert} - b = 1 \newline
\Longrightarrow &{\bf w}{\bf x}_0 + \gamma \frac{\Vert {\bf w} \Vert^2}{\Vert {\bf w} \Vert} - b = 1 \newline
\Longrightarrow &{\bf w}{\bf x}_0 + \gamma {\Vert {\bf w} \Vert} - b = 1 \newline
\Longrightarrow &{\bf w} {\bf x}_0 - b = 1 - \gamma {\Vert {\bf w} \Vert} \newline
\Longrightarrow &-1 = 1 - \gamma {\Vert {\bf w} \Vert} \newline
\Longrightarrow &\gamma = \frac{2}{\Vert {\bf w} \Vert} \newline
 \end{aligned}
\end{equation}
$$

The optimization problem then simply requires that we maximize ${\bf w}^{-1}$, which is equivalent to minimizing ${\bf w}^{2}$, and so we have to solve the optimization problem:
\$$
\begin{equation}
 \begin{aligned}
\underset{{\bf w},b}{\arg\min} &\quad \frac{1}{2}\Vert {\bf w} \Vert ^2 \newline
\text{s.t.}  &\quad t_i ({\bf w}^{\rm{T}}\phi({\bf x}^{(i)}) + b) \geqslant 1, \quad i = 1, ..., m
 \end{aligned}
\end{equation}
$$

The factor of $1/2$ is included for later convenience. This is an example of a **quadratic programming** problem in which we are tring to minimize a quadratic function subject to a set of linear inequality constraints. 

#### 3. Lagrange multipliers
We consider an optimization problem in the standard form:
\$$
\begin{equation}
 \begin{aligned}
\text{minimize} \quad &f_0(x) \newline
\text{s.t.} \quad &f_i(x) \leqslant 0, \quad i = 1, ..., m \newline
&h_i(x) = 0, \quad i = 1, ..., p
 \end{aligned}
\end{equation}
$$

with variable $x \in \mathbb R^n$.We define the Lagrangian $L : \mathbb R^n × \mathbb R^m × \mathbb R^p → \mathbb R$ associated with the problem (5.1) as:
\$$
L(x,α,β) = f_0(x) + \sum_{i=1}^m α_i f_i(x) + \sum_{i=1}^p β_i h_i(x),
$$

with **dom** $L = \mathcal D × \mathbb R^m × \mathbb R^p$.We refer to $α_i$ as the Lagrange multiplier associated with the ith inequality constraint $f_i(x) ≤ 0$; similarly we refer to $β_i$ as the Lagrange multiplier associated with the ith equality constraint $h_i(x) = 0$. The vectors $α$ and $β$ are called the dual variables or Lagrange multiplier vectors associated with the problem.

Now we consider the quantity
\$$
θ_{\mathcal P}(w) = \underset{α,β : α_i≥0}{\max} L(w, α, β)
$$

Here, the “$\mathcal P$” subscript stands for “primal”. We can write like this:
\$$
θ_{\mathcal P}(w) = \left\{
\begin{array}{lr}
f(w) & \text{if $w$ satisfies primal constraints} \newline
\infty & \text{otherwise}
\end{array} \right.
$$

$θ_{\mathcal P}$ takes the same value as the objective in our problem for all values of w that satisfies the primal constraints, and is positive infinity if the constraints are violated. Hence, if we consider the minimization problem
\$$
\underset{w}{\min} θ_{\mathcal P}(w) = \underset{w}{\min} \underset{α,β : α_i≥0}{\max} L(w, α, β)
$$

we see that it is the same problem (i.e., and has the same solutions as) our original, primal problem. For later use, we also define the optimal value of the objective to be $p^∗ = \underset{w}{\min} θ_{\mathcal P}(w)$; we call this the **value** of the primal problem.

Now, let’s look at a slightly different problem. We define
\$$
θ_{\mathcal D}(α, β) = \underset{w}{\min}L(w, α, β).
$$

Here, the “D” subscript stands for “dual.” Note also that whereas in the definition of $θ_{\mathcal P}$ we were optimizing (maximizing) with respect to $α, β$, here we are minimizing with respect to $w$.

We can now pose the dual optimization problem:
\$$
\underset{α,β : α_i≥0}{\max} θ_{\mathcal D}(α, β) = \underset{α,β : α_i≥0}{\max} \underset{w}{\min}L(w, α, β)
$$

This is exactly the same as our primal problem shown above, except that the order of the “max” and the “min” are now exchanged. Optimal value of the dual problem’s objective to be $d^∗ = \underset{α,β : α_i≥0}{\max} θ_{\mathcal D}(w)$.

How are the primal and the dual problems related? It can easily be shown that
\$$
d^* = \underset{α,β : α_i≥0}{\max} \underset{w}{\min}L(w, α, β) = \underset{w}{\min} \underset{α,β : α_i≥0}{\max} L(w, α, β) = p^*
$$

There must exist $w^∗, α^∗, β^∗$ so that $w^∗$ is the solution to the primal problem, $α^∗, β^∗$ are the solution to the dual problem, and moreover $p^∗ = d^∗ = L(w^∗, α^∗, β^∗)$. Moreover, $w^∗, α^∗$ and $β^∗$ satisfy the Karush-Kuhn-Tucker (KKT) conditions, which are as follows:
\$$
\begin{equation}
 \begin{aligned}
\frac{\partial}{\partial w_i}L(w^∗, α^∗, β^∗) & = 0, \quad i=1,...,n \newline
\frac{\partial}{\partial β_i}L(w^∗, α^∗, β^∗) & = 0, \quad i=1,...,l \newline
α_i^∗g_i(w^*) & = 0, \quad i=1,...,k \newline
g_i(w^*) & \leq 0, \quad i=1,...,k \newline
α_i^∗ & \geq 0, \quad i=1,...,k \newline
 \end{aligned}
\end{equation}
$$


### 3. Kernels
### 4. Regularization and the non-separable case
### 5. The SMO algorithm






