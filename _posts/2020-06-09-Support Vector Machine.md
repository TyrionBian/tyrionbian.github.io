---
layout:            post
title:             "Support Vector Machine"
date:              2020-06-09
tag:               Machine Learning Basic
category:          Machine Learning Basic
author:            tianliang
math:              true
---

- TOC
{:toc}

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

### 2. Why is the SVM margin equal to $\frac {2}{\Vert \vec{w} \Vert}$?

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

The optimization problem then simply requires that we maximize ${\bf w}^{-1}$, which is equivalent to minimizing ${\Vert {\bf w} \Vert}^{2}$, and so we have to solve the optimization problem:
\$$
\begin{equation}
 \begin{aligned}
\underset{w,b}{\arg\min} &\quad \frac{1}{2}{\Vert {\bf w} \Vert}^2 \newline
\text{s.t.}  &\quad t_i ({\bf w}^{\rm{T}}\phi({\bf x}^{(i)}) + b) \geqslant 1, \quad i = 1, ..., m
 \end{aligned}
\end{equation}
$$

The factor of $1/2$ is included for later convenience. This is an example of a **quadratic programming** problem in which we are tring to minimize a quadratic function subject to a set of linear inequality constraints. 

### 3. Lagrange multipliers
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

Here, the “$\mathcal P$” subscript stands for “primal”. We can write like this

\$$
\begin{equation}
 \begin{aligned}
θ_{\mathcal P}(w) = \left\{
\begin{array}{lr}
f(w) & \text{if w satisfies primal constraints} \newline
\infty & \text{otherwise}
\end{array} \right.
 \end{aligned}
\end{equation}
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

There must exist $w^∗, α^∗, β^∗$ so that $w^∗$ is the solution to the primal problem, $α^∗, β^∗$ are the solution to the dual problem, and moreover $p^∗ = d^∗ = L(w^∗, α^∗, β^∗)$. Moreover, $w^∗, α^∗$ and $β^∗$ satisfy the Karush-Kuhn-Tucker (KKT) conditions, which are as follows

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

### 4. margin classifiers
Previously, we posed the following (primal) optimization problem for finding the optimal margin classifier:
\$$
\begin{equation}
 \begin{aligned}
\underset{w,b}{\arg\min} &\quad \frac{1}{2}{\Vert {\bf w} \Vert}^2 \newline
\text{s.t.}  &\quad t_i ({\bf w}^{\rm{T}}\phi({\bf x}^{(i)}) + b) \geqslant 1, \quad i = 1, ..., m
 \end{aligned}
\end{equation}
$$

We can write the constraints as
\$$
g_i(w) = 1-t_n ({\bf w}^{\rm{T}}\phi({\bf x}^{(i)}) + b) \le 0
$$

When we construct the Lagrangian for our optimization problem we have:
\$$
L(w,β,α) = \frac{1}{2}{\Vert {\bf w} \Vert}^2 - \sum_{n=1}^N \alpha_n \{ t_n ({\bf w}^{\rm{T}}\phi({\bf x}^{(i)}) + b -1) \}
$$

where ${\bf α}  = (\alpha_1, . . . , \alpha_N)^{\mathrm T}$. Note the minus sign in front of the Lagrange multiplier term, because we are minimizing with respect to w and b, and maximizing with respect to a. Setting the derivatives of L(w, b, a) with respect to w and b equal to zero,
\$$
\begin{equation}
 \begin{aligned}
\nabla_w L(w,b,α) = {\bf w} - \sum_{n=1}^N \alpha_n t_n \phi({\bf x}^{(i)}) = 0 \newline
\frac{\partial}{\partial b} L(w,b,α) = - \sum_{n=1}^N \alpha_n t_n = 0
 \end{aligned}
\end{equation}
$$

we obtain the following two conditions:

\$$
\begin{equation}
 \begin{aligned}
{\bf w} & = \sum_{n=1}^N a_n t_n \phi({\bf x}_n) \newline
0 & = \sum_{n=1}^N a_n t_n
 \end{aligned}
\end{equation}
$$

Eliminating ${\bf w}$ and b from $L({\bf w}, b, {\bf a})$ using these conditions then gives the dual representation of the maximum margin problem in which we maximize

\$$
\begin{equation}
 \begin{aligned}
{\max}_a & \quad \overset{\sim}L ({\bf a}) = \sum_{n=1}^N a_n - \frac{1}{2}\sum_{n=1}^N \sum_{m=1}^N a_n a_m t_n t_m k({\bf x}_n,{\bf x}_m) \newline
\text{s.t.} & \quad a_n \ge 0, \quad n=1, ..., N \newline
& \quad \sum_{n=1}^N a_n t_n = 0
 \end{aligned}
\end{equation}
$$

Here the kernel function is defined by $k(x, x') = \phi(x)^{\mathrm T} \phi(x')$. Again, this takes the form of a quadratic programming problem in which we optimize a quadratic function of a subject to a set of inequality constraints. 

### 5. non-separable case
The derivation of the SVM as presented so far assumed that the data is **linearly separable**. While mapping data to a high dimensional feature space via $\phi$ does generally increase the likelihood that the data is separable, we
can’t guarantee that it always will be so. Also, in some cases it is not clear that finding a separating hyperplane is exactly what we’d want to do, since that might be susceptible to outliers. For instance, the left figure below
shows an optimal margin classifier, and when a single outlier is added in the upper-left region (right figure), it causes the decision boundary to make a dramatic swing, and the resulting classifier has a much smaller margin.

<figure>
   <img src="{{ "/images/linearly_separable.png" | absolute_url }}" />
   <figcaption>linearly separable(from wikipedia)</figcaption>
</figure>


To make the algorithm work for non-linearly separable datasets as well as be less sensitive to outliers, we reformulate our optimization (using $ℓ1$ regularization) as follows:
\$$
\begin{equation}
 \begin{aligned}
\underset{\gamma,w,b}{\min} \quad & \frac{1}{2}{\Vert w \Vert}^2 + C \sum_{i=1}^m \xi_i \newline
\text{s.t.} \quad & y^{(i)}(w^{\rm T} x^{(i)} + b) \ge 1
& \xi_i \ge 0, \quad i=1, ..., m
 \end{aligned}
\end{equation}
$$

Thus, examples are now permitted to have (functional) margin less than 1, and if an example has functional margin $1 − ξ_i ($with $ξ > 0)$, we would pay a cost of the objective function being increased by $Cξ_i$. The parameter $C$ controls the relative weighting between the twin goals of making the $\Vert w \Vert^2$ small (which we saw earlier makes the margin large) and of ensuring that most examples have functional margin at least 1.

As before, we can form the Lagrangian:
\$$
L(w,b,ξ,a,r) = \frac{1}{2}w^{\rm T}w + C sum_{i=1}^m ξ_i - sum_{i=1}^m a_i [y^{(i)}(x^{\rm T}w + b) - 1 + ξ_i] - \sum_{i=1}^m r_i ξ_i
$$

Here, the $α_i$’s and $r_i$’s are our Lagrange multipliers (constrained to be $≥ 0$). We won’t go through the derivation of the dual again in detail, but after setting the derivatives with respect to $w$ and $b$ to zero as before, substituting them back in, and simplifying, we obtain the following dual form of the problem:

\$$
\begin{equation}
 \begin{aligned}
{\max}_a & \quad \overset{\sim}L ({\bf a}) = \sum_{n=1}^N a_n - \frac{1}{2}\sum_{n=1}^N \sum_{m=1}^N a_n a_m t_n t_m k({\bf x}_n,{\bf x}_m) \newline
\text{s.t.} & \quad 0 \le a_n \le C, \quad n=1, ..., N \newline
& \quad \sum_{n=1}^N a_n t_n = 0
 \end{aligned}
\end{equation}
$$

## Reference
[1] [CS229: Machine Learning](http://cs229.stanford.edu/).

[2] Bishop C M. Pattern recognition and machine learning[M]. springer, 2006.
