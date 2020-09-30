---
layout:            post
title:             "Thomas' Calculus reading notes"
date:              2020-09-30
tag:               Reading Books
category:          Reading Books
author:            tianliang
math:              true
---


# Thomas' Calculus, 13th Edition with SI Units by George B. Thomas


- TOC
{:toc}


## Chapter 10  
### Taylor and Maclaurin Series
**DEFINITIONS**
THEOREM 23—Taylor’s Theorem If ƒ and its first n derivatives ƒ′, ƒ″, c , ƒ(n) are continuous on the closed interval between a and b, and ƒ(n) is differentiable on the open interval between a and b, then there exists a number c between a and b such that: 
\$$
\begin{equation}
 \begin{aligned}
f(b) = f(a) &+ f'(a)(b-a) + \frac{f\'\'(a)}{2!}(b-a)^2 + \cdots \newline
&+ \frac {f^{(n)}(b-a)}{n!}(b-a)^n + \frac {f^{(n+1)}(c)}{(n+1)!}(b-a)^{n+1} ;
 \end{aligned}
\end{equation}
$$

When we apply Taylor’s Theorem, we usually want to hold a fixed and treat b as an independent variable. Taylor’s formula is easier to use in circumstances like these if we change b to x. Here is a version of the theorem with this change.

**Taylor’s Formula**  
If $f$ has derivatives of all orders in an open interval $I$ containing $a$, then for each positive integer $n$ and for each $x$ in $I$, 
\$$
\begin{equation}
 \begin{aligned}
f(x) = f(a) &+ f'(a)(x-a) + \frac{f\'\'(a)}{2!}(x-a)^2  + \cdots  \newline
&+ \frac{f^{(n)}(x-a)}{n!}(x-a)^n + R_n(x);
 \end{aligned}
\end{equation}
$$

where

\$$
\begin{equation}
 \begin{aligned}
R_n(x) = \frac{f^{(n+1)}(c)}{(n+1)!}(x-a)^{n+1} \qquad \text{for some $c$ between $a$ and $x$. }
 \end{aligned}
\end{equation}
$$





