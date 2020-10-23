---
layout:            post
title:             "k-mer notes"
date:              2020-10-23
tag:               Bioinformatics
category:          Bioinformatics
author:            tianliang
math:              true
---


# k-mer notes
update time: 2020-10-23

-----------------------------

- TOC
{:toc}


## Concepts 
#### DEFINITIONS
**DEFINITIONS**
1. **k-mers** are subsequences of length $k$ contained within a biological sequence.  
2. **k-mers** are composed of nucleotides (i.e. A, T, G, and C).
3. A sequence of length $L$ will have $L-k+1$ k-mers and $n^{k}$ total possible k-mers, where $n$ is number of possible monomers.

A method of visualizing k-mers, the k-mer spectrum, shows the multiplicity of each k-mer in a sequence versus the number of k-mers with that multiplicity. The number of modes in a k-mer spectrum for a species's genome varies, with most species having a unimodal distribution. However, all mammals have a multimodal distribution. The number of modes within a k-mer spectrum can vary between regions of genomes as well: humans have unimodal k-mer spectra in 5' UTRs and exons but multimodal spectra in 3' UTRs and introns.

**k-mer size**
The choice of the k-mer size has many different effects on the sequence assembly. These effects vary greatly between lower sized and larger sized k-mers. Therefore, an understanding of the different k-mer sizes must be achieved in order to choose a suitable size that balances the effects. The effects of the sizes are outlined below.


## Metagenomics
k-mer frequency and spectrum variation is heavily used in metagenomics for both analysis and binning. 

## Reference 
[1] [k-mer (wikipedia.org)](https://en.wikipedia.org/wiki/K-mer)  
[2] [Nucleotide (wikipedia.org)](https://en.wikipedia.org/wiki/Nucleotide)

