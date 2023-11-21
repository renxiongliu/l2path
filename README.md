# l2path

The `l2path` package contains implementation of path-following algorithms for both ridge regression and L2-regularized logistic regression.
The details of the methods can be found in: 
1. [Zhu and Liu (2021):   *An algorithmic view of L2 regularization and some path-following algorithms*](https://jmlr.org/papers/volume22/19-477/19-477.pdf)
2. [Zhu and Liu (2023):   *Path following algorithms for L2-regularized M-estimation with approximation guarantee*](https://openreview.net/pdf?id=hgLMht2Z3L).

To install `l2path` from [github](http://github.com), type in R console
```R
devtools::install_github("renxiongliu/l2path")
```
Note that the installation above requires using R package [devtools](https://CRAN.R-project.org/package=devtools)
(which can be installed using `install.packages("devtools")`).

Please check the accompanying [vignette](https://github.com/renxiongliu/l2path/blob/main/vignettes/vignette.pdf) on how to use the `l2path` package.
