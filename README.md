# l2path

The `l2path` package contains implementation of path-following algorithms for both ridge regression and  <img src="https://render.githubusercontent.com/render/math?math=\ell_2">-regularized logistic regression.
The details of the methods can be found in 
[Zhu and Liu (2021) *An algorithmic view of $\ell_2$ regularization and some path-following algorithms*](https://jmlr.org/papers/volume22/19-477/19-477.pdf).

To install `l2path` from [github](http://github.com), type in R console
```R
devtools::install_github("renxiongliu/l2path")
```
Note that the installation above requires using R package [devtools](https://CRAN.R-project.org/package=devtools)
(which can be installed using `install.packages("devtools")`).

Please check the accompanying vignette on how to use the `l2path` package. To read vignette, please refer to the document [vignette](https://github.com/renxiongliu/l2path/blob/main/vignettes/vignette.pdf).
