---
title: Predictors of Goundwater Arsenic
subtitle: Feature selection in real-world data [under construction]
image: /img/7_groundwater-arsenic/poison.png
---

The State of Durango, in northern Mexico, is one of those affected with significant natural presence of arsenic in groundwater (for a national analysis, see [this previous post](https://danielmartinalarcon.github.io/2018-12-14-water-pollution-in-mexico/)).  One of the problems with studying arsenic is that it's expensive to measure it accurately.  But what if we could reliably predict the arsenic content of a water sample, based on other characteristics of the water that are easier to determine?  

I set out to determine which other features of groundwater are most closely associated with arsenic (As) levels, hoping to find a cheaper indicator that is strongly associated with As.  I used a dataset of water samples from Durango for which the levels of several common ions have been quantified.  I used feature selection, with and without polynomial expansion, to narrow the search.  I then used partial dependence plots and Shapley values to further determine the contributions of individual dataset features to the overall prediction. 

Though the dataset is small (only 150 samples), it was enough to see that potassium (K) is by far the factor most closely associated with As, followed by fluoride and pH. You can see the full code that went into these calculations, as well as the specific factors with the greatest influence, [on my Github](https://github.com/DanielMartinAlarcon/arsenic-in-durango).

# Potassium (K) is the most important feature
This dataset was gathered by researchers at the Advanced Materials Research Center (CIMAV) in Durango.  It contains 150 rows and 15 columns.

![Dataframe](/img/7_groundwater-arsenic/as1.png)

The distribution of As in the samples is almost normal, with a few outliers of super-high concentration.  I used robust scaling throughout my analysis to correct for those outliers.

![As Histogram](/img/7_groundwater-arsenic/as2.png)

I started out with ridge regression and feature selection ([selectkbest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)). Of the original features in the data, the lowest root mean square error (RMSE) under cross-validation comes from using just three features: pH, fluoride (F) and potassium (K), with K as the most important.

```
Best parameters from cross-validation:
ridge__alpha: 10.0
selectkbest__k: 3

Selected Features and regression coefficients:
pH   11.62
F    14.26
K    20.44
```

I looked for higher-dimensional interactions using polynomial expansion. I added all the possible x^2 features, and ran the ridge regression + feature selection pipeline again. The results further highlighted the importance of potassium.  K is the most significant feature on its own, it shows up as K^2, and the other features seem to matter only inasmuch as they interact with K (though there's also the interaction between F and conductivity).

```
Best parameters from cross-validation:
ridge__alpha: 1.0
selectkbest__k: 7

Selected Features and regression coefficients:
K                   -44.68
pH K                16.85
conductivity F      8.36
F K                 4.10
K^2                 29.12
K bicarbonate       0.20
K total_alcalinity  -4.90
```

# Potassium and Fluoride are most predictive at high concentrations




<!-- ![word](/img/7_groundwater-arsenic/as3.png) -->

