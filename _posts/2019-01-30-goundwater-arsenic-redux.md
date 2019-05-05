---
title: Goundwater Arsenic in Durango
subtitle: Inferring arsenic from ionic signatures
image: /img/7_groundwater-arsenic/poison.png
---

The State of Durango, in northern Mexico, is one of those affected with significant natural presence of arsenic in groundwater (for a national analysis, see [this previous post](https://danielmartinalarcon.github.io/2018-12-14-water-pollution-in-mexico/)).  One of the problems with studying arsenic is that it's expensive to measure it accurately.  But what if we could reliably predict the arsenic content of a water sample, based on other characteristics of the water that are easier to determine?  

I set out to create a simple predictive model for arsenic (As) levels, and to determine which other soluble ions are most closely associated with it.  I used a dataset of water samples from Durango for which the levels of several common ions had been quantified.  I compared regression models of increasing complexity, selected and tuned by hand and then with an automatic model selector, ([TPOT](https://epistasislab.github.io/tpot/)). Finally, I used partial dependence plots and Shapley values to determine the contributions of individual dataset features to the overall prediction. 

Though the dataset is small (only 150 samples), it was enough to see that potassium (K) is by far the factor most closely associated with As, followed by fluoride and pH. You can see the full code that went into these calculations, as well as the specific factors with the greatest influence, [on my Github](https://github.com/DanielMartinAlarcon/arsenic-in-durango).

# Data cleanup and simple regressions
This dataset was gathered by researchers at the Advanced Materials Research Center (CIMAV) in Durango.  It contains 150 rows and 15 columns.

![Dataframe](/img/7_groundwater-arsenic/as1.png)

The distribution of As in the samples is almost normal, with a few outliers of super-high concentration.  I used robust scaling throughout my analysis to correct for those outliers.

![As Histogram](/img/7_groundwater-arsenic/as2.png)

I fit a simple regression and a ridge regression with feature selection ([selectkbest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)), comparing both to the baseline of just using the mean As value everywhere. 

![Simple model results](/img/7_groundwater-arsenic/as3.png)

Ridge regression works best, and the best model is actually one that considers only three features: pH, fluoride (F) and potassium (K).

![Features selected by selectkbest](/img/7_groundwater-arsenic/as4.png)

I used polynomial expansion to add all the possible x^2 features, and ran the model again with feature selection. This illustrates just how important potassium is.  It's the most significant feature on its own, it shows up as K^2, and the other features seem to matter only inasmuch as they interact with K (though there's also the interaction between F and conductivity).

![Selected poly features](/img/7_groundwater-arsenic/as5.png)






![word](/img/7_groundwater-arsenic/as3.png)

