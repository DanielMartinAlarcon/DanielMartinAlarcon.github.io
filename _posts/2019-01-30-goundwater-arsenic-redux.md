---
title: Goundwater Arsenic in Durango
subtitle: Inferring arsenic from ionic signatures
image: /img/7_groundwater-arsenic/poison.png
---

The State of Durango, in northern Mexico, is one of those affected with significant natural presence of arsenic in groundwater (for a national analysis, see [this previous post](https://danielmartinalarcon.github.io/2018-12-14-water-pollution-in-mexico/)).  One of the problems with studying arsenic is that it's expensive to measure it accurately.  But what if we could reliably predict the arsenic content of a water sample, based on other characteristics of the water that are easier to determine?  

I set out to create a simple predictive model for arsenic (As) levels, and to determine which other soluble ions are most closely associated with it.  I used a dataset of water samples from Durango for which the levels of several common ions had been quantified.  I compared regression models of increasing complexity, selected and tuned by hand and then with an automatic model selector, ([TPOT](https://epistasislab.github.io/tpot/)). Finally, I used partial dependence plots and Shapley values to determine the contributions of individual dataset features to the overall prediction. 

Though the dataset is small (only 150 samples), it was enough to see that potassium (K) is by far the factor most closely associated with As, followed by fluoride and pH. You can see the full code that went into these calculations, as well as the specific factors with the greatest influence, [on my Github](https://github.com/DanielMartinAlarcon/arsenic-in-durango).

# Data cleanup
This dataset was gathered by researchers at the Advanced Materials Research Center (CIMAV) in Durango.  It contains 150 rows and 15 columns.
![as1](/img/7_groundwater-arsenic/as1.png)

The distribution of As in the samples is almost normal, with a few outliers of super-high concentration.

![as2](/img/7_groundwater-arsenic/as2.png)
