---
title: Goundwater Arsenic Revisited
subtitle: Predictive analytics on messy real-world data
image: /img/7_groundwater-arsenic/poison.png
---

The State of Durango, in northern Mexico, is one of those affected with significant natural presence of arsenic in groundwater (for a national analysis, see [this previous post](https://danielmartinalarcon.github.io/2018-12-14-water-pollution-in-mexico/)).  One of the problems with studying arsenic is that it's expensive to measure it accurately.  But what if we could reliably predict the arsenic content of a water sample, based on other characteristics of the water that are easier to determine?  

I used a dataset of water samples from Durango, for which the levels of several common ions had been quantified.  I trained several regression models on these data and found that potassium (K) is by far the factor that is most closely associated with arsenic presence in this dataset, followed by fluoride and pH.  When I added polynomial features to capture the importance of interactions, almost all the important interaction terms involved K as well.

You can see the full code that went into these calculations, as well as the specific factors with the greatest influence, in the [Jupyter Notebook for this project](https://github.com/DanielMartinAlarcon/Mexican-water-quality/blob/master/1_code/Groundwater_arsenic_revisited.ipynb).