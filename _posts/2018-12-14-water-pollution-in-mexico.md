---
title: Water Pollution in Mexico
subtitle: A handy guide for finding trace elements
image: /img/Mexican-water-quality/all_contaminants.png
---

You can learn quite a few things by looking at the distribution of trace elements in water bodies around Mexico. Arsenic and fluoride, two elements known to have geological origins, are present mostly in groundwater (wells, springs). Cadmium, chromium, mercury, and lead, which are frequently tied to industrial production, are mostly present in rivers.  Many sites around the country present contaminant levels much higher than the limits set for drinking water by the US Environmental Protection Agency (EPA).  In this post, I present a cursory examination of this dataset and present my methodology for the benefit of anyone interested in issues of water quality.

![All contaminants](/img/Mexican-water-quality/all_contaminants.png)

In the maps below, I isolate each contaminant and also plot small gray plus signs for each of the sites in which the concentration of that element fell under the detection limit. The plus signs remind us that there are many clean water sources for each one that is contaminated. 

Arsenic is one of the most widely distributed contaminants, appearing above the EPA limits in fully 68.3% of the surveyed sites (and above 10x the limit in 3.7%, as shown below). Its distribution roughly tracks the largest mountain ranges in the country, the Sierra Madre Occidental, which is a continuation of the American Rocky Mountains. The most arsenic-polluted sites are mostly wells, since arsenic contamination is mostly due to geological causes.

![Arsenic](/img/Mexican-water-quality/arsenic.png)
![Arsenic Pie](/img/Mexican-water-quality/arsenic_pie.png)
![Arsenic Pie2](/img/Mexican-water-quality/arsenic_pie2.png)

Cadmium is present in relatively few sites, but at higher concentrations when it does appear. The highest cadmium levels (>10x) occur only in well water, while the vast majority of sites with levels between 1x and 10x of the EPA limit are actually in rivers and other surface sources (note how well is actually the smallest category in the second cadmium pie chart, which shows everything above 1x the EPA limit.
![Cadmium](/img/Mexican-water-quality/cadmium.png)
![Cadmium Pie](/img/Mexican-water-quality/cadmium_pie.png)
![Cadmium Pie 2](/img/Mexican-water-quality/cadmium_pie2.png)

Looking at the map for Chromium below, you might think to yourself "that big pink dot looks like the wastewater stream from a hood ornament factory or something".  Actually, you're seeing the sample from Rio Turbio, a river in the state of Guanajuato, which is 28,099 times above the EPA limit of 100 µg/L.  With such a deviation from the national average, I wouldn't be surprised if this outlier were actually the result of a typo or an instrument malfunction. Relatively few sites (0.6%) are contaminated with chromium, and most of those are rivers.

![Chromium](/img/Mexican-water-quality/chromium.png)
![Chromium Pie](/img/Mexican-water-quality/chromium_pie.png)
![Chromium Pie2](/img/Mexican-water-quality/chromium_pie2.png)

The two outliers for mercury are the Adolfo Lopez Mateos dam (state of Sinaloa) and a tributary of the Coatzacoalcos river, located in an industrial zone in the state of Veracruz. Those are the only two sites with mercury levels above 10x the EPA limit.  There's a much greater variety of sites with levels above 1x, however.  At that level, there's nearly as many sites contaminated with mercury as with arsenic. In contrast to arsenic, however, mercury is mostly present in rivers.  If I had to guess, I'd say it was because of industrial pollution rather than geological causes.
![Mercury](/img/Mexican-water-quality/mercury.png)
![Mercury Pie](/img/Mexican-water-quality/mercury_pie.png)
![Mercury Pie 2](/img/Mexican-water-quality/mercury_pie2.png)

Lead also shows a high, even distribution.  Its top two sites are rivers right before they drain into the Pacific ocean. Its abundance in rivers and discharges suggests that lead is caused by industrial pollution also.
![Lead](/img/Mexican-water-quality/lead.png)
![Lead Pie](/img/Mexican-water-quality/lead_pie.png)
![Lead Pie2](/img/Mexican-water-quality/lead_pie2.png)

Fluoride, finally, shows an even distribution throughout the middle of the country.  It is mostly absent in coastal regions and shows no sites with major contamination.  If I had to guess, I would say that it is more evenly distributed because high fluoride levels are less likely to be caused by industrial pollution. The pie charts bear this out.  There are no sites in the whole country where surface waters have fluoride levels above the EPA limit, even though a really high percentage of the total sites (3.9%) have at least 1x EPA contamination.

![Fluoride](/img/Mexican-water-quality/fluoride.png)
![Fluoride Pie](/img/Mexican-water-quality/fluoride_pie.png)
![Fluoride Pie2](/img/Mexican-water-quality/fluoride_pie2.png)

My data comes from the Mexican National Water Comission (CONAGUA). This dataset contains measurements of six trace elements, measured from 18,113 water samples, collected from 4,895 sites across the country. I present these maps as a resource for researchers that study water quality in Mexico. The full code for these maps is available as a [Jupyter notebook](https://github.com/DanielMartinAlarcon/Mexican-water-quality/blob/master/empirical/1_code/Mexican-water-quality.ipynb) in my Github account. 
