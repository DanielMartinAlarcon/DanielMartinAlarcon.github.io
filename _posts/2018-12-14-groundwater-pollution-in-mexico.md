---
title: Water Pollution in Mexico
subtitle: Maps showing the distribution of trace elements in Mexican water sources
image: /img/Mexican-water-quality/all_contaminants.png
---

I recently came across a dataset of water quality created by the Mexican National Water Comission (CONAGUA). This dataset contains measurements of six trace elements, measured from 18,113 water samples, collected from 4,895 sites across the country. The type of site contains everything from drinking wells and rivers to oceans and industrial wastemater streams. It paints a comprehensive picture of the water quality in the country.

The first thing to note is the overall picture. Many sites around the country show contaminant levels much higher than the limits set by the US Environmental Protection Agency for drinking water. 

![All contaminants](/img/Mexican-water-quality/all_contaminants.png)

Each of those trace elements shows a different distribution. In each of the maps below, I isolate one contaminant and also plot small gray plus signs for each of the sites in which the concentration of that element fell under the detection limit. The plus signs remind us that there are many clean water sources for each one that is contaminated.

Arsenic presents perhaps the greatest public health risk, of the trace elements analyzed here.  It is present at dangerously high concentrations in more sites than any other contaminant.  Its distribution roughly tracks the largest mountain ranges in the country, the Sierra Madre Occidental, which is a continuation of the American Rocky Mountains.
![Arsenic](/img/Mexican-water-quality/arsenic.png)

Cadmium is present in relatively few sites, but at higher concentrations when it does appear.
![Cadmium](/img/Mexican-water-quality/cadmium.png)

Looking at the map for Chromium below, you might think to yourself "that big pink dot looks like the wastewater stream from a hood ornament factory or something".  Actually, you're seeing the sample from Rio Turbio, a river in the state of Guanajuato, which is 28,099 times above the EPA limit of 100 µg/L.  With such a deviation from the national average, I wouldn't be surprised if this outlier were actually the result of a typo or an instrument malfunction.

![Chromium](/img/Mexican-water-quality/chromium.png)

The two outliers for mercury are the Adolfo Lopez Mateos dam (state of Sinaloa) and a tributary of the Coatzacoalcos river, located in an industrial zone in the state of Veracruz.
![Mercury](/img/Mexican-water-quality/mercury.png)

Lead also shows a high, even distribution.  Its top two sites are rivers right before they drain into the Pacific ocean.
![Lead](/img/Mexican-water-quality/lead.png)

Fluoride, finally, shows an even distribution throughout the middle of the country.  It is mostly absent in coastal regions and shows no sites with major contamination.  If I had to guess, I would say that it is more evenly distributed because high fluoride levels are less likely to be caused by industrial pollution.
![Fluoride](/img/Mexican-water-quality/fluoride.png)

I present these maps as a resource for researchers that study water quality in Mexico. The full code for these maps is available as a [Jupyter notebook](https://github.com/DanielMartinAlarcon/Mexican-water-quality/blob/master/empirical/1_code/Mexican-water-quality.ipynb) in my Github account. 
