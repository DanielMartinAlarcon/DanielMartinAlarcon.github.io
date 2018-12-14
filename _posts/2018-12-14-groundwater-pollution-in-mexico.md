---
title: Groundwater Pollution in Mexico
subtitle: Maps of groundwater sources that exceed EPA limits in Mexico
image: /img/Mexican-water-quality/all_contaminants.png
---

I recently came across a dataset of water quality created by the Mexican National Water Comission (CONAGUA). This dataset contains measurements of six trace elements, measured from 18,113 water samples, collected from 4,895 sites across the country. The type of site contains everything from drinking wells and rivers to oceans and industrial wastemater streams. It paints a comprehensive picture of the water quality in the country.

The first thing to note is the overall picture. Many sites around the country show contaminant levels much higher than the limits set by the US Environmental Protection Agency for drinking water. 

![All contaminants](/img/Mexican-water-quality/all_contaminants.png)

Each of those trace elements shows a different distribution. In each of the maps below, I isolate one contaminant and also plot small gray plus signs for each of the sites in which the concentration of that element fell under the detection limit. The plus signs remind us that there are many clean water sources for each one that is contaminated.

Arsenic presents perhaps the greatest public health risk, of the trace elements analyzed here.  It is present at dangerously high concentrations in more sites than any other contaminant. 
![Arsenic](/img/Mexican-water-quality/arsenic.png)
![Cadmium](/img/Mexican-water-quality/cadmium.png)
![Chromium](/img/Mexican-water-quality/chromium.png)
![Mercury](/img/Mexican-water-quality/mercury.png)
![Lead](/img/Mexican-water-quality/lead.png)
![Fluoride](/img/Mexican-water-quality/fluoride.png)


The full code for these maps is available as a [Jupyter notebook](https://github.com/DanielMartinAlarcon/Mexican-water-quality/blob/master/empirical/1_code/Mexican-water-quality.ipynb).
