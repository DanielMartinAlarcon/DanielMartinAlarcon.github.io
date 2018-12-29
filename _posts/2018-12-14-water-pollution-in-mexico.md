---
title: Water Pollution in Mexico
subtitle: An exploratory data analysis
image: /img/4_water-pollution-in-mexico/all_contaminants_cropped.png
---

In 2018, the Mexican National Water Commission (CONAGUA) published one of the most comprehensive datasets ever to be collected on the topic of water quality in Mexico. Water pollution with arsenic and fluoride is a well-known public health problem, but the scale and distribution of the problem had been difficult to visualize until now. 

This post is a quick exploratory analysis of the CONAGUA dataset. I confirm the importance of arsenic and fluoride as top contaminants, as well as the unexpectedly high prevalence of lead. I also find two very distinct distribution patterns. Some trace element contamination (arsenic, fluoride) is mainly caused by geology, and appears at low levels in a wide variety of sites. Other pollution (mercury, chromium, cadmium) is less widespread but can reach comically high levels in some sites. Industrial activity seems the most likely culprit.

Without further ado, let's look at the overall picture.

![All contaminants](/img/4_water-pollution-in-mexico/all_contaminants.png)

The CONAGUA dataset contains the details of 18,113 water samples from 4,895 sites around Mexico. The most common types of sites were rivers and wells, though many other bodies of surface water were also sampled. All these samples were collected in 2017. The limits mentioned in the legend are the safe limits for drinking water established by Mexican regulations.

![As and F](/img/4_water-pollution-in-mexico/arsenic_and_fluoride.png)

I mentioned before that arsenic and fluoride (As and F) are known to be the main contaminants in Mexican drinking water. This map shows their high co-occurrence along a northwest-southeast axis that roughly follows the mountain ranges and plateaus of the middle of the country. The relation to geography is particularly apparent when contrasted to how cadmium, mercury, and chromium are distributed.

![Cadmium](/img/4_water-pollution-in-mexico/cadmium.png)

![Mercury](/img/4_water-pollution-in-mexico/mercury.png)

![Chromium](/img/4_water-pollution-in-mexico/chromium.png)

The distribution of chromium, in particular, is downright comical. The giant pink dot is actually a cluster of contaminated samples from Rio Turbio, a river in the state of Guanajuato, the worst of which is 281 times above the safe limit of 100 µg/L. The two outliers for mercury are the Adolfo Lopez Mateos dam (state of Sinaloa) and a tributary of the Coatzacoalcos river, located in an industrial zone in the state of Veracruz. Those are the only two sites with mercury levels above 10x the safe limit.  

![Lead](/img/4_water-pollution-in-mexico/lead.png)

Lead shows a high, even distribution reminiscent of arsenic and fluoride, but spread throughout different parts of the country. Does this lead distribution track different geological features, or is it the result of industrial pollution? One suggestive piece of evidence comes from the types of sites where each pollutant was found:

![Stacked site types](/img/4_water-pollution-in-mexico/stacked_types.png)

If pollution comes from geological causes, I expect to find it in groundwater (the green bars in this chart). This happened with 1/3 of the high-arsenic sites, and every single high-fluoride site. Lead behaves more like cadmium, chromium, and mercury, appearing almost exclusively in rivers and surface water. 

It is also apparent from this chart that arsenic is the most widespread contaminant in Mexico by far, accounting for more contaminated sites than all the other pollutants put together. Arsenic and fluoride are usually cited as the most important contaminants because drinking water in Mexico is most often drawn from groundwater. This graph, however, shows how lead is actually present in more contaminated sites than fluoride and constitutes a significant problem.

It's also worth noting that I drew a separate color for samples labeled 'discharge', under the hypothesis that industrial contaminants would primarily show up in industrial or municipal discharges. To my surprise, those sites were actually no worse than rivers and other bodies of surface water (and represented a tiny amount of the total sites anyway). This is at least partly a result of sampling bias.  As the following figure shows, rivers are by far the type of site that were most heavily sampled for this dataset.

![status_by_site_type](/img/4_water-pollution-in-mexico/status_by_site_type.png)

As you can see, the problem of water pollution in Mexico is mostly one of arsenic, combined with lead in surface water and fluoride in groundwater. Exactly 1/3 of the samples in the CONAGUA dataset showed acceptable levels of all contaminants, though I'm tempted to report a higher pollution level because it seems like cheating to include a large number of clean samples from the ocean and coastal lagoons (I assume they're coastal).

I hope that these visualizations will be useful to the scientists working to understand water pollution and increase access to clean drinking water.  You can find the full code of this analysis, including several Jupyter notebooks and high-resolution versions of the maps and charts, [on my Github](https://github.com/DanielMartinAlarcon/Mexican-water-quality/blob/master/empirical/1_code/Mexican-water-quality.ipynb).