---
title: Reproducible data science with Docker and Luigi
subtitle: The case of arsenic and fluoride in Mexican drinking water
image: /img/11_reproducible-science/all_contaminants_cropped.png
---

# TL;DR
I describe a workflow that uses [Docker](https://www.docker.com/) and [Luigi](https://luigi.readthedocs.io/en/stable/index.html) to create fully transparent and reproducible data analyses. End users can repeat the original calculations to produce all the final tables and figures starting from the original raw data. End users (and the author, at a later date) can easily make isolated changes to the middle of the pipeline and know that their changes propagate to all the other steps in the process. I demonstrate this workflow by applying it to a paper (currently under peer review) that studies drinking water contamination in Mexico.

![All contaminants](/img/11_reproducible-science/all_contaminants.png)

# The motivation
Historically, my data science workflows were built in Jupyter notebooks. `pd.read_csv()` imported raw data in the first cell, and `df.to_csv()` or `plt.savefig()` exported the final results in the last cell. I carefully commented the intermediate steps so that my thought process was clear, and I saved the notebook for posterity. 

A few weeks later, though, I would always run into the same problem. I'd revisit the project and want to change a detail in one of the figures. Or maybe I'd have a new question about one of the intermediate dataframes. Either way, if I wanted to observe or change anything in the middle of the pipeline I'd have to run the whole notebook again just to get to the middle. More often than not, those changes would also break the code written in the last cells. Even more commonly, I would discover that right in the middle of the notebook I'd imported more data that was generated elsewhere and whose exact origin and nature were undocumented. What _was_ that data, again? Which version did I use? Code written a few months ago might as well have been written by someone else, as they say, and I frequently discovered just how difficult it is to keep track of information flows in and out of notebooks. This was not code that could be used in production.

What I love about software engineering is that there are really good tools to address this universal problem, which plagues science in general and data science in particular. Researchers are more and more frequently using [Git](https://git-scm.com/) to track versions, [Docker](https://www.docker.com/) to package their compute environments, and workflow managements systems like [Airflow](https://airflow.apache.org/) or [Luigi](https://luigi.readthedocs.io/en/stable/index.html#) to manage the dependencies between all the tasks in their analyses. I've applied all these tools to a real-world research project. Let's take a look at how they work together.

# The science
In 2018, the Mexican National Water Commission (CONAGUA) published one of the most comprehensive datasets ever to be collected on the topic of water quality in Mexico. This data included measurements of six trace-element contaminants (arsenic, cadmium, chromium, mercury, lead, and fluoride) in 14,058 water samples from 3,951 sites throughout the country, all collected during 2017.

Water pollution with trace elements, particularly arsenic and fluoride, is a public health problem whose existence is well documented but whose exact scale remains unknown. I dove into this dataset to find out how arsenic and fluoride (As and F) are distributed in the country, and to estimate the health burden associated with continuous exposure to these contaminants. My task required mapping all the sampling sites, visualizing which sites are contaminated, determining the population exposed to that contamination, and estimating the health burden associated with that exposure. 

This analysis is be the basis of a paper currently under review, entitled _Co-occurrence of arsenic and fluoride in drinking water sources in Mexico: Geographical data visualization._

# The tools

