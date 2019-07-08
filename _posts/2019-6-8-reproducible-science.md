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

This analysis is be the basis of a paper currently under review, _Co-occurrence of arsenic and fluoride in drinking water sources in Mexico: Geographical data visualization._

# The tools
## Cookiecutter
[Cookiecutter](http://drivendata.github.io/cookiecutter-data-science/) is a tool that generates structure for your data science project.  It creates a good folder structure, and populates it with some necessities like `.gitignore` files, a license statement, and a few files for packages that generate documentation automatically (though I don't use that functionality in this project). It also has the most sensible project structure I have found, one that clearly separates your raw data, generated data, python scripts, reports, etc.

Moreover, peole have made versions of this package for different niche applications, and I found a really useful one for Luigi workflows: [Cookiecutter Data Science with Luigi](https://github.com/ffmmjj/luigi_data_science_project_cookiecutter). The nice thing about this version is that it comes with:

1. A script `final.py` with a Luigi task `FinalTask` that you will populate with dependencies to all the other tasks in your project. 
2. A [GNU Make](https://www.gnu.org/software/make/) Makefile with a couple of commands that check your programming environment, clean up data, or execute `FinalTask`.

The upshot of this arrangement is that, once everything is in place and you're in the root folder of your Docker container, you can type `make data` into the terminal and Luigi will run all the tasks that haven't been run yet. One command to re-create any parts that are missing or the entire project. (There's also `make data_clean` to remove all the stuff you just created, and both can be customized).

I adapted the Cookiecutter folder structure to include a folder for Docker operations and to remove stuff that I wasn't using, and wound up with this:

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources (zipped maps of Mexico, INEGI town data)
│   ├── interim        <- Intermediate data that has been transformed. Includes unzipped maps.
│   ├── processed      <- The final, canonical data sets for mapping.
│   └── raw            <- The original, immutable CONAGUA data.
│
├── docker             <- Dockerfile and requirements.txt
│        
├── notebooks          <- Jupyter notebooks that explore the processed datasets.
│
├── reports            <- Generated analysis
│   └── figures        <- Figures generated by this analysis
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data_tasks     <- Scripts to unzip or generate data
│   │
│   └── visualization  <- Scripts that generate maps
│
└── test_environment.py <- Checks current python version.  For use with Make.
```

For more about the usefulness of Cookiecutter and how to think about organizing your data science project, check out [Cookiecutter Data Science — Organize your Projects — Atom and Jupyter](https://medium.com/@rrfd/cookiecutter-data-science-organize-your-projects-atom-and-jupyter-2be7862f487e).

## Docker
[Docker](https://www.docker.com/) is a tool for building lightweight containers (similar to virtual machines) that contain everything you need to run your analysis, from the operating system to all the python packages that your process depends on. For a good introduction to Docker, check out [How Docker Can Help You Become A More Effective Data Scientist](https://towardsdatascience.com/how-docker-can-help-you-become-a-more-effective-data-scientist-7fc048ef91d5).

In addition to the standard folder structure created with Cookiecutter, [this project's GitHub repo](https://github.com/DanielMartinAlarcon/Arsenic-and-Fluoride-Mexico) includes a `docker` folder with nothing but a `Dockerfile` and `requirements.txt`. These two files are all you need locally to build a Docker image, instantiate it into a Docker container, and fully recreate my programming environment. Alternatively, if you don't want to wait for Docker to build the image from these instructions, you can pull the image straight from [this project's DockerHub repo](https://hub.docker.com/r/danielmartinalarcon/arsenic-and-fluoride-in-mexico).

To test out my system, I cloned the GitHub repo to an [AWS Sagemaker](https://aws.amazon.com/sagemaker/) instance, pulled the Docker image from DockerHub, instantiated and entered a new container from that image, and typed `make data` into the terminal. After a few minutes, the whole analysis and all its figures had been reproduced on-site.

The most useful source of Docker commands is not the official documentation, but actually this [Docker cheat sheet](https://github.com/wsargent/docker-cheat-sheet) that links to it. [This presentation](https://www.youtube.com/watch?v=oO8n3y23b6M) ([and its slides](https://docs.google.com/presentation/d/1LkeJc-O5k0LQvzcFokj3yKjcEDns10JGX9uHK0igU8M/edit#slide=id.g23c212af60_0_0)) from ChiPy 2017 has several good examples, though note that some of the Dockerfile syntax has changed since it was published. [This presentation](https://www.youtube.com/watch?v=gBalsA-x300) ([and its repo](https://github.com/harnav/pydata-docker-tutorial/blob/master/dev-env/00-devenv.org)) from PyData LA 2018 has updated syntax and a much clearer walk-through for beginners.

## Luigi
[Luigi](https://luigi.readthedocs.io/en/stable/index.html#) is one of the few workflow management tools that are built entirely in Python (as opposed to having their own language, like GNU Make does). Luigi allows you to design modular, atomic tasks with clear dependencies. The idea is that all the steps in your project—from unzipping files to training models and making figures—should be written out as a separate task with known inputs and outputs. Luigi tasks can be swapped out for upgrades or repairs like fuses in a fusebox, which vastly increases the ability of _future_ you to come back to an old project and change a little detail in the middle of the workflow. 

When I needed to update my figure legends, for example, I went back to the parent class `BaseMap` (which inherits from `Luigi.Task`) and updated the method `make_legends()`, thus updating the 7 different descendant tasks that created the maps for the paper. I deleted the existing figures so that Luigi would run the task again, and `make data` promptly made the new figures for me.

For more about project organization and Luigi, check out [A Quick Guide to Organizing [Data Science] Projects (updated for 2018)](https://medium.com/outlier-bio-blog/a-quick-guide-to-organizing-data-science-projects-updated-for-2016-4cbb1e6dac71) and [Why we chose Luigi for our NGS pipelines](https://medium.com/outlier-bio-blog/why-we-chose-luigi-for-our-ngs-pipelines-5298c45a74fc). For the best introduction to Luigi syntax and motivation, check out minute 8:25 of [this presentation](https://www.youtube.com/watch?v=jpkZGXrhZJ8) from PyCon 2017. For a more complex machine learning project built around the same principles, check out [this presentation](https://www.youtube.com/watch?v=jRkW5Uf58K4) (and [repo](https://github.com/crazzle/pydata_berlin_2018)) from PyData Berlin 2018.