---
toc: true
layout: post
description: Summaries of the talks and my thoughts for Day 2
categories: [python, data science, conference]
title: EuroPython Conference 2020, Day 2
---
## Other posts in series
{% for post in site.posts %}
{% if (post.title contains "EuroPython Conference 2020") and (post.title != page.title) %}
* [{{ post.title }}]({{ site.baseurl }}{{ post.url }})
{% endif %}
{% endfor %}

## Introduction
I am attending the online conference [EuroPython 2020](https://ep2020.europython.eu/), and I thought it would be good to record what my thoughts and the things I learn from the talks.


## 7:00, [Automating machine learning workflow with DVC](https://ep2020.europython.eu/talks/CXG7TcM-automating-machine-learning-workflow-with-dvc/), Hongjoo Lee

### Notes from talk
* Works for SK hynix, a memory chip maker in South Korea.
* Waterfall vs agile production. Waterfall = design, then build, then release, then done. Agile, more iterative approach.
* Software dev/dev ops: code git, build sbt maven, test jUnit, release Jenkins, deploy Docker aws, operate Kubernetes, monitor ELK stack
* ML dev lifecycle: get data, preprocess, build model, optimise model, deployment.
    * Often getting data requires domain expertise
    * Often software engineers are needed for deployment
    * Still improvement and modification needed for ML dev workflow. In early stages.
* Data versioning.
    * Can be terrible, with names like `raw_data`, `cleaned_data`, `cleaned_data_final`.
    * Need to have system where any change in data will trigger pipeline
* ML is metric driven. Software engineering is feature driven.
    * ML Models version should be tracked.
    * Metrics should be versioned/tracked
* DVC helps. Some laternatives: git-LFS, MLflow, Apache Airflow
    * Easy to use
    * Language independent
    * Useful for individuals and for large teams
* Example code: [on github](https://github.com/midnightradio/handson-dvc)
* Hongjoo works through a cats vs dogs example in talk
    * Download template directory
    * Use git to keep track of changes
    * Use dvc command-line commands to define each of the steps in ml pipeline
    * dvc dag command - shows ascii diagram of pipeline
    * dvc repro - checks if there are changes, and if so, re-runs pipeline. E.g. change model by adding a layer, then dvc repro does everything. you have to manually to git commit command and naming of versions
    * dvc metric show. shows metrics of all versions done.

### My thoughts
Looks simple enough! I could follow the talk. :D It is clear that finding a good ML workflow is a big theme. At the end of the conference, I will have to go through these notes and collate the various tools and workflows people use, so I have a reference for when I need to use it.


## 7:30, [Tips for Data Cleaning](https://ep2020.europython.eu/talks/CivrR5y-top-15-python-tips-for-data-cleaning-understanding/), Hui Ziang Chua

### Notes from talk


### My thoughts


## 9:00, [Neural Style Transfer and GANs](https://ep2020.europython.eu/talks/BSeL2FG-painting-with-gans-challenges-and-technicalities-of-neural-style-transfer/), Anmol Krishan Sachdeva

### Notes from talk


### My thoughts




## 10:00, [Data Visualisation Landscape](https://ep2020.europython.eu/talks/B5Vff6U-the-python-data-visualization-landscape-in-2020/), Bence Arato 

### Notes from talk


### My thoughts




## 10:30, [Binder](https://ep2020.europython.eu/talks/BqQBN6J-sharing-reproducible-python-environments-with-binder/), Sarah Gibson

### Notes from talk


### My thoughts




## 11:00, [Data pipelines with Python](https://ep2020.europython.eu/talks/4bNvaVk-mastering-a-data-pipeline-with-python-6-years-of-learned-lessons-from-mistakes/), Robson Junior

### Notes from talk


### My thoughts




## 12:15, [Probabilistic Forecasting with DeepAR](https://ep2020.europython.eu/talks/ANSma2D-probabilistic-forecasting-with-deepar-and-aws-sagemaker/), Nicolas Kuhaupt

### Notes from talk


### My thoughts




## 13:15, [Fast and Scalable ML in Python](https://ep2020.europython.eu/talks/CR4ben4-the-painless-route-in-python-to-fast-and-scalable-machine-learning/), Victoriya Fedotova and Frank Schlimbach

### Notes from talk


### My thoughts



## 14:15 , [Small Big Data in Pandas, Dask and Vaex](https://ep2020.europython.eu/talks/A7TniMV-making-pandas-fly/), Ian Ozsvald

### Notes from talk


### My thoughts

## 14:45, [IPython](https://ep2020.europython.eu/talks/5LGWwvT-ipython-the-productivity-booster/), Miki Tebeka

### Notes from talk


### My thoughts



## 15:15, [NLPeasy](https://ep2020.europython.eu/talks/6x7ezDb-nlpeasy-a-workflow-to-analyse-enrich-and-explore-textual-data/), Philipp Thomann

### Notes from talk


### My thoughts



## 17:45, [30 Golden Rules for Deep Learning Performance](https://ep2020.europython.eu/talks/30-golden-rules-deep-learning-performance/), Siddha Ganju

### Notes from talk


### My thoughts

## 19:00, [Analytical Functions in SQL](https://ep2020.europython.eu/talks/AWFiM7F-sql-for-data-science-using-analytical-function/), Brendan Tierney

### Notes from talk


### My thoughts



## 19:30, [Collaborative data pipelines with Kedro](https://ep2020.europython.eu/talks/45GhXwE-writing-and-scaling-collaborative-data-pipelines-with-kedro/), Tam-Sanh Nguyen (microsoft strem)

### Notes from talk


### My thoughts




