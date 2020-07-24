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
* Background. Singaporean. Works at essence. Blog data double confirm.
* Will try to give business centred context - very different from academic/research/learning contetx
* Tasks
    * Get column names
    * Get size of dataset. Make sure all data has been loaded.
    * Check datatypes. Sometimes goes wrong. Sign for data-cleaning.
    * Get unique values. Some cases, need to combine different values into one. E.g. 'Male' and 'M'
    * Get range of values
    * Get count of values. Group-by various columns as appropriate.
    * Rename column names. E.g. for merging
    * Remove symbols in values. E.g. currency signs
    * Convert strings to numeric or to dates
    * Replace values with more sensible values. E.g. 'Male' vs 'M'
    * Identify variables/columns similar or different across datasets
    * Concatenate data. Data from different quarters added to single table.
    * Deduplication. Remove duplicate data.
    * Merge
    * Recoding. Feature engineering
    * Data profiling (optional)
    * Input missing values (optional)
* Common issues
    * inconsistent naming of variables
    * bad data formats
    * invalid,missing values
* Resources: tinyurl.com/y5b3y7to

### My thoughts
It is useful to have a checklist of tasks one should do when they have to clean data. Interesting that they considered the input of missing values as a bonus task - the impression I got from the Kaggle tutorials is that one ought to do some imputation.

## 9:00, [Neural Style Transfer and GANs](https://ep2020.europython.eu/talks/BSeL2FG-painting-with-gans-challenges-and-technicalities-of-neural-style-transfer/), Anmol Krishan Sachdeva

### Notes from talk
* Recap of GANs
    * Discriminative model. Supervised classification model. Fed in data.
    * Generative model. Mostly unsupervised. Generates new data by underlying underlying data distribution. Generates near-real looking data. (Conditional generators have some element of supervised learning included). Learning of distribution called implicit density estimation.
    * In end, have GAN which takes random input and creates an output that has similar distribution it was trained on.
    * Training. Dicriminator gets true entry x and fake input from generator network x\*. Generator tries to classify. Compute error and backpropogate. Generator then trained. Similar.
* Book: [GANs in Action by Langr and Bok](https://www.manning.com/books/gans-in-action#toc)
* GANs have been successful in creating near-real images.
* Style Transfer: Content image + style image gives new image with content from first in style of second image.
* Aim of Style Transfer Networks
    * Not to learn underlying distrbution.
    * Somehow learn style from style image and embed it into content cimage
    * Interpolation is bad
    * Used in gaming industry, mobile applications, fashion/design
* Popular networks: Pix2Pix, CycleGAN, Neural Style Transfer
* Neural Style Transfer.
    * No training set! Just take as input the two images.
    * Example from Neural Algorithm of Artistic Style. Photo turned into painting of certain artist
    * Content loss - measure of content between new image and content image
    * Style loss - measure of style between new image and style image.
    * total variation loss. Check for blurriness, distortions, pixelations
    * How is learning done? No trianing set, no back prop?
* Shows example notebook, using TensorFlow and Keras. Imports pre-trained model.
* Content loss.
    * Pixel by pixel comparison not done.
    * Compare higher level features, obtained from pre-trained model. E.g. VGG19 is 19 layered CNN, which classifies images into 1000 categories. Trained on million images.
    * VGG architecture. See [Generative Deep Learning by David Foster](https://www.amazon.co.uk/Generative-Deep-Learning-Teaching-Machines/dp/1492041947).
    * Keras repo. Neural style transfer.
    * Take 3rd layer of block 5 from VGG19 model as measure of higher level features.
    * Content loss = mean squared error between this vector encoding of high level features from VGG19 for content image and generated image.
* Style loss
    *  Take dot product of 'flattened feature maps'
    * Lower level layers represent style.
    * GRAM matrix. Dot product of 'flattened features' of image with itself. 
    * Didn't understand this.
    * Loss = MSE (gram matrix(style image), gram matrix(generated image))
    * Then take weighted sum over different sets of layers
    * Shows example code.
    * `features = K.batch_flatten(K.permute...)`
    * Take first layer in each block
* Total variation loss
    *  sum of squared difference between image and image shifted on pixel down, and of shifted one pixel right
* Training
    * Add up all the loss
    * Use L-BFGS optimisation. Essentially just gradient descent on individual pixel values.
    * Example code shown in talk.
* Pix2Pix
    * Various cool things you can do. E.g. aerial image to a map. Sketch to full image.
* CycleGAN
    * E.g. convert image of apples to oranges. Or horses to zebras!

### My thoughts
Excellent talk! I had seen some of the neural style transfer images before, and now I have some understanding of how they are created!



## 10:00, [Data Visualisation Landscape](https://ep2020.europython.eu/talks/B5Vff6U-the-python-data-visualization-landscape-in-2020/), Bence Arato 

### Notes from talk
* There are lots of libraries out there.
* Imperative vs declarative. Imperative: specify how something should be done. Declarative: specify what should be done.
* Matplotlib. Biggest example. Has gallery + code examples
    * Background in MATLAB
    * Challenge: imperative
* Seaborn. Aim to provide higher level package on top of matplotlib. Good defaults for charts that look good.
    * Example scatterplot much easier in seaborn than in matplotlib
* plotnine. Aim to higher-level. Based on ggplot2 in R.
    * Syntax is basically same as ggplot syntax! 
    * `aes`, `geom_point` etc.
* bokeh. 2nd most widely known tool.
    * Big thing is interactivity, like sliders, checkboxes, etc.
    * Example scatterplot. Quite long and low-level
    * Based on web/javascript background
* HoloViews
    * Higher level language for bokeh, matplotlib, plotly
    * Just have to change one line code to switch between bokeh or matplotlib output
    * Scatterplot example. Short code.
* hvPlot. built on top of holoviews
* pandas bokeh. add plot_bokeh() method in pandas.
* Chartify. From spotify. Built on top of bokeh. 
* Plotly.
    * Might be only one to do 3d charts well
    * Low level charts
* Plotly express
    * Higher level version of plotly.
* Vega, vega-lite
    * Visualisation 'grammar'. 
    * JSON based way of describing charts
* Altair
    * High level version of vega
* Dashboards.
    * Plotly Dash
    * Panel. Anaconda related. Built on top of four big charting libaries above
    * Voila. Looks cool - can visualise ML things. Need to check out!
    * Streamlit. Datascience specific tools. Example of GAN Image generator with sliders. Change slider will re-run model and create new image.
* PyViz. Thorough list of all visualisation libraries.
* Python Data Visualisation, at AnacondaCON 2020


### My thoughts
Excellent talk. Well structured, good examples, good summary of key things I should know about.



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




