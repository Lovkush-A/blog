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
* Table of levels of reproducibility
* Same or different data or analysis. 
    * Rep = same, same
    * Replicable = different data
    * Robust = different analysis
    * Generalisable = diff, diff.
    * Repeatable is subset of reproducible. Literally same programs running same data and analysis to get same result. 
    * Reproducible means getting same result using same data and method - but maybe implemented in different language or program or ...
* CI/CD = continous integration / development
* Two big tools needed for repeatable research: dockers and version control. But requires learning. Not for everyone.
* Shows example from ligo about gravitational waves.
* Steps
    * Use jupyter notebook
    * Upload on public repositiy, e.g. GitHub.
    * Describe software needed to run notebook. Binder automatically identifies common configuration files
    * Done!
* Brief history of Binder. Now 140,000 sessions per week!
* Binder is open source, can adapt to your own needs. E.g. share only with specific people in your institution.
* Technologies
    * Github, clone reposity
    * repo2docker. Build docker image based on standard configuration files. Don't need docker file!
    * Docker. Execute docker image
    * Jupyter Hub. Allocate resources, make image accessible at url
    * Binder, redirect user to the url
* Scaling up
    * Created federation
    * Highly stable. Uses different kubernetes implementations
* User surveys
    * Around 80% of respondants would recommend service
    * Most common use case is teaching related. Examples, workings, uni teaching, demos, etc.
    * Biggest complaint: needs to be faster to load
    * Hard to speed up. But fully explained on jupyter hub blog. Why it is slow but also tips to speed things up.


### My thoughts
I had already heard of Binder - because I am friends with the spaker Sarah Gibson! However, the talk was still good, and I learnt more than I already knew. In particular, I liked the classification of different levels of reproducibility.



## 11:00, [Data pipelines with Python](https://ep2020.europython.eu/talks/4bNvaVk-mastering-a-data-pipeline-with-python-6-years-of-learned-lessons-from-mistakes/), Robson Junior

### Notes from talk
* Agenda: Not about code. Anatomy of data product, different architectures, qualities of data pipelines, how python matters.
* Anatomy of data product.
    * Ingress. Logs, databases, etc. Volume and variety are important.
    * Processes. Processing both input and output data.  Veracity and velocity are important here. E.g bank processing payment may have to run fraud detection - has to be very quick! 
    * Egress. Apis or databases. Veracity is important.
* Lambda and Kappa Architeture
    * Above anatomy of data product is same as computer program: input is files and memory, processes in ram, output to screen.
    * Lambda. Input data. Processing split into two layers. Speed layer for stream data, real time view. Batch layer, all data, batch views, usually processed periodically. Then output via query. See talk for diagram. Some pros and cons given in talk
    * Kappa. Only have a speed layer - no batch layer. Pros and cons given in talk.
* Qualities of a pipeline
    * Should be secure. Who has access to which data levels, use common format for data storage, be conscious of who has access to which parts of data and why.
    * Should be automated. Use good tools. Versioning, ci/cd, code review.
    * Monitoring. ??
    * Testable and tracable. Test all parts of the pipeline. Try to containerise tools.
* Python tools
    * ELT. Apache Spark, dash, luigi, mrjob, ray
    * Streaming. faust based on Kafka streams, streamparse via Apache Storm
    * Analysis. Pandas, Blaze, Open Mining, Orange, Optimus
    * Mangaement and scheduling. Apache Airflow. Programmatically create workflow
    * Testing. pytest, mimesis (create fake data), fake2db, spark-test-base
    * Validation. Cerberus, schema, voluptuous
     

### My thoughts
Unfortunately, I found the talk/speaker hard to follow. But I still got a list of tools that I can use as a reference.



## 12:15, [Probabilistic Forecasting with DeepAR](https://ep2020.europython.eu/talks/ANSma2D-probabilistic-forecasting-with-deepar-and-aws-sagemaker/), Nicolas Kuhaupt

### Notes from talk
* Freelance data scientist. German.
* DeepAR published by Amazon Research
* It is probabilistic, like ARIMA and regression models, but not like Plain LSTMs
* Automatic Feature Engineering. Key point of neural networks! Like Plain LSTMs but unlike ARIMA and regression models.
* One algorithm for multiple timeseries. Seems unlike other algorithms.  Like meta learning? Transfer learning??
* Disadvantages: time and resource intensive to train, difficult to set hyperparameters (like most neural networks).
* How it works: inputs time series x. At time t, outputs `z_t`, which are parameters for a probability distribution to predict. Then `z_t` *and* `x_t+1` are inputted into next stage. 
* Example datasets. Sales at amazons - one time series for each product, sales of magazines in different stores, forecast loads of servers in datacenters, car traffic - each lane has its own time series, energy consumption in households - each household has its own time series.
* Integrated into Sagemaker. 
    * Provides notebook
    * Lots of different tools for different stages of data science workflow. e.g. Ground Truth to use mechanical turk to build data sets. Studio is IDE. Autopilot for training models, Neo for deployment, etc.
* boto3 - access services in aws. s3fs - file storage stuff. various other details in talk
* data inform of json lines, (not pandas)
    * start - start time, target - the time series, cat - some categories that timeseries belongs to, dynamic_feat - extra time series (of same length as target).  note that different json lines can have time series of different lengths
* hyperparameters. standard stuff, with few extras for deepar.
* code to set up model and train it


### My thoughts
Looks like a powerful algorithm. Probabilistic algorithms are definitely the way to go - how else can you manage and predict risk?



## 13:15, [Fast and Scalable ML in Python](https://ep2020.europython.eu/talks/CR4ben4-the-painless-route-in-python-to-fast-and-scalable-machine-learning/), Victoriya Fedotova and Frank Schlimbach

### Notes from talk
* Python is useful but slow. So often companies hire engineers to translate python to faster languages like C++.
* Intel made a python distribution. No code changes required. * Just-in-time computation. JIT gives big speed boost.
* ML workflow: input and preprocessing via pandas, spark, SDC. model creation and prediction: scikit learn, spark, dl frameworks, daal4py.
* In this talk, talk about SDC, sckit learn, daal4py
* Intel Daal. Data analytics acceleration library. Linear Algebra already sped up (e.g. with MKL from intel), but this new library helps in situations whcih do not use linear algebra - e.g. tree based algorithms.
* Talk gives details on how to install packages and use it.
* Many algorithms have equivalent output to scikit learn algorithms. But not all - e.g. randomforest does not have 100% same output.
* Example of k-means in scikit-learn versus daal4py. Similar structure. Slightly different syntax. e.g. `n_clusters` versus `nClusters`.
* Scalable dataframe compiler SDC. a just in time compiler. Extension of Numba - made by anaconda.
* Easy to use. Just add decorator `@numba.jit` to function that you want to be compiled.
* Explanation of compiler pipeline.
* Talks through basic example of reading file, storing in frame, calculating mean, ordering a column. E.g. reading of file is parallelised, whereas pandas just reads data in single line.
* SDC requires code to be statically compilable - i.e. type stable. Examples where this wouldn't work.
* Charts showing speed-ups of different operations, as you increase the number of threads. Some operations get good speeds up, and some get mega speeds ups. Most was `apply(lambda x:x)` with 400x speed up. Something to do with lambda function being compiled too, not just apply function.



### My thoughts
Looks easy to use and can give big speed boosts. Is there a catch?


## 14:15 , [Small Big Data in Pandas, Dask and Vaex](https://ep2020.europython.eu/talks/A7TniMV-making-pandas-fly/), Ian Ozsvald

### Notes from talk
* Discuss how to speed things up in pandas. Why when there are tools out there? Ought to increase our knowledge and push our speed in current tools.
* Example of company registration data in uk
* Ram considerations
    * Strings are slow. Takes up lots of ram.
    * In example of company category taking up 300MB. Convert to category type and it takes 4.5 MB. Numeric code stored instead of strings.
    * Get speed up on value_counts. 0.485s vs 0.028s. 
    * Example of settign this column to index and then creating mask based on index. 281ms vs 569 microseconds.
    * Try using category for low cardinality data.

* float64 is default and expansive.
    * Example of age of company, up to 190 years.
    * Use float32 or float16 instead. 
    * Less RAM. Small time saving. (float16 might actually be slower!)
    * Might have loss in precision in data. Depends on data and usage
* Has a tool `dtype_diet` to automate process of optimising dataframe.
    * Produces table showing how different things can improve RAM usage. 
* Saving RAM is good. Can process more data. Speeds things up.
* Dropping to NumPy. 
    * `df.sum()` versus `df.values.sum()`.
    * In example, 19.1ms to 2ms.
    * Somethings to watch out for, e.g. NaN. 
    * James Powell produced diagram showing all files and functions called when doing sum in pandas versus in numpy. (Doing this using `ser.sum()`).

* Is pandas just super slow.
    * Can get big boost just by using bottleneck. see code in example.
    * just instal bottleneck, numexpr
    * Investigate dtype_diet
    * ipython_memory_usage

* Pure python is slow
    * Use numba, njit wrapper. See Intel talk above for newer extensions to numba.
    * Parallelise with Dask. Overhead may overwhelm benefit. USe profiling and timing tools to check.

* Big time savings come from our own habits
    * Reduce mistakes. Try nullable Int64, boolean
    * Write tests, unit and end-to-end
    * Lots of other examples from blog

* Vaex and Modin. Two other tools.
 

### My thoughts
Excellent talk! Ian clearly knows his stuff. Lots of insights. These are things I should start to implement to get some easy time savings.


## 14:45, [IPython](https://ep2020.europython.eu/talks/5LGWwvT-ipython-the-productivity-booster/), Miki Tebeka

### Notes from talk
* Programmer for 30 years.
* Wrote book, Python Brain Teasers
* Likes ipython. It is a REPL: Real, Eval, Prompt Loop
* Magic commands, via `%`. E.g. `%pwd`
* Can refer to outputs like variables. `logs_dir = Out[4]`
* Command line. `!ls $logs_dir`
* Auto-complete features
* He uses Vim! Woo!
* pprint for pretty printing.
* `?` for help. `??` for source code
* `%timeit function` to give time analysis
* `%%timeit`` for multiline stuff
* Can do sql stuff. have to install extension. see video for example.
* `%cow IPython Rocks!` Produces ascii art of cow!

### My thoughts
Always good to see a live demo to see exactly how somebody does things. I learnt some neat little features. Also, cool to see someone using Vim!


## 15:15, [NLPeasy](https://ep2020.europython.eu/talks/6x7ezDb-nlpeasy-a-workflow-to-analyse-enrich-and-explore-textual-data/), Philipp Thomann

### Notes from talk
* Co-creator of liquidSVM, Nabu, NLPeasy, PlotVR
* Works at D One. ML consultancy
* NLP. Big progress recently. Word2Vec, Deep learning and many good pre-trained models. Lots of data, many use cases
* Challenges for data scientists.
    * NLP is generally harder - high dimensions, specialised pre-processing required, nlp experts/researchers focus only on text but there is usually other data in business. 
    * Methods have repuation of being hard to use
    * standard tools not good for text. e.g. seaborn, tableau
* NLPeasy to the rescue!
* Pandas based pipeline, built in Regex based tagging, spaCy based NLP methods, Vader, scraping with beautiful soup, ...
* ElasticSearch. ??
* See Github repo for code, ntoebook example, etc.
* Talk through example, of looking at abstracts from some conference. 
* Live demo!
    * Scraping EP 2020 data.
    * Doing NLPeasy stuff
    * Go to Elastic dashboard
    * Lots of things shown, possible. E.g. entity extraction, tSNE
* REstaurant similarity using clustering algorithms. based only on reviews! can detect similarities
* Kibana (I think) can produce geoview / heatmap
* Can create networks using entity recognition
* Can try examples in different setups, e.g. Binder, or do it all yourself.

### My thoughts
Not much to say. Another tool that I now know about.


## 17:45, [30 Golden Rules for Deep Learning Performance](https://ep2020.europython.eu/talks/30-golden-rules-deep-learning-performance/), Siddha Ganju

### Notes from talk
* Forbes 30 under 30!
* Recommended [book](https://www.PracticalDeepLearning.ai)
* 95% of all AI Training is Transfer Learning
    * Playing melodica much easier if you already know how to play the piano
    * In Neural Network, earlier layers contains generic knowledge, and later layers contain task specific knowledge - at least in CNNs
    * So remove last 'classifier layers', and classify on new task, keeping first layers as is.
    * See github PracticalDL for examples and runnable scripts* Optimising hardware use.
    * In standard process, CPU and GPU switch between being idle or active.
    * Use a profiler. E.g. TensorFlow Profiler + TensorBoard
    * Example code shown to use profiler.
    * Simpler: nvidia-smi.
* We have thing we want to optimise and metric. So how can we optimise. Here comes the 30 rules.

* DATA PROCESSING
* Use TFRecords
    * Anti-pattern: thousands of tiny files/gigantic file.
    * Better to have handful for large files.
    * Sweet spot: 100MB TFRecord files
* Reduce size of input data
    * Bad: read image, rezie, train. Then iterate
    * Good: read all images, resizes, save as TFRecord. Then read and train - iterating as appropriate.
* Use TensorFlow Datasets
    * `import tensorflow_datasets as tfds`
    * If you have new datasets, publish your data on TensorFlow datasets, so people can build on your research.
* Use tf.data pipeline
    * Example code given
* Prefetch data
    * Somehow breaks circular dependency, where CPU has to wait for GPU and vice versa. 
    * Does asynchronous stuff
    * Code given in talk
* Parallelize CPU processing.
    * same as number of cpu cores
* Paralleize input and output. interleaving.
* Non-deterministic ordering. If randomising, forget ordering.
    * Somehow, not reading files 'in order' avoids potential bottlenecks.
* Cache data
    * Avoid repeated reading from disk after first epoch
    * Avoid repeatedly resizing
* Turn on experimental optimisations
* Autotune parameter values. code given in talk. Seems to refer to hardware based parameters
* Slide showing it all combined.

* DATA AUGMENTATION
* Use GPU for augmentation, with tf.image
    * Still work in progress. Limited functionality. Only rotate by 90 degrees as of now.
* Use GPU with NVIDIA DALI

* TRAINING
* Use automatic mixed precision.
    * Using 8-bit or 16-bit encodings rather than 32 or 64-bit.
    * Caveat - fp.16 can cause drop in accuracy, and even loss of convergence. E.g. if gradient is tiny, fp.16 will treat it as zero.
    * auto mixed precision somehow deals with this issue
* Use larger batch size.
    * Larger batch size leads to smaller time per epoch (but less steps per epoch too, no?)
    * Larger batch size leads to greater GPU utilization
* Use batch szies that are multiples of eight. Big jump in performance from 4095 to 4096!
    * Video describes other restrictions/options
* Finding optimal learning rate
    * Use keras_lr_finder
    * 'point of greatest decrease in loss' corresponds to best learning rate, somehow. Leslie N Smith paper
* Use tf.function
    * `@tf.function`.
* Overtrain, then generalise. STart from from dataset and increase.
* Install optimised stack
* Optimise number of parallel threads
* Use better hardware.
* Distribute training. MirroredStrat vs Multiworker...
* Look at industiral benchmarks.   

* INFERENCE
    * Use an efficient model. Fewer weights.
    * Quantize. 16 to 8bit.
    * Prune model, remove weights close to zero
    * Used fused operations
    * Enable GPU persistence


### My thoughts
Very handy list of tips and tricks. Sevreal of them go beyond my understanding, but does not mean I can not benefit from using them!


## 19:00, [Analytical Functions in SQL](https://ep2020.europython.eu/talks/AWFiM7F-sql-for-data-science-using-analytical-function/), Brendan Tierney

### Notes from talk
* TALK CANCELLED

### My thoughts
Not applicable


## 19:30, [Collaborative data pipelines with Kedro](https://ep2020.europython.eu/talks/45GhXwE-writing-and-scaling-collaborative-data-pipelines-with-kedro/), Tam-Sanh Nguyen

### Notes from talk
* Apparently 40% of Vietnamese have surname Nguyen.
* Data engineering is relatively new discipline, so there aren't established practices.
* QuantumBlack addressed this issue with Kedro.
* Pipelines.
    * Kedro viz used to visualise messy data pipeline.
    * But how does it get there?
    * Starts off simple. Iris data -> cleaning function, cleaned data, analyze function, analyzed data.
    * But then splitting data to test/train, gets messy.
    * Then have multiple data sources, each which needs to be split, and then all jumbled up
    * Without any tools, hard to grow larger and more complex than this. 
    * But Kedro can allow you to deal with more complex pipelines.
* One source of tension is difference between data engineer and data science
    * Data science usually have strong engineering skills. More data modelling skills
    * Data science has more research bent / experimental bent / want to be close to data and have many iterations
    * Data engineers more like engineers. Focus is on making things tidy, rather than experimentation.
* Two other challenges
    * Being ready for production quickly, for business use
    * Is it easy to pass the pipeline to future users - which may even be the future you!
* QuantumBlack startup from London, famous for doing work on F1. Got bought by McKinsey
* Kedro made by QuantumBlack. Big focus on standardisation, and making it as easy as possible for long-term use.

* How does Kedro work
* Analogy - audio world has standardised:
    * Input and output tools. Microphones, speakers, etc.
    * Functional transformers. Filters, etc.
    * Redirecting components. Make it easy for output from one tool easy to input into others.
    * Standard organisational conventions. Mic, audio mixer, computer.
* Standard is to use jupyter notebook. Some conventions (e.g. inputs and outputs via `pd.read_csv` and `pd.to_csv`), but mostly hard to follow. E.g. many many parameters is hard-coded, e.g. names of files being saved, parameters throughout processing, etc.
* Live example:
    * install kedro
    * kedro new. follow steps, e.g. naming things, etc.
    * created default template
    * kedro viz to visualise pipeline

* Run out of time, so rushes through lots of kedro features
* Has YouTube series DataEngineerOne


### My thoughts
* Looks like an intuitive system. Looks simpler than other pipelines presented in the conference. But is it because it actually is simpler, or is it because I am just getting used to pipelines. (Before this conference, I hadn't studied pipelines at all).



