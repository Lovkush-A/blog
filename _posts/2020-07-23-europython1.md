---
toc: true
layout: post
description: I am attending my first ever tech-related conference. Here I record my thoughts on Day I.
categories: [python, data science, conference]
title: EuroPython Conference 2020, Day 1
---
## Other posts in series
{% for post in site.posts %}
{% if (post.title contains "EuroPython Conference 2020") and (post.title != page.title) %}
* [{{ post.title }}]({{ site.baseurl }}{{ post.url }})
{% endif %}
{% endfor %}

## Introduction
I am attending the online conference [EuroPython 2020](https://ep2020.europython.eu/), and I thought it would be good to record what my thoughts and the things I learn the talks.

## 08:00, Waking up
I struggled to wake up in time, for two main reasons:
* For the past several months, I have had no need to wake up early, and so my standard wake up time has been 9am.
* I stayed up until 2am watching Round 2 of the [Legends of Chess](https://www.youtube.com/watch?v=CDZJynwAIV4) tournament...

But I managed! I then got onto the Discord server to get to the first keynote talk, [30 Golden Rules of Deep Learning Performance](https://ep2020.europython.eu/talks/30-golden-rules-deep-learning-performance/), which sounds like it would be particularly insightful. But, unfortunately, the speaker could not give the talk so it was cancelled. At least it gives me time to get [breakfast](www.huel.com)!


## 09:00, [Docker and Python](https://ep2020.europython.eu/talks/4bVczWt-docker-and-python-making-them-play-nicely-and-securely-for-data-science-and-ml/), Tania Allard

### Notes of the talk
* Why use Docker? Without Docker, hard to share your models because hard to make sure everyone is using the same modules/appropriate versions of packages.
* What is Docker? Helps you solve this issue with 'containers'. Bundles together the application and all packages/requirements.
* Difference to virtual machines: Each application has its own container in Docker, which is small and efficient, whereas virutal machine is highly bloated as each applications is grouped with whole OS and unnecessary extra baggage.
* Image - an archive with all the data need to run the app. Running an image creates a container.
* Common challanges in DS:
    * Complex setups/dependencies, reliance on data, highly iterative/fast workflow, docker can be hard to learn.
* Describes differences to web apps
* Building docker images. Tania had many struggles/frustrations to learn Docker.
    * Many bad examples online/in tutorials.
    * If building from scratch, use official Python images, and the slim versions.
    * If not building from scratch (highly recommended), use Jupyter Docker stacks.
* Best practices:
    * Be explicit about packages. Avoid 'latest' or 'python 3'.
    * Add security context. LABEL securitytxt='...'. E.g. snake.
    * Split complex run statements
    * Prefer Copy to Add
    * Leverage cache
        * Clean, e.g. conda clean
        * Only use necessary packages
        * Use Docker ignore (similar to .gitignore)
    * Minimise privilege.
        * Run as non-root user. Dockers runs as root by default
        * Minimise capabilities user
    * Avoid leak sensitive information
        * Information in middle 'layers' may appear hidden, but there are tools to find them.
        * Use multi-stage builds
* This can be overwhelming, and that is normal. Try to automate and avoid re-inventing the wheel.
    * Use standard project template, e.g., cookie cutter data science.
    * Use tools like repo2docker. `conda instal jupyter repo2docker`, `jupyter-repo2docker "."`
* Re-run docker image regularly. One benefit is making sure you have latest security patches. Don't do this manually, use GitHub Actions or Travis, for example.
* Top top tips:
    * Rebuild images frequently. Get security updates.
    * Do not work as root/minimise privileges
    * Don't use Alpine Linux. You are paying price for small size. Use buster, stretch, or Jupyter stack.
    * Be explicit about what packages you are require, version EVERYTHING.
    * Leverage build cache. Separate tasks, so you do not need to rebuild whole image for small change.
    * Use one dockerfile per project
    * Use multi-stage builds.
f   * Make images identifiable
    * Use repo2docker
    * Automate. Do not build/push manually
    * Use a linter. E.g. VSCode has docker extension
    
### My thoughts
A lot of this went over my head. The main lesson I learnt is that I should expect things to be tricky when I eventually do start using Docker. I will refer back to this video when I do start using Docker.


## 10:00 [spaCy](https://ep2020.europython.eu/talks/7TXpVro-15-things-you-should-know-about-spacy/), Alexander Hendorf

### Notes of the talk
* NLP: avalanche of unstructured data
* Estimate 20:80 split between structured and unstructured ata
* Examples of NLP: chatbots, translation, sentiment analysis, speech-to-texxt or vice versa, spelling/grammar, text completion ([GPT-3](https://github.com/openai/gpt-3)! :D)
* Example of Alexander's work: certain group had large number of documents and search engine was not helping them. Used NLP to create clusters of documnets, create keywords, create summaries.
* spaCy. Open source library for NLP, comes with pretrained language models, fast and efficient, designed for production usage, lots of out-of-the-box support.
* Building blocks of spaCy
    * Tokenization
    * Part of speech tagging. E.g. which words are nouns or verbs, etc.
    * Lemmatization. cats->cat
    * Sentence boundary detection
    * Named Entity Recognition. (Apple -> company). Depends on context!
    * Serialization. saving
    * Dependency parsing. How different tokens depend on each other
    * Entity linking
    * Training. Updating models
    * Text classification 
    * Rule-based matching
* Built-in rules
    * Rules for specific languages, e.g., adding 's' to end of noun makes it plural in English.
    * Usually does not cover many exceptions
    * Most languages won't be supported. Most research done on English.
* Built-in models
    * Language models. E.g. word vectors
    * Can train your own models with nlp.update()
    * Need a lot of data to train these models. Few documents is not enough
* spaCy is pythonic
    * should understand objects, iterations, comprehensions, classes, methods
    * Might be overwhelming, but not as overwhelming as Pandas yet!
* Pipelines
    * Has nice image in slides
    * default pipline: tokenize, tag, parse, then your own stuff
* Visualisation
    * used to be separate package displacy
    * visualisation of sentence grammar/dependencies
    * visualise entities. e.g. given sentence, highlight grouping of nounes. e.g. is a word a person, or a date, or an animal, or...
* Serialization
    * uses pickle
* Danger zones
    * Privacy, bias, law, language is not fixed in stone
* Can't do all languages
* Extensions
    * spaCy universe
    * E.g. NeuralCoref.  Matching up 'Angela Merkel' and 'chancellor' recognised as same person.
* Bugs
    * spaCy will maintained, and quick response to bug reports
    * Extensions more variable
* Status
    * Other options: NLTK, Gensin, TextBlob, Pattern
    * spaCy: usually close to the state of the art, especially for language models, flexible, extendable via spacy universe, fast (powered by cython)
    * For special cases, probably use other models. E.g. RASA for contextual textbots


### My thoughts
I have not yet done any NLP, but when I do, I will be sure to look into spaCy after this talk!

## 10:30 [Differential Privacy](https://ep2020.europython.eu/talks/6Js4E4r-diffprivlib-privacy-preserving-machine-learning-with-scikit-learn/), Naoise Holohan

### Notes of the talk
* Many examples where 'anonymised' but actually could still identify individuals by matching data with other data publically available.
    * Netflix data. Matched with imdb databse
    * AOL data. NYT managed to identify individual and make available their full search
    * Limosine service. Person matched it with images of celebrities, and managed to find out about individuals use of taxis/limos
    * Many other examples out there.

* Differential privacy. Main idea: blur noise.
* Implemented in diffprivlib, for scikit learn.
* Modules
    * Mechanisms. Algorithms to add noise
    * Models. Has scikit learn equivalents as its parent class
    * Tools. Analogue for NumPy.
    * Accountant. Track privacy budget. Help optimise balance between accuracy and privacy.
* Examples
    * Gives warning for privacy leakage with certain bound is not specified. If bound is not specified, then original dataset is used to estimate the parameter!
    * Pipelines. With and without privacy.
        * Without, got 80.3% accuracy
        * With, got 80.7% accuracy! Adding noise can actually reduce over-fitting. !!
        * Graph of epsilon vs accuracy. Low epsilon highly variable performance. Higher epsilon tends to 80% baseline
    * Some exploratory data analysis

### My thoughts
I have actually heard of differential privacy before, via [this talk](https://www.youtube.com/watch?v=bScJdHX0Hac) from [FacultyAI](https://faculty.ai/). To anybody interested in this topic, I recommend watching the FacultyAI talk for more background on differential privacy itself.

Main lesson here is that if I want to analyse sensitive data, diffprivlib is a good open source option.

The big surprise factor was that the accuracy can sometimes be better after adding privacy! Adding noise can reduce over-fitting.


## 12:15, [Parallel and Asynchronous Programming in DS](https://ep2020.europython.eu/talks/8DboZjY-speed-up-your-data-processing/), Chin Hwee Ong

### Notes of the talk
* Background. Engineer at ST Engineering. Background in aerospace engineer and modelling. Contributor to pandas. Mentor at BigDataX.
* Typical flow: extract raw data, process data, train model, evaluate and deploy model.
* Bottlenecks in real world
    * Lack of data. Poor quality data
    * Data processing. 80/20 dilemma. More like 90/10!
* Data processing in python
    * For loops, `list = [], for i in range(100), list.append(i*i)`
    * This is slow!
    * Comprehensions. `list = [i*i for i in range(100)]`
    * Slightly better. No need to call append on each iteration
    * Pandas, optimised for in-memory analytics. But get performance issues when dealing with large datasets, e.g. 1>GB. Particularly in 100GB plus range.
    * Why not just use spark? Overhead cost of communication. Need very big data for this to be worthwhile. What to do in 'small big data'?
    * *Small Big Data Manifesto* by Itamar Turner-Trauring
* Parallel processing
    * Analogy: preparing toast.
    * Traditional breakfast in Singapore is tea, toast and egg
    * Sequential processing: one single-slice toaster
    * Parallel processing: four single-slice toaster. Each toaster is independent
* Synchronous vs asynchronous
    * Analogy: also want coffee. Assume it takes 5 mins for each coffee, 2 mins for single toaster.
    * Synchronous execution: first make coffee, and then make toast.
    * Asynchronous: make coffee and toast at the same time
* Practical considerations
    * Parellelism sounds great. Get mega time savings
    * Is code already optimised? Using loops instead of array operations
    * Problem architecture. If many tasks depends on previous tasks being completed, parallelism isn't great. Data dependency vs task dependency.
    * Overhead costs. Limit to parallelisation. Amdahl's. 
    * Multi processing vs multi threading
* In python
    * concurrent.futures module
    * ProcessPoolExecutor vs ThreadPoolExecutor.
* Example
    * Obtaining data from API. JSON data. 20x speed up versus list comprehension.
    * Rescaling x-ray images. map gives 40 seconds. list comprehension 24 seconds. ProcessPoolExecutor about 7s with 8 cores.
* Takeaways
    * Not all processes should be parallelized. Amdahl's law, system overhead, cost of re-writing code.
    * Don't use for loops!

### My thoughts
Something for me to investigate. I have not needed to use this yet. Nice bonus - I learnt about Singaporean breakfasts!


## 12:45, [Automate NLP model deployment](https://ep2020.europython.eu/talks/5hXHveq-deploy-your-machine-learning-bots-like-a-boss-with-cicd/), William Arias
### Notes of the talk
* Background. Colombian, lives in Prague. Works at GitLab
* Often, data scientists are a one-person band. Have to do learn many different tool.
* Define a symphony. Produce flowchart of data workflow. Make explicit where different people can/should contribute to process.
    * Favour for yourself
    * Makes easier for everyone to understand process
* Symphony components.
![image]({{ site.baseurl }}/images/europython_arias1.png)
* I found it hard to follow details here. I'd have to re-watch the video.
* This might be standard knowlege for people who already work in software engineering. But somebody with maths background, say, this kind of automation and workflow is not obvious.
* This will make your life easier!
* Has video showing example of making small change to chat bot, and how much is automated.

### My thoughts
Probably too advanced for me at this stage. But, something I should be aware of when I work on bigger projects. 



## 13:15, [Building models with no expertise with AutoML](https://ep2020.europython.eu/talks/C8WFfBR-building-smarter-solutions-with-no-expertise-in-machine-learning/), Laurent Picard

### Notes of the talk

### My thoughts




### Notes of the talk

### My thoughts





### Notes of the talk

### My thoughts





