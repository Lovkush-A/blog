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
* Why use Docker? Without Docker, hard to share your models because hard to make sure everyone is using the same modules/appropriate versions of packages.
* What is Docker? Helps you solve this issue with 'containers'. Bundles together the application and all packages/requirements.
* Difference to virtual machines: Each application has its own container in Docker, which is small and efficient, whereas virutal machine is highly bloated as each applications is grouped with whole OS and unnecessary extra baggage.
* Image - an archive with all the data need to run the app. Running an image creates a container.
* Common challanges in DS:
    * Complex setups/dependencies, reliance on data, highly iterative/fast workflow, docker can be hard to learn.
* Describes differences to web apps
* 
 
