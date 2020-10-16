---
toc: true
layout: post
description: I describe the rabbit hole I ended up in by trying to learn about the CAP Theorem. 
categories: [data science, data engineering]
title: The CAP Theorem's never ending rabbit hole 
---
## Introduction
I recently started this [EdEx course](https://courses.edx.org/courses/course-v1:AWS+OTP-AWS-D6+2T2019/course/) on Amazon's DynamoDB service. Within one of the first few videos, the CAP Theorem was
discussed. As someone with a pure maths background, theorems are irresistable bait, so I wanted to find out more about it.

## My journey
I cannot remember all the details, but here are some of the things I have found and thought in my trip down the rabbit hole.

* The wikipedia page was the starting point, of course. I couldn't say it really helped me, as most of the terms described were not helpful.
* However, when looking at the references, something stood out. One of my friend's brothers, Martin Kleppmann, has written a bunch of stuff on this!
* This [blogpost](https://martin.kleppmann.com/2015/05/11/please-stop-calling-databases-cp-or-ap.html) on why the CAP framework is practically unhelpful, was helpful. It gave an intuitive proof of the theorem.  However, the proof is so simple that I wonder what the big deal is. There is clearly something I am missing.
* The blogpost contained various links. At this stage, I lose track of things and which links came from where.
* A couple of other helpful blogposts were [this](https://www.the-paper-trail.org/page/cap-faq/) and [this](http://www.julianbrowne.com/article/brewers-cap-theorem). I gained a better sense of why this is an important issue, and what underlying problems are that people are thinking about.
* One thing that is frustrating is how many terms people take for granted. E.g. the concepts of *nodes* and *network* are taken as self-evident, yet they are fundamental to the statement of the theorem. They cannot just be the same thing as nodes and networks from Graph Theory, because stuff is actually happening within the nodes, but none of the blog posts I found actually explain these concepts.
* There is whole bunch of things I found in this journey that look interesting and I intend to read at some point:
    * [This](https://www.cl.cam.ac.uk/~pes20/weakmemory/cacm.pdf) paper on how certain multiprocessors do not provide sequentially consistent memory, and their proposed solution.
    * These [lecture notes](http://www.cs.cmu.edu/~harchol/ISCA15show.pdf) on queueing theory and its application to computer systems.
    * This [article](https://www.microsoft.com/en-us/research/wp-content/uploads/2011/10/ConsistencyAndBaseballReport.pdf) by Microsoft explaining data consistency in databases using baseball.
    * This [book](http://dataintensive.net/) on 'The big ideas behind reliable, scalable, maintainable data-intensive systems'. Seems to be well-regarded by many people.
    * This [blogpost](https://www.benkuhn.net/progessays/) containing a list of highly recommended essays on programming.
    * This [book](https://smile.amazon.com/Explain-Cloud-Like-Im-10-ebook/dp/B0765C4SNR) 'Explain the Cloud Like I'm 10' looks like an easy to read book where I will learn a lot about what really goes on behind the scenes.

## Conclusions
There is too much to learn! One thing I still need to improve in myself is my ability to prioritise all these difference things.
