---
toc: true
layout: post
description: I briefly describe my first experience web scraping, for STEP past papers and solutions.
categories: [python, scraping]
title: Web Scraping for STEP past papers and solutions
---

## Introduction
Though I am transitioning from teaching to data science, I still continue to teach via private tuition. I am currently helping somebody prepare for the [STEP](https://www.admissionstesting.org/for-test-takers/step/about-step/), so I wanted to obtain as many past papers and solutions as I could. To do this manually would have required hours of clicking on links and 'Saving as...', so I decided to automate it with python.

## Downloading the step papers
This was straightforward. The website [stepdatabase.com](https://stepdatabase.maths.org) has past papers on them, and the urls and naming system they use is systematic and 100% consistent. I was able to download the papers using a simple loop with a `wget` command.  See the code below.

## Downloading the answers
This was less straightforward. [This website](https://mei.org.uk/step-aea-solutions) provides answers to the STEP papers but the urls did not have a consistent format, so I could not just use wget again. Therefore, I used some web-scraping tools to help me. 

There was a considerable learning curve, as I had not done any web scraping before. There were actually a couple of moments where I was going to give up. However, I persisted in the knowledge that if I want to be successful in a tech role, I will encounter such difficulties, and the only way to improve is to perservere.

In the end, I succeeded and the final code is given below. I will not go through the full process of trying things out, failing, tweaking, de-bugging, etc., but I will provide some of the key learning points for me.

* BeautifulSoup is only suitable for statically generated pages. For dynamic ones, you can use Selenium.
* The `dir` function in python is extremely handy. I have to thank this [Kaggle tutorial](https://www.kaggle.com/colinmorris/working-with-external-libraries) for introducing me to this function. On two or three occasions, I wanted to do something, and by looking at the list of methods, I was able to find one which worked. The example I remember is `find_elements_by_partial_link_text`.
* StackOverFlow is extremely handy. I am not sure what I would have done without it.
* Don't make assumptions. I assumed that the answers would all have different file names, but that was not always the case. This meant that previously downloaded were sometimes over-written by later answers. Fortunately, the was apparent in the first few minutes of running the program, so I could stop the program, and quickly modify it to add my own naming convention.


## Code
The code to download the past papers is:

```python
import subprocess

for i in range(87,119):
    for j in range(1,4):
        i = i%100
        url = f"https://stepdatabase.maths.org/database/db/{i:02}/{i:02}-S{j}.pdf"
        subprocess.run(["wget", url])

for i in range(87,119):
    for j in range(1,4):
        i = i%100
        url = f"https://stepdatabase.maths.org/database/db/{i:02}/{i:02}-S{j}.tex"
        subprocess.run(["wget", url])
```

The (ugly) code I created to download the solutions is:

```python
from bs4 import BeautifulSoup
from selenium import webdriver
import requests
import re

# open browser and go to the url
browser = webdriver.Chrome()
URL = 'https://mei.org.uk/step-aea-solutions'
browser.get(URL)

# click button, which opens new tab, so move to new tab
browser.find_element_by_xpath('//button[text()="STEP Solutions"]').click()
browser.switch_to_window(browser.window_handles[1])

# click on link to go to next page
browser.find_element_by_link_text('STEP past paper worked solutions').click()

# obtain list of links in this page
# each of these refers to a group of step papers, e.g. STEP I, 2016-2019
elements = browser.find_elements_by_partial_link_text('STEP solutions')
groups = []
for element in elements:
    groups.append(element.get_attribute("href"))

# define regex pattern to help identify correct links
pattern = r"STEP (1|I)+: \d\d\d\d"

# loop through links in groups
for url_group in groups:
    # open the link
    browser.get(url_group)
    
    # obtain list of links and names of papers in this group.
    # this requires the regex pattern from above
    papers = []
    for paper in browser.find_elements_by_partial_link_text('STEP '):
        if re.match(pattern,paper.get_property('textContent')):
            papers.append([paper.get_attribute("href"), paper.get_property('textContent')])
    
    # loop through the list of papers
    for url_paper, paper_name in papers:
        # open link for an individual paper
        browser.get(url_paper)

        # obtain list of links for answers to individual questions
        questions = browser.find_elements_by_partial_link_text('Question ')

        # loop through the questions
        for question in questions:
            
            # open link to answer for individual question
            question.click()
            browser.switch_to_window(browser.window_handles[2])

            # download image
            # note that the url you end on is different to the url you use to get to this page
            url = browser.current_url
            filename = paper_name+'-'+url.split('/')[-1]
            r = requests.get(url, allow_redirects=True)
            open(filename, 'wb').write(r.content)

            # close browser and switch back to page with list of questions
            browser.close()
            browser.switch_to_window(browser.window_handles[1])
```
