---
toc: true
layout: post
description: There was a major bug in my first attempt at scraping!
categories: [python, scraping]
title: Web Scraping for STEP past papers and solutions, Part II, a bug
---
## Other posts in series
{% for post in site.posts %}
{% if (post.title contains "Scraping for STEP") and (post.title != page.title) %}
* [{{ post.title }}]({{ site.baseurl }}{{ post.url }})
{% endif %}
{% endfor %}

## A bug
Earlier this week, I tried to open one of the files containing a solution to a past STEP paper. To my surprise, it did not open. I tried several other files, and none of them opened. A big lesson here is to actually check the files have properly downloaded! I just assumed that the files successfully downloaded because I saw the correct filenames appear in the directory. (This is somewhat ironic because one of my takeaways at the end of the previous post was to not make assumptions...)

So I had to go back to my previous attempt and try to work out what went wrong. I compared my code to various other examples online, and I could not see the error. I then tried several other variations that I found while searching, and none of them worked.

Next I tried to look at the object obtained after running `r = requests.get(url)`. I first ran `print(r.content)` to actually see what I was writing to the files. The output was html for a webpage - that made no sense, since the url directly goes to a jpg file. I was feeling a bit clueless here. I used `dir(r)` to see if there were any functions that might help me, but none stood out. One of the methods was `url` and in desperation I decided to do run `print(r.url)`, expecting just to the url that was inputted into the `.get` method. However, the output was:

`https://2017.integralmaths.org/login/index.php`

Bingo! The problem was now clear. The `requests` function is distinct from the selenium objects, so you have to separately log-in to the website for `requests`. After some Googling, I found out how you can enter log-in credentials using requests, and it worked! What a relief. The new bits of code are given at the end.

## Lessons learnt
* As mentioned above, I really need to absorb the lesson that one should not make assumptions.
* If you're completely stuck, it might be worthwhile to go through all the methods of the object, to see what can be uncovered.
* There is a limit to my understanding of the code and the modules. Stitching together code from random blogs and stackoverflow works, but I have no real understanding. I did try looking at the documentation for some of the modules I was using, but they are incomprehensibly dense. I am not sure what best practice is here.
* Try to have as much of your work saved, so if there is a bug, you do not need to repeat everything. In this case, I should have saved a list of the urls during my first attempt, so that for any potential future attempts, all I would have to do is loop through these urls and download the images, rather than having to navigate via a browser to find all the urls again.

## Code
Below is the new bits of code I had to add into the original code. It may be incomprehensible without the context, but I think the syntax is mostly self-explanatory.

```python
login_url = "https://2017.integralmaths.org/login/index.php"
payload = {
    'username': 'mei-step',
    'password': 'Stepaea1'
}

with requests.Session() as s:
    s.post(login_url, data=payload)
    r = s.get(url, allow_redirects=True, stream=True)
    open(filename, 'wb').write(r.content)
```
