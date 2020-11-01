---
toc: true
layout: post
description: I describe my first ever open-source contribution!
categories: [github, python]
title: Contributing to Darts by Unit8
---


## Introduction
A few days ago, I successfully completed my [first ever open source contribution](https://github.com/unit8co/darts/pull/163). :)

![image]({{ site.baseurl }}/images/firstopen.png)

In this article, I will briefly describe the experience, and some little things I learnt.

## Summary of process
I contributed to the relatively new package [darts](https://github.com/unit8co/darts) by [Unit8](https://unit8.co/). It is a package that makes TimeSeries analysis easy. It does not create any new models, but just creates a TimeSeries class which uses the `fit` and `predict` syntax from `scikitlearn` to carry out analyses imported from other packages.

I went to the issues tab and saw an issue that seemed accessible, implementing a `map` method for the Time Series object. Next I read the contribution guidelines, to make sure I follow the right steps and conventions. Next I forked the repository and then pulled it to my local machine.  After this, I spent some time getting familiar with the package and working out how the TimeSeries class works. Next I wrote the code for the map function. I would refer to other bits of the code to make sure I was using consistent conventions and wording (e.g. wording for the doc text). Then I tested the code by creating a Jupyter Notebook, and seeing that the function does what was intended. Once it all looked fine, I then pushed the changes to my forked copy of the repository, and then made a pull request.

It was a bit nerve-racking, because there are so many little things to get correct to successfully contribute to a project. Shortly after, I got some feedback, and fortunately it was positive! Phew. They liked how the function was implemented, but had various minor comments. I made the changes, pushed them to my forked copy and was about to make another pull request. But then I noticed that the original pull request automatically absorbs any new commits - that is a neat feature!

After several other minor comments and changes, the owners of the project were happy and merged the changes into the main code. Woohoo! My first ever contribution is done.

## Things I learnt
Like I said above, there are many little things to get correct to successfully contribute. Here I list several of the details and things I learnt about as a result.
*

## The code
In case you care, here is the code I wrote for the map function:

```python
def map(self,
        fn: Callable[[np.number], np.number],
        cols: Optional[Union[List[int], int]] = None) -> 'TimeSeries':
"""
    Applies the function `fn` elementwise to all values in this TimeSeries, or, to only those
    values in the columns specified by the optional argument `cols`. Returns a new
    TimeSeries instance.

    Parameters
    ----------
    fn
        A numerical function
    cols
        Optionally, an integer or list of integers specifying the column(s) onto which fn should be applied
    Returns
    -------
    TimeSeries
        A new TimeSeries instance
    """
    if cols is None:
        new_dataframe = self._df.applymap(fn)
    else:
        if isinstance(cols, int):
            cols = [cols]
        raise_if_not(all([0 <= index and index < self.width for index in cols]),
                 'The indices in `cols` must be between 0 and the number of components of the current '
                 'TimeSeries instance - 1, {}'.format(self.width - 1), logger)
        new_dataframe = self.pd_dataframe()
        new_dataframe[cols] = new_dataframe[cols].applymap(fn)
    return TimeSeries(new_dataframe, self.freq_str())
```
