---
toc: true
layout: post
description: I summarise the two main ideas in Sathe and Aggarwal's paper Similarity Forests.
categories: [data science]
title: Similarity trees and NaN trees
---

## Introduction
I summarise the two main ideas of the paper [Similarity Forests](http://saketsathe.net/downloads/simforest.pdf) by Sathe and Aggarwal: trees via similarities and trees with `NaN` values.

## Notation
This is a supervised learning setting so the dataset consists of a set of tuples `(X_i, y_i)` where:
* `i` ranges from `1` to `N`, so `N` is the number of datapoints
* Each `X_i` is a `D`-dimensional vector, so `D` is the number of features. The `d`th entry of `X_i` will be denoted `X_id`
* Each `y_i` is a scalar and is the target or prediction variable

## Trees via similarities
In a standard decision tree, the decision or rule at each node takes the following form:
* Let `d` be one of the possible feature columns
* Let `a` be a constant
* Then `X_i` is in the left or right child node depending on whether `X_id` is smaller than or bigger than `a`

At each step, the `d` and `a` are chosen to (greedily) minimise some loss function, e.g. Gini impurity.

For this process to be possible, we need to have access to the individual entries `X_id`. What Sathe and Aggarwal show is that you can construct a decision tree without access to these individual entries `X_id` as long as you have access to some form of similarity between your datapoints. This is analagous to the kernel trick for SVMs: you only need to know how to calculate the dot products between data points; you do not need to calculate the vector representation of each of the data points.

So suppose we have access to the similarities between all data points `S`, where `S_ij` is equal to the similarity between `i`th and `j`th datapoints. Then the decision rule at each node takes the following form:
* Let `j` and `k` be distinct values from `1` to `N`, corresponding to two of the datapoints
* Let `a` be a constant
* Then `X_i` is in the left or right child node depending on whether `S_ij - S_ik` is smaller than or bigger than `a`

The intuition for this is that you are projecting all your datapoints onto the line joining `X_j` and `X_k`, and then picking a point on the line to split the datapoints in two. For a derivation of why `S_ij - S_ik` corresponds to this projection, and a diagram that helps visualise this projection, you can read the original paper.

And that's it! There are various details I have skipped over (e.g. how to choose `j` and `k` at each node), but using `S_ij - S_ik` instead of `X_id` is the central idea.

## Trees with `NaN` values
What happens if `X_id` is a NaN value? The standard algorithm will not work. In the paper, they describe a possible way of dealing with this. I do not know if this paper is the first to come up with this idea, but it is where I found out about it so I am giving them the credit.

Before describing their solution, spend a couple of minutes thinking about how you might adjust the decision tree algorithm to deal with NaN values (without imputation).

...

...

...

No really, I recommend you give it some thought. You will learn more by actively engaging and thinking for yourself, rather than passively reading.

...

...

...

The idea in their paper is quite neat. In the standard decision tree, each node has two children nodes. To deal with with NaN values, just introduce a third child containing the NaN entries! From here, there are various ways of dealing with these NaN-nodes. In the paper, Sathe and Aggarwal make all these nodes be leaf nodes and the prediction at these NaN-nodes is equal to the modal class of its parent. Another option is to just treat these NaN-nodes like any other node, and continue to split on them.  I imagine the strategy one chooses will depend on the particular situation one is in, and whether there are any patterns in the NaN values.

## Conclusion
To summarise, the paper contains two main new ideas:
* To create decision trees from similarities, use `S_ij - S_ik` instead of `X_id` to decide how to split at each node.
* To deal with NaN values, create a third NaN-node.
