---
toc: true
layout: post
description: I end this project by summarising what I did and summarising what I learnt by having a look at other people's examples on Kaggle.
categories: [python, data science]
title: Credit Card Fraud, Part VI, Summary and Lessons from Kaggle
image: "images/creditcard_6_xgb_roc2.png"
---
## Summaries
### Part I
In Part I, I described the framework and created the first set of models using default settings. I tried logistic regression, decision tree, random forest and xgboost models, and they respectively achieved an AUPRC of 0.616, 0.746, 0.842 and 0.856. Since then, I have learnt about more models and if I were to do this project again, I would also have included a support vector machine model and a k-nearest-neighbour model.

![]({{ site.baseurl }}/images/creditcard_1_logistic.png)
![]({{ site.baseurl }}/images/creditcard_1_tree.png)
![]({{ site.baseurl }}/images/creditcard_1_forest.png)
![]({{ site.baseurl }}/images/creditcard_1_xgb.png)

### Part II
In Part I, it look some time to fit the models, the forest model in particular, and I wanted to do some hyper-parameter optimisations. I wanted to find out if I could reduce the time taken to fit by removing non-fraudulent claims. The results of the experimentation showed that the time to fit was proportional to the size of the training data set, but the AUPRC did not take a massive hit.  This is good because it means I can do more hyper-parameter optimisations than before.

![]({{ site.baseurl }}/images/creditcard_2_forest_aucs.png)
![]({{ site.baseurl }}/images/creditcard_2_forest_times.png)

### Part III
Though the models were able to identify fraudulent transactions, I had gained no understanding. I tried creating a simple model: for each feature, determine whether the value is closer to the fraudulent mean or the non-fraudulent mean. This achieved an AUPRC of 0.682 and was able to identify about 70% of the frauduluent claims. This was satisfying, and better lets me appreciate what is gained by using more sophisticatd models.

![]({{ site.baseurl }}/images/creditcard_3_2.png)

### Part IV
I started doing some hyper-parameter optimisations on the forest model, and noticed the AUPRC varied a lot between the different folds. I decided to investigate how the AUPRC can vary, to better appreciate what is gained by choosing one hyper-parameter over another. After doing this, I could confidently say that choosing 50 estimators is better than the default of 100 estimators.

![]({{ site.baseurl }}/images/creditcard_4_forest_n_est50_hist.png)
![]({{ site.baseurl }}/images/creditcard_4_forest_n_est50_scatter.png)

### Part V
Here I actually carry out the hyper-parameter optimisations, and train the final models. The random forest's AUPRC increased from 0.842 to 0.852, and the xgboost's AUPRC increased from 0.856 to 0.872. Modest gains, and from the few articles I have read, this is to be expected.

![]({{ site.baseurl }}/images/creditcard_5_forest.png)
![]({{ site.baseurl }}/images/creditcard_5_xgb2.png)



## Lessons learnt from Kaggle
I had a skim through the several most up-voted kernels on Kaggle. Below are the the things I found out by doing so.

### AUROC versus AUPRC
Many of the examples (including the most upvoted [example](https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/notebook)!) use AUROC instead of AUPRC. The main reason this surprised me is that the desription of the dataset recommended using AUPRC; I suppose there was an advantage to not knowing much before hand! The second reason this surprised me is that AUPRC is a more informative measure than AUROC for unbalanced data. I try to explain why.

The PRC and ROC are quite similar. They are both plots that visualise false positives against false negatives.
* False negatives are recorded in the same way in both plots, namely, using recall/true positive rate. Recall tells you what percentage of truly fraudulent transactions that the model successfully labels as fraudulent. (And so 1 - Recall measures how many false negatives we have, as a percentage of truly fraudulent claims.)
* False positive are recorded differently in the two plots.
    * In PRC, precision is used. This is the percentage of transactions labelled as fraudulent that actually are fraudulent. Equivalently, 1-PRC is the number of false positives expressed as a percentage of *claims labelled as fraudulent*.
    * In ROC, the false-positive rate is used. This is the number of false positives expressed  s a percentage of *truly non-fraudulent transactions*.

To make this more concrete, lets put some numbers to this:
* Imagine there are 100100 transactions altogether, 100 which are fraudulent and 100000 which are not.
* Suppose a model predicts there are 200 fraudulent claims, and further suppose 50 of these were correct and 150 of these were incorrect.
* For both PRC and ROC, the true positive measurement would be 50%: 50% of the fraudulent claims were found.
* For PRC, the false positive measurement is 75%: 75% of the claims labelled as fraudulent were incorrectly labelled.
* For ROC, the false positive measurement is 0.15%: only 0.15% of the non-fraudulent claims were incorrectly labelled as fraudulent.

In short, ROC is much more forgiving of false positives than PRC, when we have highly unbalanced data.

I have also decided to plot PRC and ROC for a couple of the models in this series of posts, so you can visually see the difference. (Note that I have rotated the ROC curve to match up the variables with the PRC curve, so the comparison is easier.)

**PRC and ROC for the final XGBoost model**
![]({{ site.baseurl }}/images/creditcard_5_xgb2.png)
![]({{ site.baseurl }}/images/creditcard_6_xgb_roc2.png)
