## Co-training Regressors (COREG) in Python


Implementation of Co-training Regressors (COREG) semi-supervised regression algorithm from [Semi-Supervised Regression with Co-Training](http://dl.acm.org/citation.cfm?id=1642439) by Zhou and Li (IJCAI, 2005).


Original implementation author: [Neal Jean](https://github.com/nealjean) 


## Description

The main idea of the CoReg algorithm is to fit two estimators with different parameters, which are then used to make predictions on unlabeled data. Predictions of the first estimator with high confidence (see the article for details) are added to the training set of the second estimator, and vice versa until a certain criteria is met.

In theory, this should be helpful when there's a lot of unlabeled examples, but few labeled ones.
