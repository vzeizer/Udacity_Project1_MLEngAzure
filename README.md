# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**

The dataset investigated [here](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) consists of investigating a bank marketing problem and predict if the client subscribed a term deposit (classification problem).

Attribute Information (from [here](https://archive.ics.uci.edu/ml/datasets/bank+marketing)):

Input variables:

### bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
### other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
### social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

### 
**Output variable** (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')




**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

We have run an AutoML run as well as a Logistic Regression algorithm with regularization to find the optimal model that can predict the...

The best performing model was ...
## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

Essentially, the following steps have been performed:

- data collection (via URL);
- data cleaning (clean data function);
- split the dataset into train and test data;
- model training along with hyperparameter tuning with Hyperdrive;
- evaluation of the best model score on the test data.

The algorithm that has been used to be tuned was the Logistic Regression (LR).
Basically, the LR fits a sigmoid function to the label y (has the client subscribed a term deposit? (binary: 'yes','no')) as a function of the input features.
Mathematical details of the LR algorithm can be found [here](https://medium.com/data-science-group-iitr/logistic-regression-simplified-9b4efe801389).
Additionally, we have added an inverse regularization parameter, **"C"**, ranging from 0.1 to 100 in steps of 10 to survey a vast range of possible values for inverse regularization strength.
The other parameter that we have added was the **"max_iter"** as to be either 100, or 500 or 1000 to explore a vast number of maximum iterations for the LR algorithm to converge.

**What are the benefits of the parameter sampler you chose?**

We have chosen the RandomParameterSampling (Random Search), which in contrast to grid search (brute force search over all the parameters), not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).
The greatest benefit of the random search is that the best (or near to the best) parameters can be found in a fast way without looping over the whole model parameters.


**What are the benefits of the early stopping policy you chose?**

We use the Bandit Policy because it  is an early termination policy based on slack factor/slack amount and evaluation interval.
The policy early terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing training run.
As parameters, we use the **slack_factor** as to be 0.1 and the **delay_evaluation** as to be 5 (the number of intervals to delay evaluation).
This can be justified, since slack factor is the slack allowed with respect to the best performing training run ( Any run whose best metric is less than (1 / (1 + 0.15)) or 87\% of the best performing run willbe terminated).
Additionally, the **evaluation_interval**, the frequency for applying the Bandit policy, meaning that each time the training script logs the primary metric counts as one interval.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

Future improvements of the current project could be:

- Run the AutoML experiment for a much longer time;
- perform different feature engineering and data cleaning strategies before training the model;
- instead of just training a LR algorithm (in the train.py), use a greater variety of classification algorithms, such as KNN (K-Nearest Neighbors) and Decision Tree Classifiers;
- We could use a Bayesian Grid Search from AzureML to perform the hyperparameter tuning, and compare it to the Random Search in terms of efficiency and accuracy.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. 
Otherwise, delete this section.**
**Image of cluster marked for deletion**
