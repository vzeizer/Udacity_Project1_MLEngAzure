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

The dataset investigated [here](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) consists of investigating a bank marketing problem and predict if the client subscribed a term deposit (classification problem).

Attribute Information (from [here](https://archive.ics.uci.edu/ml/datasets/bank+marketing)):

Input variables:

### bank client data:
1. - age (numeric)
2. - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3. - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4. - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5. - default: has credit in default? (categorical: 'no','yes','unknown')
6. - housing: has housing loan? (categorical: 'no','yes','unknown')
7. - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8. - contact: contact communication type (categorical: 'cellular','telephone')
9. - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10. - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11. - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
### other attributes:
12. - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13. - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14. - previous: number of contacts performed before this campaign and for this client (numeric)
15. - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
### social and economic context attributes
16. - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17. - cons.price.idx: consumer price index - monthly indicator (numeric)
18. - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19. - euribor3m: euribor 3 month rate - daily indicator (numeric)
20. - nr.employed: number of employees - quarterly indicator (numeric)

###
 
**Output variable** (desired target):
21. - y - has the client subscribed a term deposit? (binary: 'yes','no')





We have run an AutoML run as well as a Logistic Regression algorithm with regularization to find the optimal model that can predict whether the client has subscribed a term deposit.

The best performing model was found through AutoML, whose model is the following:
**
Pipeline(memory=None,
         steps=[('datatransformer',
                 DataTransformer(enable_dnn=False, enable_feature_sweeping=True, 
                 feature_sweeping_config={}, feature_sweeping_timeout=86400, 
                 featurization_config=None, force_text_dnn=False, is_cross_validation=True, 
                 is_onnx_compatible=False, observer=None, task='classification', 
                 working_dir='/mnt/batch/tasks/shared/LS_root/mount...
                 PreFittedSoftVotingClassifier(classification_labels=array([0, 1]), 
                 estimators=[('0', Pipeline(memory=None, steps=[('maxabsscaler', MaxAbsScaler(copy=True)), 
                 ('lightgbmclassifier', 
                 LightGBMClassifier(min_data_in_leaf=20, n_jobs=1, problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), 
                 random_state=None))], verbose=False)), 
                 ('24', Pipeline(memory=None, steps=[('standardscalerwrapper', StandardScalerWrapper(copy=True, with_mean=False, with_std=False)), 
                 ('xgboostclassifier', XGBoostClassifier(booster='gbtree', colsample_bytree=1, eta=0.05, gamma=0, max_depth=6, max_leaves=0, 
                 n_estimators=200, n_jobs=1, objective='reg:logistic', problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), 
                 random_state=0, reg_alpha=0.625, reg_lambda=0.8333333333333334, subsample=0.8, 
                 tree_method='auto'))], verbose=False)), 
                 ('1', Pipeline(memory=None, steps=[('maxabsscaler', MaxAbsScaler(copy=True)), 
                 ('xgboostclassifier', XGBoostClassifier(n_jobs=1, problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), 
                 random_state=0, tree_method='auto'))], verbose=False)), 
                 ('21', Pipeline(memory=None, steps=[('standardscalerwrapper', StandardScalerWrapper(copy=True, with_mean=False, with_std=False)), 
                 ('xgboostclassifier', XGBoostClassifier(booster='gbtree', colsample_bytree=0.5, eta=0.2, gamma=0, max_depth=7, max_leaves=7, 
                 n_estimators=25, n_jobs=1, objective='reg:logistic', 
                 problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), 
                 random_state=0, reg_alpha=0, reg_lambda=0.20833333333333334, subsample=1, tree_method='auto'))], verbose=False)), 
                 ('18', Pipeline(memory=None, steps=[('standardscalerwrapper', StandardScalerWrapper(copy=True, with_mean=False, with_std=False)), 
                 ('xgboostclassifier', XGBoostClassifier(booster='gbtree', colsample_bytree=0.7, eta=0.1, gamma=0.1, 
                 max_depth=9, max_leaves=511, n_estimators=25, n_jobs=1, objective='reg:logistic', problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), 
                 random_state=0, reg_alpha=0, reg_lambda=1.7708333333333335, subsample=0.9, tree_method='auto'))], verbose=False)), 
                 ('14', Pipeline(memory=None, steps=[('standardscalerwrapper', StandardScalerWrapper(copy=True, with_mean=False, with_std=False)), 
                 ('xgboostclassifier', XGBoostClassifier(booster='gbtree', colsample_bytree=1, eta=0.3, gamma=0, max_depth=10, max_leaves=511, 
                 n_estimators=10, n_jobs=1, objective='reg:logistic', problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), 
                 random_state=0, reg_alpha=2.1875, reg_lambda=0.4166666666666667, subsample=0.5, tree_method='auto'))], verbose=False)), 
                 ('16', Pipeline(memory=None, steps=[('standardscalerwrapper', StandardScalerWrapper(copy=True, with_mean=False, with_std=False)), 
                 ('logisticregression', LogisticRegression(C=51.79474679231202, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False))], verbose=False))], flatten_transform=None, weights=[0.125, 0.125, 0.125, 0.125, 0.125, 0.25, 0.125]))],
         verbose=False)
**

A little hard to interpret, but it is a Voting classifier of XGboost and LightGBM, it has 0.9174 accuracy.


## Scikit-learn Pipeline

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


We have chosen the RandomParameterSampling (Random Search), which in contrast to grid search (brute force search over all the parameters), not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).
The greatest benefit of the random search is that the best (or near to the best) parameters can be found in a fast way without looping over the whole model parameters.


We use the Bandit Policy because it  is an early termination policy based on slack factor/slack amount and evaluation interval.
The policy early terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing training run.
As parameters, we use the **slack_factor** as to be 0.1 and the **delay_evaluation** as to be 5 (the number of intervals to delay evaluation).
This can be justified, since slack factor is the slack allowed with respect to the best performing training run ( Any run whose best metric is less than (1 / (1 + 0.15)) or 87\% of the best performing run willbe terminated).
Additionally, the **evaluation_interval**, the frequency for applying the Bandit policy, meaning that each time the training script logs the primary metric counts as one interval.

## AutoML

Following is the architecture of the pipeline found by automl.

**
Pipeline(memory=None,
         steps=[('datatransformer',
                 DataTransformer(enable_dnn=False, enable_feature_sweeping=True, 
                 feature_sweeping_config={}, feature_sweeping_timeout=86400, 
                 featurization_config=None, force_text_dnn=False, is_cross_validation=True, 
                 is_onnx_compatible=False, observer=None, task='classification', 
                 working_dir='/mnt/batch/tasks/shared/LS_root/mount...
                 PreFittedSoftVotingClassifier(classification_labels=array([0, 1]), 
                 estimators=[('0', Pipeline(memory=None, steps=[('maxabsscaler', MaxAbsScaler(copy=True)), 
                 ('lightgbmclassifier', 
                 LightGBMClassifier(min_data_in_leaf=20, n_jobs=1, problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), 
                 random_state=None))], verbose=False)), 
                 ('24', Pipeline(memory=None, steps=[('standardscalerwrapper', StandardScalerWrapper(copy=True, with_mean=False, with_std=False)), 
                 ('xgboostclassifier', XGBoostClassifier(booster='gbtree', colsample_bytree=1, eta=0.05, gamma=0, max_depth=6, max_leaves=0, 
                 n_estimators=200, n_jobs=1, objective='reg:logistic', problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), 
                 random_state=0, reg_alpha=0.625, reg_lambda=0.8333333333333334, subsample=0.8, 
                 tree_method='auto'))], verbose=False)), 
                 ('1', Pipeline(memory=None, steps=[('maxabsscaler', MaxAbsScaler(copy=True)), 
                 ('xgboostclassifier', XGBoostClassifier(n_jobs=1, problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), 
                 random_state=0, tree_method='auto'))], verbose=False)), 
                 ('21', Pipeline(memory=None, steps=[('standardscalerwrapper', StandardScalerWrapper(copy=True, with_mean=False, with_std=False)), 
                 ('xgboostclassifier', XGBoostClassifier(booster='gbtree', colsample_bytree=0.5, eta=0.2, gamma=0, max_depth=7, max_leaves=7, 
                 n_estimators=25, n_jobs=1, objective='reg:logistic', 
                 problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), 
                 random_state=0, reg_alpha=0, reg_lambda=0.20833333333333334, subsample=1, tree_method='auto'))], verbose=False)), 
                 ('18', Pipeline(memory=None, steps=[('standardscalerwrapper', StandardScalerWrapper(copy=True, with_mean=False, with_std=False)), 
                 ('xgboostclassifier', XGBoostClassifier(booster='gbtree', colsample_bytree=0.7, eta=0.1, gamma=0.1, 
                 max_depth=9, max_leaves=511, n_estimators=25, n_jobs=1, objective='reg:logistic', problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), 
                 random_state=0, reg_alpha=0, reg_lambda=1.7708333333333335, subsample=0.9, tree_method='auto'))], verbose=False)), 
                 ('14', Pipeline(memory=None, steps=[('standardscalerwrapper', StandardScalerWrapper(copy=True, with_mean=False, with_std=False)), 
                 ('xgboostclassifier', XGBoostClassifier(booster='gbtree', colsample_bytree=1, eta=0.3, gamma=0, max_depth=10, max_leaves=511, 
                 n_estimators=10, n_jobs=1, objective='reg:logistic', problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), 
                 random_state=0, reg_alpha=2.1875, reg_lambda=0.4166666666666667, subsample=0.5, tree_method='auto'))], verbose=False)), 
                 ('16', Pipeline(memory=None, steps=[('standardscalerwrapper', StandardScalerWrapper(copy=True, with_mean=False, with_std=False)), 
                 ('logisticregression', LogisticRegression(C=51.79474679231202, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False))], verbose=False))], flatten_transform=None, weights=[0.125, 0.125, 0.125, 0.125, 0.125, 0.25, 0.125]))],
         verbose=False)
**

A "little" hard to interpret, but it is a Voting classifier of XGboost and LightGBM, it has 0.9174 accuracy.
It has many steps of data rescaling strategies and data model ensembling.


## Pipeline comparison

The best model with AutoML was **SoftVoting classifier with 0.9174 accuracy**, while the hyperdrive found a LR with the following best run metrics: {'Regularization Strength:': 0.1, 'Max iterations:': 100, 'Accuracy': 0.9111785533636824}.
The accuracy of the AutoML model was found to be slightly better because of the fact that the LightGBM classifier can capture more complex than a simple LR.
However, the LR provided a great result and the LR is a simple model, highly interpretable and the model would be light and easy to put into production. 
The architecture found by AutoML is much more complex and it resembles ensemble models.

## Future work

Future improvements of the current project could be:

- Run the AutoML experiment for a much longer time;
- perform different feature engineering and data cleaning strategies before training the model through hyperdrive;
- instead of just training a LR algorithm in the hyperdrive (in the train.py), use a greater variety of classification algorithms, such as KNN (K-Nearest Neighbors) and Decision Tree Classifiers;
- We could use a Bayesian Grid Search from AzureML to perform the hyperparameter tuning, and compare it to the Random Search in terms of efficiency and accuracy.
