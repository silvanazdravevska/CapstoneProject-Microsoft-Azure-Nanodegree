
# Capstone Project - Azure Machine Learning Engineer

In this Project 2 models are created, one using Automated ML ant the other using Hyperdrive in Microsoft Azure Python SDK. The dataset used is the Hearth Failure Prediction dataset from Kaggle in order to built a classification model to predict mortality by heart failure.Then the model with the best accuracy will be deployed as a Web service.

## Dataset

### Overview

[Dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) from Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020)

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

### Task

In this project a classification model will be built both with Automated Machine Learning and a Hyperdrive model with tuned hyperparameters to determine the best model for prediction of death events based on 12 features:

-age: age of the patient (years)

-anaemia: decrease of red blood cells or hemoglobin (boolean)

-high blood pressure: if the patient has hypertension (boolean)

-creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)

-diabetes: if the patient has diabetes (boolean)

-ejection fraction: percentage of blood leaving the heart at each contraction (percentage)

-platelets: platelets in the blood (kiloplatelets/mL)

-sex: woman or man (binary)

-serum creatinine: level of serum creatinine in the blood (mg/dL)

-serum sodium: level of serum sodium in the blood (mEq/L)

-smoking: if the patient smokes or not (boolean)

-time: follow-up period (days)

Both models will train the data and that is used to compare the performance of both the models, and deploy the model with the best accuracy and then test the model endpoint. Both models were trained in the Microsoft Azure Python SDK.

### Access

The Heart Failure Dataset is downloaded from Kaggle as a csv file on local computer, and after it is registered via ML Studio as a Dataset in the Azure Workspace in a Tabular form, uploaded as local file. It is then accessed with `Dataset.get_by_name(ws, dataset_name)` in the Python SDK.

## Automated ML

The main goal of classification models is to predict which categories new data will fall into based on learnings from its training data.  The `primary metric` parameter determines the metric to be used during model training for optimization; if not specified, accuracy is used for classification tasks for Image classification, Sentiment analysis, Churn prediction. `enable_early_stopping` will flag to enable early termination if the score is not improving in the short term.

The `automl` settings and configuration used for this experiment are as follows:
```
automl_settings = {"experiment_timeout_minutes": 30,
    "max_concurrent_iterations": 3,
    "primary_metric" : 'accuracy',
    "n_cross_validations": 4}

automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=data_train,
                             label_column_name="DEATH_EVENT",   
                             enable_early_stopping= True,
                             featurization= 'auto',
                             **automl_settings)
```                             
The main goal of classification models is to predict which categories new data will fall into based on learnings from its training data. AutoML typically performs cross validation, data balance check, cardinality check in prior to machine learning process with a variety of algorithms.

### Results

The results I got with this automated ML model is the VotingEnsemble with Accuracy of ~0.885946.Voting Ensemble technique predicts based on the weighted average of predicted class probabilities for classification tasks. `ensembled_iterations`: '[23, 17, 2, 29, 9, 5, 24, 26]', `ensembled_algorithms`: "[`LightGBM`, `ExtremeRandomTrees`, `RandomForest`, `LightGBM`, `RandomForest`, `LightGBM`, `LightGBM`, `RandomForest`]", `ensemble_weights`: '[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1]', `best_individual_pipeline_score`: '0.8733050847457627', `best_individual_iteration` '23'. To improve the model we can use different target metric like AUC_weighted.

![AutoML RunDetails Widget](https://raw.githubusercontent.com/silvanazdravevska/CapstoneProject-Microsoft-Azure-Nanodegree/main/starter_file/Screenshots/AutoML_RunDetails_Widget.jpg)

![](https://raw.githubusercontent.com/silvanazdravevska/CapstoneProject-Microsoft-Azure-Nanodegree/main/starter_file/Screenshots/AutoML_Best_Model_Run_ID_1.jpg)

![](https://raw.githubusercontent.com/silvanazdravevska/CapstoneProject-Microsoft-Azure-Nanodegree/main/starter_file/Screenshots/AutoML_Best_Model_Run_ID_2.jpg)

## Hyperparameter Tuning

The model for this experiment is Liner Regression, trains easy, fast and is easy to understand.Parameters used for hyperparameter tuning are: Regularization Strength (C) with range 0.0 to 1.0 -- Inverse of regularization strength. Smaller values cause stronger regularization and Max Iterations (max_iter) with values 50, 100, 150, 200 and 250 -- Maximum number of iterations to converge.

```
early_termination_policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

if "training" not in os.listdir():
    os.mkdir("./training")

#TODO: Create the different params that you will be using during training
param_sampling = RandomParameterSampling(
    {
        '--C': uniform(0.0, 1.0), 
        '--max_iter': choice(50, 100, 150, 200, 250)
    }
)

#TODO: Create your estimator and hyperdrive config
estimator = SKLearn(source_directory = "./",
            compute_target=compute_target,
            vm_size='STANDARD_D2_V2',
            entry_script="train.py")

hyperdrive_run_config = HyperDriveConfig(estimator=estimator,
                                     hyperparameter_sampling=param_sampling,
                                     policy=early_termination_policy,
                                     primary_metric_name='Accuracy',
                                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                     max_total_runs=20,
                                     max_concurrent_runs=4)
```

Hyperparameter tuning is the process of finding the configuration of hyperparameters that results in the best performance. Random sampling supports early termination of low-performance runs. The early termination policy uses the primary metric to identify low-performance runs. BanditPolicy terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.

### Results

The results I got with this hyperdrive model is an accuracy of ~0.7833. The parameters of the model are Regularization Strenght and Max Iterations. The results of the parameters were Regularization Strenght ~0.74, Max Iterations = 150. To improve it we can also try increasing the range of the hyperparameters and with imbalanced data we can do better pre-processing of the data or get more data to balance it.

![](https://raw.githubusercontent.com/silvanazdravevska/CapstoneProject-Microsoft-Azure-Nanodegree/main/starter_file/Screenshots/HyperDrive_RunDetails_Widget.jpg)

![](https://raw.githubusercontent.com/silvanazdravevska/CapstoneProject-Microsoft-Azure-Nanodegree/main/starter_file/Screenshots/HyperDrive_Best_Model_Run_ID_1.jpg)

![](https://raw.githubusercontent.com/silvanazdravevska/CapstoneProject-Microsoft-Azure-Nanodegree/main/starter_file/Screenshots/HyperDrive_Best_Model_Run_ID_2.jpg)

## Model Deployment

The AutoMl model is deployed using Azure Container Instance as a WebService. Best run environment and score.py file is provided to the InferenceConfig. The aci service is then created using workspace, aci service name, model, inference config and deployment configuration.

First the model is registered. A registered model is a logical container for one or more files that make up your model. After the registration, it can be downloaded or deployed and receive all the files that were registered. Then we need to define an inference configuration. An inference configuration describes how to set up the web-service containing your model. It's used later, when you deploy the model. Before deploying your model, you must define the deployment configuration. For this model deployment Azure Container Instances is the instance associated with my workspace. Then the model is deployed.

The model is successfully deployed as a web service and a REST endpoint is created with status Healthy - The service is healthy and the endpoint is available. A scoring uri is also generated to test the endpoint.

After deployment of the machine learning model as a web-service, the web-service-endpoint is queried by sending the request to it.
The endpoint is tested by using an endpoint.py file which passes 2 data points as json. Steps for querying the endpoint: Require scoring uri, json data and primary key.
Logistic Regression is a binary classification algorithm(0 or 1). It uses logistic function called the sigmoid function in order to predict outcomes.

![](https://raw.githubusercontent.com/silvanazdravevska/CapstoneProject-Microsoft-Azure-Nanodegree/main/starter_file/Screenshots/Model%20Deployment_1.jpg)

![](https://raw.githubusercontent.com/silvanazdravevska/CapstoneProject-Microsoft-Azure-Nanodegree/main/starter_file/Screenshots/Model-Deployment_2.jpg)

## Screen Recording

A [link](https://drive.google.com/file/d/1Zv3IOT9VT0iXe_sVxZE3LYurxfddZLuA/view?usp=sharing) to a screen recording of the project in action. 

##Future Improvements
Larger dataset can be used to increase data quality, Different models can also be used with hyperparameter tuning, Feature engineering can be performed using PCA, To improve the AutoML model we can use different target metric like AUC_weighted
