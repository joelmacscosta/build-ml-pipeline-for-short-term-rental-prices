# Build an ML Pipeline for Short-term Rental Prices in NYC

This project builds and deploys an end-to-end rental price prediction pipeline using scikit-learn, MLflow, and Weights & Biases. The main focus is on MLOps-tracking experiments, managing artifacts, and deploying the pipeline—while EDA and modeling play a smaller role.


This project was developed as part of the requirements for the course: [Building a Reproducible Model Workflow](https://www.udacity.com/enrollment/cd0581)


---

## MLOps Tools Used for this Project

* MLflow to orchestrate the pipeline and store model artifacts
* Weights & Biases to track experiments and version datasets, metrics, and configs
* Conda to manage dependencies for each pipeline step
* Hydra to handle configuration
* Scikit-learn for modeling
* Pytest to validate the cleaned data
* GitHub for version control and releases

## Project Links

 * [Weights & Biases NYC rental properties project](https://wandb.ai/jcsc-usp/nyc_airbnb)
 * [Github repository](https://github.com/joelmacscosta/build-ml-pipeline-for-short-term-rental-prices)

## Overview of the Pipeline Component and Code Organisation 

```
├── components
│   ├── get_data
│   ├── test_regression_model
│   └── train_val_test_split
├── cookie-mlflow-step
│   └── {{cookiecutter.step_name}}
├── images
└── src
    ├── basic_cleaning
    ├── data_check
    ├── eda
    └── train_random_forest
```

## Instructions 

This will run the entire pipeline.

```bash
>  mlflow run .
```

Override any config parameter using Hydra via hydra_options:

```bash
> mlflow run . \
  -P steps=download,basic_cleaning \
  -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```

### 1.Exploratory Data Analysis (EDA)
* Component ``get_data`` 

Basic EDA performed in a notebook using data-profiling report.

**To run this pipeline component:**
```bash
> mlflow run . -P steps=download
```


### 2.Data Cleaning
* Component ``basic_cleaning`` 

Performs basic cleaning and filtering on the price feature and it range:
```
min_price: 10  # dollars
max_price: 350  # dollars
```

**To run this pipeline component:**
```bash
> mlflow run . -P steps=basic_cleaning
```


### 3.Data Testing
* Component ``data_check`` 

After the cleaning, a set of test validate the resulting dataset characteristics:
```
    test_column_names
    test_neighborhood_names
    test_proper_boundaries
    test_similar_neigh_distrib
    test_row_count
    test_price_range
```

**To run this pipeline component:**
```bash
> mlflow run . -P steps=data_check
```


### 4.Data Splitting
* Component ``train_val_test_split``

Splits the date is an training, validation and testing set.

**To run this pipeline component:**
```bash
> mlflow run . -P steps=data_split
```

### 5.Train Random Forest
* Component ``train_random_forest``

Train the Random Forest model with the with the following best tuned hyper parameters:
```
    max_tfidf_features: 30 # old 5
    n_estimators: 100
    max_depth: 15
    min_samples_split: 4
    min_samples_leaf: 3
    n_jobs: -1
    criterion: squared_error
    max_features: 0.5
    oob_score: true 
```

**To run this pipeline component:**
```bash
> mlflow run . -P steps=train_random_forest
```

### 6.Optimize Hyper-parameters
Re-runs the entire pipeline varying the hyper-parameters of the Random Forest model. 
We use the multi-run feature (adding the `-m` option 
at the end of the `hydra_options` specification), and a grid search with the parameters: `modeling.max_tfidf_features` and `modeling.random_forest.max_features`

**To run this pipeline component:**
```bash
> mlflow run . \
   -P steps=train_random_forest \
   -P hydra_options="modeling.random_forest.max_depth=10,50,100 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 modeling.max_tfidf_features=10,15,30 -m"
```


### 7.Select Best Model

From W&B interface and we select the best performing model considering the Mean Absolute Error as the main evaluation metric.

### 8. Evaluate Model

* Component ``test_regression_model``
Tests the production model against the test set. 
With our tuned hyper parameter with no feature engineering we obtain the following on the test set:
    * MAE ~ 31.7
    * R^2 ~ 0.59

**To run this pipeline component:**
```bash
> mlflow run . -P steps=test_regression_model
```

### 9. Test Final Release

On the following test set, the results were:

   * MAE ~ 31.7
   * R^2 ~ 0.59

```
 mlflow run https://github.com/joelmacscosta/build-ml-pipeline-for-short-term-rental-prices.git \
  -v 1.0.1 \
  -P hydra_options="etl.sample='sample2.csv'"
```


