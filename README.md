# CreditRisk_Classification

  Our Classifier makes it possible to analyze loan data collected by a bank in order to determine whether a customer is able to repay his credit or not. Our dataset consists of 1000 clients with attributes (categorical and numeric). Thanks to our model that we have implemented in the form of an API, we are able to tell if a customer is good or bad with regard to the repayment of his bank loan. And this has been possible thanks to the different machine learning models. The ones we have chosen for production are: random forest, logistic regression and Adaboost.
Throughout this repo, we will present the different stages of our work to you.
  
  NB: You will find in this repository a jupyter notebook, in which we have detailed all our different methods and results obtained
  
 # Dataset
The dataset consists of 1000 datatpoints each with 20 variables (dimensions) 7 are numerical and 13 are categorical. For a description of the dataset, please visit the site https://datahub.io/machine-learning/credit-g

 # Steps for building the model
1. Definition of objectives<br />
2. Acquisition and description of the dataset<br />
3. Exploratory data analysis<br />

    * Data inventory<br />
    * Detection of missing values<br />
    * Outlier detection<br />
    * Understanding of data (histograms, distributions, correlations, etc.)<br />

4. Data preparation (preprocessing)<br />

    * Feature split
    * Processing of missing values(imputation of numeric and categorical values),
    * Processing of outliers (IQR),
    * Encoding of categorical variables (hot encoding and hand encoding), 
    * Scaling of numeric variables (RobustScaler), 
    * Balancing of data (oversampling), 
    * Data split (60% training, 20% tests, 20% validation)

5. Modeling<br />

    * Model training (manage overfitting and underfitting, cross validation), 
    * Optimization of hyperparameters.
   
6. Evaluation and scoring (Iteration)<br />
    
    * Following metrics (Precision, Recall, Accurac, F1-score, Confusion matrix), 
    * Final choice of models for setting in production (random forest, logistic regression, AdaBoost)
8. Deployment<br />
  For the production of our models, we used the fastApi framework and the Docker container.
  
  
  

