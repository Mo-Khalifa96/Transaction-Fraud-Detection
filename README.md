# Transaction Fraud Detection (Machine Learning for Classification)

## About The Project 
**This project utilizes machine learning algorithms for classification, particularly for transaction fraud detection. The dataset presented here consists of thousands of transaction records, in fact, more than 6 million transactions, and key information about each transaction, including the sender's and recipient's account balances before and after the transaction, the amount of money transferred, and whether the transaction was in fact fraudulent or not. The aim of this project is to build a machine learning classification model that can accurately detect transaction fraud. Further, it focuses on both prediction efficiency and model interpretability, and tries to leverage both. This project was originally completed as per required for the final project of my course, 'Supervised Machine Learning: Classification', offered online by IBM. Overall, it displays a wide variety of data analysis tasks, classification algorithms, and model interpretation techniques.**
<br>
<br>
**In line with the objectives of the project, the data is first prepared, statistically analyzed, and preprocessed, before being used to train and test different classifiers. Furthermore, given that fraudulent cases, as we will see, make up less than 1% of all transactions, thus resulting in an extremely skewed or imbalanced dataset, different techniques are applied to deal with this problem first before developing the classifiers. Particularly, oversampling techniques are used to try and generate more data points for the minority class (fraud), by which way to balance the classes in the dataset to ensure that the classifiers are not biased during learning and can indeed predict transaction fraud reliably. Along the way, different classifiers are trained, tested and their parameters tuned to try to identify the best one for the data. Also, a subset of data was reserved for testing or out-of-sample evaluation to estimate how each classifier is likely to perform in the real world with novel datasets, unseen during the model training. Finally, after selecting the best classifier, model interpretation techniques are applied, such as permutation feature importance and partial dependence plots, in order to better understand how the final classifier was making its predictions, or, that is, based on which factors it was classifying one transaction as fraudulent and another as not. The most impactful features are identified and analyzed in more detail using a partial dependency plot, which illustrates through visualization the nature and direction of the relationship between that given feature and the likelihood of fraud as discerned by the model.** <br>
<br>

**Overall, the project is broken down into five parts: <br>
&emsp; 1) Reading and Inspecting the Data <br>
&emsp; 2) Exploratory Data Analysis <br>
&emsp; 3) Data Preprocessing <br>
&emsp; 4) Model Development, Tuning, and Evaluation <br>
&emsp; 5) Model Interpretation** <be>

<br>
<br>


## About The Data 
**The dataset being used here was taken from Kaggle.com, a popular website for finding and publishing datasets. You can quickly access it by clicking [here](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset). It's a huge dataset that presents, as mentioned, thousands and thousands of transactions and whether or not they were deemed fraud by the relevant authority, which will be utilized as material for developing and training the classification models. Note however, given that the dataset is considerably large, with its csv file totalling a size of over 400MB, it will not be possible to upload it on here. Thus, I will upload only a sample of the data.** <br> 
<br> 
**You can view each column and its description in the table below:** <br> 

| **Variable**      | **Description**                                                                                         |
| :-----------------| :------------------------------------------------------------------------------------------------------ |
| **step**          | Represents a unit of time where 1 step = 1 hour                                                         |
| **type**          | Type of online transaction (Transfer, Payment, Debit, Cash-in, Cash-out)                                |
| **amount**        | The amount of money in a transaction                                                                    |
| **nameOrig**      | Name of the sender                                                                                      |
| **oldbalanceOrg** | The sender's balance before the transaction                                                             |
| **newbalanceOrig**| The sender's balance after the transaction                                                              |
| **nameDest**      | Name of the recipient                                                                                   |
| **oldbalanceDest**| The recipient's balance before the transaction                                                          |
| **newbalanceDest**| The recipient's balance after the transaction                                                           |
| **isFraud**       | Specifies if a transaction is fraud (1) or not fraud (0)                                                |
| **isFlaggedFraud**     | Indicates if a transaction was flagged as fraud (1) or not (0)                                     |

<br>
<br>

**Here's a sample of the dataset being analyzed:**
<br> 

<img src="transaction data screenshot.jpg" alt="https://github.com/Mo-Khalifa96/Transaction-Fraud-Detection/blob/main/transaction%20data%20screenshot.jpg" width="800"/>

<br>
<br> 

## Quick Access 
**To quickly access the project, I provided two links, both of which will direct you to a Jupyter Notebook with all the code and corresponding output, broken down and organized into individual sections and sub-sections. Each section is provided with thorough explanations and insights that gradually guide the unfolding of the project from one step to the next. As presented below, the first link allows you to view the project, with its code and corresponding output rendered and organized into different sections and cells. The second link allows you to view the code and its output too, however, in addition, it also allows you to interact with it directly and reproduce the results if you prefer so. To execute the code, please make sure to run the first two cells first in order to install and be able to use the Python packages for performing the necessary analyses. To run any given block of code, simply select the cell and click on the 'Run' icon on the notebook toolbar.**
<br>
<br>
<br>
***To view the project only, click on the following link:*** <br>
https://nbviewer.org/github/Mo-Khalifa96/Transaction-Fraud-Detection/blob/main/Transaction%20Fraud%20Detection%20%28ML%20for%20Classification%29.ipynb
<br>
<br>
***Alternatively, to view the project and interact with its code, click on the following link:*** <br>
https://mybinder.org/v2/gh/Mo-Khalifa96/Transaction-Fraud-Detection/main?labpath=Transaction%20Fraud%20Detection%20(ML%20for%20Classification).ipynb
<br>
<br>

