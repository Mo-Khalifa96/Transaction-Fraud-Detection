import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, precision_recall_fscore_support, roc_curve, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.kernel_approximation import Nystroem

import warnings
warnings.simplefilter("ignore")


#Defining functions for model evaluation and interpretation
#Defining functions to compute and report error scores
def error_scores(ytest, ypred, classes):
    error_metrics = {
        'Accuracy': accuracy_score(ytest, ypred),
        'Precision': precision_score(ytest, ypred, average=None),
        'Recall': recall_score(ytest, ypred, average=None),
        'F5': fbeta_score(ytest, ypred, beta=5, average=None) }

    return pd.DataFrame(error_metrics, index=classes).apply(lambda x:round(x,2)).T

def error_scores_dict(ytest, ypred, strategy):
    #create empty dict for storing results 
    error_dict = {}
    
    #specify type of result 
    error_dict['Strategy'] = strategy
    
    #Get accuracy score
    error_dict['Accuracy'] = round(accuracy_score(ytest, ypred),2)
    
    #Get Precision, recall, F-beta scores
    precision, recall, f_beta, _ = precision_recall_fscore_support(ytest, ypred, beta=5, average='binary')
    #store results
    error_dict['Precision'], error_dict['Recall'], error_dict['F5'] = round(recall,2), round(precision,2), round(f_beta,2) 
    
    return error_dict

#Defining a function to plot out confusion matrix
def plot_cm(ytest, ypred, classes):
    cm = confusion_matrix(ytest, ypred)
    fig,ax = plt.subplots()
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20, "weight": "bold"})
    labels = classes
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Prediction', fontsize=15)
    ax.set_ylabel('Actual', fontsize=15)
    plt.show()

#Defining a function to plot the ROC curve
def plot_ROC_curve(model, xtest, ytest):
    #Get model's predicted probabilities
    y_prob = model.predict_proba(xtest)
    y_pred = model.predict(xtest)
    #Get false positive and true positive rates
    false_pos_rate, true_pos_rate, thresholds = roc_curve(ytest, y_prob[:,1])
    #Get best auc score 
    auc_best = roc_auc_score(ytest, y_pred, average=None)

    #Plot the ROC curve
    fig, ax = plt.subplots()
    ax.plot(false_pos_rate, true_pos_rate, linewidth=2.5)
    plt.fill_between(false_pos_rate, true_pos_rate, alpha=0.1)
    
    #Plot the diagonal chance line 
    ax.plot([0, 1], [0, 1], ls='--', color='black', lw=.3) 
    plt.annotate('AUC=0.5', xy=(0.5, 0.5), xytext=(0.6, 0.3), 
                 arrowprops=dict(facecolor='black', headwidth=8, width=2.5, shrink=0.05))
    
    #Plot the best auc score
    ax.plot(auc_best, marker='o', color='r')
    plt.annotate(f'AUC={round(auc_best,2)}', xy=(0,auc_best), xytext=(0.1,0.8), 
                 arrowprops=dict(facecolor='gray', headwidth=7, width=2, shrink=0.15))
    
    #Set title and labels 
    ax.set(title='ROC curve',
           xlabel='False Positive Rate',
           ylabel='True Positive Rate')
    
    #add grid 
    ax.grid(True)
    plt.tight_layout()
    plt.show()

#Defining a function to plot ROC curve for multiple models 
def plot_ROC_curve_multiple(xtest, xtest_svc, ytest, estimators):
    #Get ROC curve for each model
    LR_fpr, LR_tpr, thresold = roc_curve(ytest, estimators[0].predict(xtest))
    KNN_fpr, KNN_tpr, threshold = roc_curve(ytest, estimators[1].predict(xtest))
    SVC_fpr, SVC_tpr, threshold = roc_curve(ytest, estimators[2].predict(xtest_svc))
    RF_fpr, RF_tpr, threshold = roc_curve(ytest, estimators[3].predict(xtest))
    
    #Obtain figure and set title and labels 
    plt.figure(figsize=(13,7))
    plt.title('ROC Curve per classifier', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    
    #Plot the ROC curve for each model 
    plt.plot(LR_fpr, LR_tpr, label='Logistic Regression classifier score: {:.2f}'.format(roc_auc_score(ytest, estimators[0].predict(xtest))))
    plt.plot(KNN_fpr, KNN_tpr, label='KNN classifier score: {:.2f}'.format(roc_auc_score(ytest, estimators[1].predict(xtest))))
    plt.plot(SVC_fpr, SVC_tpr, label='Kernel SVM classifier score: {:.2f}'.format(roc_auc_score(ytest, estimators[2].predict(xtest_svc))))
    plt.plot(RF_fpr, RF_tpr, label='Random Forest classifier score: {:.2f}'.format(roc_auc_score(ytest, estimators[3].predict(xtest))))    

    #plot the diagonal chance line (line of no discrimination)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.annotate('AUC=0.5', xy=(0.5, 0.5), xytext=(0.6, 0.3), fontsize=13, arrowprops=dict(facecolor='black', headwidth=8, width=2.5, shrink=0.07))
    plt.annotate('Line of no discrimination', xy=(0.4,0.45), xytext=(0.4,0.45), fontsize=12, rotation=25)
    
    #set axes and grid 
    plt.axis([-0.01, 1, 0, 1])
    plt.grid(True)
    
    #add legend 
    plt.legend()
    
    #display plot 
    plt.tight_layout()
    plt.show()
    
#Defining a function to visualize feature importances via box plot (for permutation feature importance)
def visualize_feature_importances(importance_array, cols):
    # Sort the array based on mean value
    sorted_idx = importance_array.importances_mean.argsort()      #sorts array from lowest to highest and returns their indices

    # Visualize the feature importances using boxplot
    fig, ax = plt.subplots()     #figsize=(16,10)
    fig.tight_layout()

    #create box plot 
    ax.boxplot(importance_array.importances[sorted_idx].T,
               labels=cols[sorted_idx], 
               vert=False)    #makes the box plot horizontal rather than vertical 
    
    #assign title 
    ax.set_title("Permutation Importances (training set)")
    #display figure 
    plt.show()


#Defining a random state for reproducible results
#specify random seed
rs = 10


#Part One: Reading and Inspecting the Data
#Loading and reading the dataset
#Access the data file
df = pd.read_csv("Online Transaction Fraud.csv")

#drop unnecessary columns 
df = df.drop(['isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1)

#inspect data shape
shape = df.shape
print('Number of coloumns:', shape[1])
print('Number of rows:', shape[0])

#preview first 10 entries 
df.head(10)

#due to limited computational resources, I will extract only a subset of data to work with (100,000 entries)
# through stratified shuffle splitting.


#Preview data distribution before data splitting 
#Create a histogram for each column separately 
fig,axes = plt.subplots(ncols=8, figsize=(20,8))
for col,ax in zip(df.columns, axes):
    ax.hist(df[col])
    ax.set_title(col)
    if col=='type':
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)        
plt.tight_layout()


#Data Sampling
#Using stratified shuffle splitting to extract a smaller sample of 100,000 entries only whilst simultaneously maintaining the same class distribution as the original 
#identify the target coloumn 
target = 'isFraud'
#stratified sampling and obtaining a new, smaller dataframe 
sample_inx, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=100000, random_state=rs).split(df[df.columns.drop(target)], df[target]))
x_data, y_data = df.loc[sample_inx, df.columns[:-1]], df.loc[sample_inx, df.columns[-1:]]
df = pd.concat([x_data, y_data], axis=1).reset_index(drop=True)

#report data shape again
shape = df.shape
print('Number of coloumns:', shape[1])
print('Number of rows:', shape[0])

#preview first 10 entries
df.head(10)


#Preview data distribution after data splitting 
#Create a histogram for each column separately 
fig,axes = plt.subplots(ncols=8, figsize=(20,8))
for col,ax in zip(df.columns, axes):
    ax.hist(df[col])
    ax.set_title(col)
    if col=='type':
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)        
plt.tight_layout()



#Part Two: EXPLORATORY DATA ANALYSIS 
#Descriptive Statistics
df.describe().apply(lambda x:round(x,2)).T

#Check data types and values count
df.info()

#Making sure there are no missing or null entries 
print(f'Number of missing entries per coloumn:\n{df.isnull().sum()}')
print()


#Identify variables that need preprocessing
#Get the total number of unique values for each variable (& data type)
for col in df.columns:
    print(f'{col} ({df[col].dtype}): {len(df[col].unique())}')

#We can see all of our variables are continuous variables, except for the two variables: 'type', which specifies the type of transaction, and the target 
# variable, 'isFraud'. Now I take a closer look at some of the important variables in the data and how they relate to the target variable.


#Bivariate Analysis
#Relationship between type of transaction and fraud
#first, checking frequency of each type of transaction again
df['type'].value_counts().plot(kind='bar', edgecolor='k')

#show fraud count by type of transaction (looking at a maximum count of 1000 only for comparison)
fraud_by_type = pd.crosstab(index=df['type'],columns=df['isFraud'])
fraud_by_type.plot(kind='bar', ylim=[0,1000])
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)


#Relationship between amount of money transferred and fraud
#Get dataframe with only fraud cases
df_fraud = df[df['isFraud']==1]

#Sort dataframe by amount 
fraud_by_amount = df_fraud.sort_values(by=['amount'], ascending=False)

#Create a histogram to show distribution of transaction amounts for fraud cases 
fraud_by_amount['amount'].plot(kind='hist', figsize=(10,5), edgecolor='black')
plt.xlabel('Transaction Amount (currency unspecified)')
plt.gcf().axes[0].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))


#Relationship between fraud and sender's balance before and after transaction
#Sort data by balance before transaction 
fraud_by_SenderBalanceBefore = df_fraud.sort_values(by=['oldbalanceOrg'], ascending=False)
fraud_by_SenderBalanceBefore = fraud_by_SenderBalanceBefore['oldbalanceOrg']

#Sort data by balance after transaction  
fraud_by_SenderBalancAfter = df_fraud.sort_values(by=['newbalanceOrig'], ascending=False)
fraud_by_SenderBalanceAfter = fraud_by_SenderBalancAfter['newbalanceOrig']

#Create histogram for balance before vs. after transaction 
fig,ax = plt.subplots(1,2, figsize=(12,6), sharex=True)
ax[0].hist(fraud_by_SenderBalanceBefore, edgecolor='black')
ax[1].hist(fraud_by_SenderBalanceAfter, edgecolor='black')
fig.suptitle('Sender\'s balance before vs. after fraud transaction', fontsize=16)
ax[0].set_title('Balance before fraud transaction')
ax[1].set_title('Balance after fraud transaction')
plt.gcf().axes[0].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.gcf().axes[1].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()


#Relationship between fraud and recipient's balance before and after transaction
#Sort data by recipient's balance before transaction 
fraud_by_RecipientBalanceBefore = df_fraud.sort_values(by=['oldbalanceDest'], ascending=False)
fraud_by_RecipientBalanceBefore = fraud_by_RecipientBalanceBefore['oldbalanceDest']

#Sort data by recipient's balance after transaction  
fraud_by_RecipientBalancAfter = df_fraud.sort_values(by=['newbalanceDest'], ascending=False)
fraud_by_RecipientBalanceAfter = fraud_by_RecipientBalancAfter['newbalanceDest']

#Create histogram for recipient's balance before vs. after transaction 
fig,ax = plt.subplots(1,2, figsize=(12,6), sharex=True)
ax[0].hist(fraud_by_RecipientBalanceBefore, color='orange', edgecolor='black')
ax[1].hist(fraud_by_RecipientBalanceAfter, color='orange', edgecolor='black')
fig.suptitle('Recipient\'s balance before vs. after fraud transaction', fontsize=16)
ax[0].set_title('Balance before fraudulent transaction')
ax[1].set_title('Balance after fraudulent transaction')
plt.gcf().axes[0].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.gcf().axes[1].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()



#Part Three: Data Preprocessing 
#In this section, I will make the necessary preperations and preprocessing for the data to make sure it is ready for analysis and model development. This 
# will involve: one-hot encoding to deal with the categorical variables, data selection and data splitting, feature scaling, and oversampling to balance 
# the classes in the dataset. 

#Dealing with Categorical Variables: One-Hot Encoding
#one-hot encoding the 'type' coloumn
df = pd.get_dummies(df, columns=['type'], drop_first=True, dtype='int')

#examine shape again 
print('Data shape:', df.shape, '\n\n') 
#preview the data again 
df.head()


#Data Selection
#Label the classes for later analysis 
classes = ['Not Fraud', 'Fraud']

#Identify predictors and target variables
features = df.columns.drop(target) 
x_data = df[features]
y_data = df[target]

#Examine Class Distribution 
#We can check the distribution of classes for the target variable, 'isFraud', to see the proportion of the majority class to that of the minority class.
#Get percentage of the non-fraud vs. fraud cases in the dataset
print('The percentage of normal vs. fraudulent transactions:\n')
print(y_data.value_counts(normalize=True).apply(lambda x: str(x*100)+'%'),'\n\n')
#Visualizing the class distribution in the data using count plot
sns.countplot(x=y_data)

#The two classes are extremely unbalanced, with fraud cases making up only about 0.13%! As such, I will now 
# perform stratified data splitting, followed by oversampling in order to ensure that the two classes in the 
# dataset are balanced and of the equivalent proportions.


#Stratified Data Splitting 
#Performing stratified data splitting (80% training/20% testing)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, stratify=y_data, random_state=rs)

#check the sizes of the training and testing sets
print('Number of training samples:', x_train.shape[0])    
print('Number of testing samples:', x_test.shape[0])


#Feature Scaling 
#First, identify numerical variables only for feature normalization 
numeric_vars = [col for col in df.columns if len(df[col].unique())!=2] 

#feature scaling numerical features 
Scaler = MinMaxScaler()
x_train[numeric_vars] = Scaler.fit_transform(x_train[numeric_vars])
x_test[numeric_vars] = Scaler.transform(x_test[numeric_vars])

#now we can look at the distribution of data after rescaling 
x_train.describe().drop(['25%','50%','75%']).apply(lambda x:round(x,2)).T


#Dealing with Imbalanced Classes: Oversampling
#Developing a simple logistic regression classifier and evaluating its performance before oversampling
#create logistic regression object
LR = LogisticRegression(solver='saga', random_state=rs, max_iter=500, n_jobs=10)
#fit the model
model = LR.fit(x_train, y_train)
#generate predictions
y_pred = model.predict(x_test)
#get error scores 
error_scores(y_test, y_pred, classes)


#Oversampling
#Get samplers 
samplers = [('ROS', RandomOverSampler(random_state=rs)), ('SMOTE',SMOTE(random_state=rs)), 
            ('Borderline SMOTE', BorderlineSMOTE(random_state=rs)), ('ADASYN', ADASYN(random_state=rs))]

#create empty dictionary to store results 
results=[]

for label,sampler in samplers:
    x_over, y_over = sampler.fit_resample(x_train, y_train)
    LR = LogisticRegression(solver='saga', random_state=rs, max_iter=500, n_jobs=10)
    model = LR.fit(x_over, y_over)
    y_pred = model.predict(x_test)
    result = error_scores_dict(y_test, y_pred, label)
    results.append(result)

results_table = pd.DataFrame(results).set_index('Strategy')
results_table

#we can see that Borderline SMOTE is the most appropriate oversampling method for the data, raising the F-beta score to 0.25 for the fraud class, 
# which is a significant feat considering the first model had an F-beta of 0.


#Borderline SMOTE Oversampling  
#Obtain new training data balanced with borderline SMOTE 
x_over, y_over = BorderlineSMOTE(random_state=rs).fit_resample(x_train, y_train)

#preview class distribution after oversampling  
print('The percentage of normal vs. fraudulent transactions:\n')
print(y_over.value_counts(normalize=True).apply(lambda x: str(x*100)+'%'),'\n\n')
sns.countplot(x=y_over)

#As illustrated, both classes have the same size now and are perfectly balanced. Now I will move 
# to model development, tuning, and selection.



#Part Four: Model Development, Tuning, and Evaluation 
#In this section, I will train, tune, and evaluate different classification models to predict transaction fraud. I will compare and contrast the performances 
# of each classifier using the same error metrics as above, and if necessary, I will develop a stacking ensemble model that leverages the strengths of the 
# individual models. The final model should ideally maximize recall as well as precision.

#Model One: Logistic Regression classifier 
#Model Development and Hyperparameter Tuning
#training and tuning a logistic regression classifier 
LR = LogisticRegression(solver='saga', max_iter=500, random_state=rs, n_jobs=10)
Grid_LR = GridSearchCV(LR, scoring='f1', cv=4, n_jobs=10, 
                            param_grid={
                                'penalty': ['l1','l2'], 
                                'C': [0.001, 0.01, 0.1, 1, 10, 100]
                            }).fit(x_over, y_over)

#Report best parameters after grid search
print('Best parameter values for the logistic regression classifier:')
Grid_LR.best_params_

#Model Evaluation
#get best estimator based off grid search 
LR_classifier = Grid_LR.best_estimator_

#Generate predictions 
y_pred = LR_classifier.predict(x_test)
#Compute and report error scores
print('Logistic Regression classification results:')
error_scores(y_test, y_pred, classes=classes)
#Visualize confusion matrix
plot_cm(y_test, y_pred, classes)


#Model Two: K-Nearest Neighbors classifier 
#Model Development and Hyperparameter Tuning
#training and tuning a KNN classifier 
KNN = KNeighborsClassifier(weights='distance', n_jobs=10)
Grid_KNN = GridSearchCV(KNN, scoring='f1', cv=4, n_jobs=10,
                            param_grid={
                                'n_neighbors': range(1,11),
                            }).fit(x_over, y_over)

#Report best parameters after grid search
print('Best parameter values for the KNN classifier:')
Grid_KNN.best_params_

#Model Evaluation 
#get best estimator based off grid search 
KNN_classifier = Grid_KNN.best_estimator_

#Generate predictions 
y_pred = KNN_classifier.predict(x_test)
#Report error scores
print('K-Nearest Neighbors classification results:')
error_scores(y_test, y_pred, classes=classes)
#Visualize confusion matrix
plot_cm(y_test, y_pred, classes)


#Model Three: Kernel SVM Classifier
#Given that it is a large dataset I will perform kernel approximation first before fitting the SVM classifier
#Kernel Approximation
#create an instance of Nystroem class and set characteristics
NystroemSVC = Nystroem(kernel='rbf', n_components=500, random_state=rs, n_jobs=10)
#Fit and transform the data
x_train_aprx = NystroemSVC.fit_transform(x_over)
x_test_aprx = NystroemSVC.transform(x_test)

#Model Development and Hyperparameter Tuning
#training and tuning a kernel SVM classifier 
linearSVC = LinearSVC(random_state=rs)
Grid_SVC = GridSearchCV(linearSVC, scoring='f1', cv=4, n_jobs=10, 
                            param_grid={
                                'penalty': ['l1', 'l2'], 
                                'C': [0.01, 0.1, 1, 10, 100]
                            }).fit(x_train_aprx, y_over)

#Report best parameters after grid search
print('Best parameter values for the kernel SVM classifier:')
Grid_SVC.best_params_

#Model Evaluation
#get best estimator based off grid search 
SVC_classifier = Grid_SVC.best_estimator_

#Generate predictions 
y_pred = SVC_classifier.predict(x_test_aprx)
#Report error scores
print('Kernel SVM classification results:')
error_scores(y_test, y_pred, classes=classes)
#Visualize confusion matrix
plot_cm(y_test, y_pred, classes)


#Model Four: Random Forest
#Model Development and Hyperparameter Tuning
#training and tuning a random forest model 
RF = RandomForestClassifier(random_state=rs, max_depth=10, max_features=5, n_jobs=10)
Grid_RF = GridSearchCV(RF, scoring='f1', cv=4, n_jobs=10,
                            param_grid={'n_estimators': [50, 100, 200, 300, 400]}).fit(x_over, y_over)

#Report best parameters after grid search
print('Best parameter values for the random forest classifier:')
Grid_RF.best_params_

#Model Evaluation
#get best estimator based off grid search
RF_classifier = Grid_RF.best_estimator_

#Generate predictions 
y_pred = RF_classifier.predict(x_test)
#Report error scores
print('Random Forest classification results:')
error_scores(y_test, y_pred, classes=classes)
#Visualize confusion matrix
plot_cm(y_test, y_pred, classes)


#Model Comparison 
#Comparing model performances so far using ROC curve plots  
#Get list of estimators 
estimators_lst = [LR_classifier, KNN_classifier, SVC_classifier, RF_classifier]
#Plot ROC curve for each model 
plot_ROC_curve_multiple(x_test, x_test_aprx, y_test, estimators_lst)


#Stacking Model
#Model Development
#first, making a list of classifiers to incorporate into the ensemble 
estimators = [('LR', LR_classifier), ('KNN', KNN_classifier), ('RF', RF_classifier)]

#Using Stacking Classifier with a logistic regression meta-model
#create Stacking classifier object and specify a meta classifier 
SC = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=500), n_jobs=10)
#training the stacking classifier 
SC_classifier = SC.fit(x_over, y_over)

#Model Evaluation
#Generate final class predictions 
y_pred = SC_classifier.predict(x_test)
#Report error scores
print('Stacking classification results:')
error_scores(y_test, y_pred, classes) 

#Visualizing confusion matrix 
plot_cm(y_test, y_pred, classes) 

#Visualize ROC curve 
plot_ROC_curve(SC_classifier, x_test, y_test)



#Part Five: Model Interpretation
#Permutation Feature Importance 
#Now I will perform permutation feature importance (with 20 repeats) in order to determine which features are most 
# important or impactful for predicting transaction fraud

#Calculate and store feature importances 
feature_importances = permutation_importance(estimator=SC_classifier, X=x_over, y=y_over, scoring='f1', 
                                             n_repeats=20, random_state=rs, n_jobs=10)

#get the shape of the resulting feature importances array 
print('Feature importances array shape:', feature_importances.importances.shape)

#Get the mean importances for each feature (mean of the n permutation repeats)
print('\n\nNumber of features:', len(feature_importances.importances_mean))
print('Mean feature importance for each feature:')
print(np.round(feature_importances.importances_mean,3))

#Visualize the feature importances using a Box plot 
visualize_feature_importances(feature_importances, features)


#Partial Dependence Plot
#To better understand how each of the important variables contributed to fraud prediction, I will use a partial dependency plot. This plot should 
# provide a visual representation of how the values of a given feature relates or contributes to the predicted target, fraud.

#List important features in pairs 
features_lst = [['type_TRANSFER', 'type_CASH_OUT'], ['oldbalanceOrg', 'newbalanceOrig'], 
                ['oldbalanceDest', 'newbalanceDest'], ['amount', 'step']]

#Create and display partial dependency plot for each of the listed features 
with plt.rc_context({'figure.figsize': (9.5, 5)}):
    pdp_plots = [PartialDependenceDisplay.from_estimator(estimator=SC_classifier, X=x_over, features=lst, categorical_features=features_lst[0], n_jobs=-1) for lst in features_lst]
    pdp_plots[0].figure_.suptitle('Partial Dependence Plot per Feature', fontsize=15)
    for ax in pdp_plots[0].axes_[0]:
        ax.set_xticks([0, 1])
