# importing libraries and modules
from math import sqrt
from scipy.spatial.distance import pdist
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error,mean_absolute_error,mean_squared_error
from sklearn.metrics import pairwise_distances
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(color_codes=True)
import warnings
from pandas.plotting import scatter_matrix
import seaborn as sns; sns.set(color_codes=True)
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

#Constants
splitConst =5


# function that plots Pair Grid with data input
def disp_pairGrid(corr_df):
    corr_plot = sns.PairGrid(corr_df)
    corr_plot = corr_plot.map_upper(disp_pearson, color = "lightblue")
    corr_plot = corr_plot.map_diag(plt.hist)
    corr_plot = corr_plot.map_lower(sns.regplot , line_kws = {"color" : "red"}, color = "black")

#Pair Grid Plot corr calcultions function
def disp_pearson(x, y, **kws):
    corrmat = np.corrcoef(x,y)
    pearson = round(corrmat[0,1],2)
    ax = plt.gca()
    ax.annotate(pearson,[0.5,0.5], xycoords = "axes fraction", ha = "center", va = "center", fontsize =50 )

#a function needs to be defined for calculation of MAPE as this couldnt be found in any of the libraries
def mean_absolute_percentage_error(y_true,y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true -y_pred)/y_true))*100

#this function evaluates train and test scores for the given model
def evaluate_scores_Numeric(y_train, y_test,y_pred_train, y_pred_test):
    #the average magnitude of the error
    RMSE_test  = sqrt(mean_squared_error(y_test,y_pred_test))
    RMSE_train = sqrt(mean_squared_error(y_train,y_pred_train))

    #shows how good the model is as acompared to where the actual points are
    R2_test   = r2_score(y_test,y_pred_test)
    R2_train  = r2_score(y_train,y_pred_train)

    #average of abolute errors (amount of error on average)
    MAE_test  = mean_absolute_error(y_test,y_pred_test)
    MAE_train = mean_absolute_error(y_train,y_pred_train)

    # measure of prediction accuracy
    MAPE_test  = mean_absolute_percentage_error(y_test,y_pred_test)
    MAPE_train = mean_absolute_percentage_error(y_train,y_pred_train)
    
    #here we are priniting both training and testing data to look for overfitting
    print("Training data")
    print("RMSE: ", round(RMSE_train,2), "| R2: ", round(R2_train,2), "| MAE", round(MAE_train,2), "| MAPE", round(MAPE_train,2))
    print()
    print("Testing data")
    print("RMSE: ", round(RMSE_test,2), "| R2: ", round(R2_test,2), "| MAE", round(MAE_test,2), "| MAPE", round(MAPE_test,2))
    print()
    print("% difference between train and test")
    print("RMSE: ", 100 * round(1- RMSE_train / RMSE_test, 2), "| R2: ", round(100 * (1- R2_test / R2_train),2), "| MAE",  100 * round(1- MAE_train / MAE_test, 2), "| MAPE", 100 * round(1- MAPE_train / MAPE_test, 2))
    
    
#this function is a template for creation and evaluating any model. it packages all repetative steps into one function.
def model_build(X_train, y_train, X_test, y_test, model, resid):
    model = model.fit(X_train,y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    evaluate_scores_Numeric(y_train, y_test,y_pred_train, y_pred_test)
    if (resid == "lr"):
        residual_plot(y_test, y_pred_test)
    
    
#this function aims to graph residual plot for numerical values (regression modeling)
def residual_plot(y_test, y_pred_test):
    residual = y_test.values - y_pred_test
    plt.figure(figsize=(16,9))
    plt.scatter(y_test, residual, color = "none", edgecolor = "black")
    plt.title("Model Residual Plot", fontsize =18, fontweight = "bold")
    plt.xlabel("Average Total Income(standardised)")
    plt.ylabel("Residuals")
    plt.axhline(0, color = "black")


#this function is the same as the model_build() function, yet it is aims to recursively eliminate all featues and find optimal number of features to be considered for linear regression using cross validation
def RFECV_LR(X_train,y_train, X_test, y_test):
    model = LinearRegression()
    rfecv = RFECV(estimator = model, step = 1, cv = splitConst, scoring = "neg_mean_squared_error")
    rfecv = rfecv.fit(X_train, y_train)
    
    #plotting findings
    plt.figure(figsize=(16,9))
    print("")
    plt.xlabel("No. of best features selected")
    plt.ylabel("Cross Validation Score : RMSE")
    plt.title("RMSE scores on train set using RFECV", fontsize =18, fontweight = "bold")
    
    rmse_cv_scores = np.sqrt(-rfecv.grid_scores_)
    plt.plot(range(1,len(rfecv.grid_scores_) +1), rmse_cv_scores, marker = "o", color = "black")
    
    #printing the hyperparameter optimisation realted inoformation to drive decissionmaking
    print("")
    print("Total number of features : ", len(X_train.columns))
    print("Optimal Number of Features : ", rfecv.n_features_)
    print("Best features ranked : ", X_train.columns[rfecv.support_])
    
    
    print("") #creating spacial separation for better reading
    print("Result Evaluation")
    y_pred_test = rfecv.predict(X_test)
    y_pred_train = rfecv.predict(X_train)
    evaluate_scores_Numeric(y_train, y_test,y_pred_train, y_pred_test)
    
    #residual plot to deterine when the model performed well and when it didnt
    residual_plot(y_test, y_pred_test)
    
  
    
    
def tree_tuning(X,y):
    # For each criterion, we use 10-fold cross validation to report the RMSE for each depth of tree model
    cv_scores = []
    cv_scores_std = []
    alphas = np.arange(1, 30)
    for i in range(1,30):
        regress = DecisionTreeRegressor(random_state=42,  max_depth= i)
        scores = cross_val_score(regress, X, y, scoring= "neg_mean_squared_error", cv= splitConst)
        # here we convert the neg_mean_squared_error into root of positive mean squared errors or in other words RMSE
        RMSE = np.sqrt(-scores.mean())
        # this list is required to find stadard deviations of RMSE items from cross validation scores for each depth level
        RMSE_list = np.sqrt(-scores)
        cv_scores.append(RMSE)
        cv_scores_std.append(RMSE_list.std())
   
   #plotting findings into a graph to allow for analysis
    plt.figure(figsize=(16,9))
    plt.errorbar(alphas,  cv_scores, yerr=cv_scores_std, marker='x', label='Accuracy')
    plt.xlabel('max_depth')
    plt.ylabel('RMSE on train set using CV')
    plt.legend(loc='best')
    plt.show()
    
    
    
    # this function was create out of need to exmaine how the total data changes when we  hyperparameters to veryfy the integrity of tree_tuning() function
def tree_tuning_noCV(X_train,y_train, X_test, y_test):
    # For each criterion, we use 10-fold cross validation to report the RMSE for each depth of tree model
    cv_scores = []
    cv_scores_std = []
    alphas = np.arange(1, 30)
    for i in range(1,30):
        regress = DecisionTreeRegressor(random_state=42,  max_depth= i)
        model = regress.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        RMSE = sqrt(mean_squared_error(y_test,y_pred))
        cv_scores.append(RMSE)
   
   #plotting findings into a graph to allow for analysis
    plt.figure(figsize=(16,9))
    plt.errorbar(alphas,  cv_scores, marker='x', label='Accuracy')
    plt.xlabel('max_depth')
    plt.ylabel('RMSE on train set using CV')
    plt.legend(loc='best')
    plt.show()
    
def class_model_build(X_train, y_train, X_test, y_test, modelName):
    model = modelName
    model = model.fit(X_train, y_train)

    #accuracy scores for training and testing to look for overfitting and performance of the model
    y_pred_test = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print("Testing accuracy:  ", round(accuracy_test,3))
    
    y_pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print("Training accuracy:  ", round(accuracy_train,3))
    
    #plotting confusion matrix for aesthetic appeal
    labels = modelName.classes_
    cm = confusion_matrix(y_pred_test, y_test, labels)
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot()
    confusion_matrix_plot(ax, cm, labels,'Confusion matrix for test data', 15)

#the function below creates a visually appealing confusion_matrix diagram and it was heavily inspired by https://gist.github.com/jcboyd/2d4427b2b5ffa464da2d599d217d0dd9
def confusion_matrix_plot(ax, matrix, labels, title, fontsize):
    ax.set_xticks([x for x in range(len(labels))])
    ax.set_yticks([y for y in range(len(labels))])
    # Place labels on minor ticks
    ax.set_xticks([x + 0.5 for x in range(len(labels))], minor=True)
    ax.set_xticklabels(labels,fontsize=fontsize, minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(labels))], minor=True)
    ax.set_yticklabels(labels[::-1], fontsize=fontsize, minor=True)
    # Hide major tick labels
    ax.tick_params(which='major', labelbottom='off', labelleft='off')
    # Finally, hide minor tick marks
    ax.tick_params(which='minor', width=0)
    # Plot heat map
    proportions = [1. * row / sum(row) for row in matrix]
    ax.pcolor(np.array(proportions[::-1]), cmap=plt.cm.Reds)
    # Plot counts as text
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            confusion = matrix[::-1][row][col]
            if confusion != 0:
                ax.text(col + 0.5, row + 0.5, int(confusion),fontsize=fontsize,horizontalalignment='center',verticalalignment='center')

    # Add finishing touches
    ax.grid(True, linestyle=':')
    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel('Prediction', fontsize=fontsize)
    ax.set_xlabel('Actual', fontsize=fontsize)
    plt.show()
    
def RFECV_cat(X_train,y_train, X_test, y_test, modelName):
    model = modelName
    rfecv = RFECV(estimator = model, step = 1, cv = StratifiedKFold(n_splits= splitConst, random_state=42), scoring = "accuracy")
    rfecv = rfecv.fit(X_train, y_train)
    
    #accuracy scores for training and testing to look for overfitting and performance of the model
    y_pred_test = rfecv.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print("Testing accuracy:  ", round(accuracy_test,3))
    
    y_pred_train = rfecv.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print("Training accuracy:  ", round(accuracy_train,3))
    
    #plotting confusion matrix for aesthetic appeal
    labels = rfecv.classes_
    cm = confusion_matrix(y_pred_test, y_test, labels)
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot()
    confusion_matrix_plot(ax, cm, labels,'Confusion matrix on testing data', 15)
    
    #tuning related details
    print("")
    print("Parameter Tuning info")
    print("Total number of features : ", len(X_train.columns))
    print("Optimal Number of Features : ", rfecv.n_features_)
    print()
    print("Best features ranked : ", X_train.columns[rfecv.support_])

    
    
def parameterTuning_CV_KNN(X_train, y_train,inputRange):
    # For each criterion, we use 10-fold cross validation to report the RMSE for each depth of tree model
    cv_scores = []
    cv_scores_std = []
    for neigbour in inputRange:
        model = KNeighborsClassifier(n_neighbors = neigbour)
        scores = cross_val_score(model, X_train, y_train, scoring= "accuracy", cv= StratifiedKFold(n_splits= splitConst, random_state=42))
        cv_scores.append(scores.mean())
        cv_scores_std.append(scores.std())
    
       #plotting findings into a graph to allow for analysis
    plt.figure(figsize=(16,9))
    plt.errorbar(inputRange,  cv_scores, yerr=cv_scores_std, marker='x', label='Accuracy')
    plt.xlabel('k-neighbours')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()
    

def compare_distributions(y_train, y_test):
    #creating subplot objects in order to set up a layout that will allow for comparision of 2 graphs
    figure, axes = plt.subplots(nrows = 2, ncols = 1, sharex=True,figsize=(12, 5))

    
    #creating the first box plot and giving it various titles for ease of comprehansion
    axes[0].set_title("train_set income distribution(standardised)",fontsize =12, fontweight = "bold")
    plot1 = sns.boxplot(x = y_train, orient = "h", color = "lightblue", linewidth =0.5, ax=axes[0])

    #creating the second box plot and giving it various titles for ease of comprehansion
    axes[1].set_title("test_set income distribution(standardised)",fontsize =12, fontweight = "bold")
    plot2 = sns.boxplot(x = y_test, orient = "h", color = "lightblue", linewidth =0.5, ax=axes[1])
    
    #improving the display format of 2 graphs to allow for better comparision by getting them closer together
    figure.tight_layout()
    plt.xlabel('average income/loss (standardised)')
    plt.ylabel('Freuqency')
