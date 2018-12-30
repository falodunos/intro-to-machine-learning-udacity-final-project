#!/usr/bin/python
'''
all_features = ['salary','to_messages','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value',
'expenses','loan_advances','from_messages','other','from_this_person_to_poi','poi','director_fees','deferred_income','long_term_incentive','email_address','from_poi_to_this_person']

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

exercised_stock_options :  is the amount of stock options exercised (i.e bought or sold) within a vesting period
restricted_stock : a nontransferable stock that is subject to forfeiture under certain conditions, such as termination of employment or failure to meet either corporate or personal performance benchmarks.
restricted_stock_deferred : A deferred stock is a stock that does not have any rights to the assets of a company undergoing bankruptcy until all common and preferred shareholders are paid.
total_stock_value : total or cummulative amount of all stock values
deferred_income : Deferred income (also known as deferred revenue, unearned revenue, or unearned income) is, in accrual accounting, money received for goods or services which have not yet been delivered. 
	According to the revenue recognition principle, it is recorded as a liability ( because it represents products or services that are owed to a customer) until delivery is made, at which time it is converted into revenue.
deferral_payments : A loan arrangement in which the borrower is allowed to start making payments at some specified time in the future.
'''

import sys
import pickle

sys.path.append("../tools/")

import numpy as np
from tester import dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import average_precision_score as precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


# ============= TASK 1 : DATASET EXPLORATION ... ================
def datasetExploration(data_dict):
    print "------ Exploration of Enron E+F Dataset --------"
    keys = data_dict.keys()
    print ('Number of data points: %d' %len(keys))
    print ('Number of Features Per Person: %d' %len(data_dict[keys[0]].keys()))
    num_poi = 0
    num_real_salary = 0
    for key, val in data_dict.items():
        if (val['poi'] == 1.0):
            num_poi = num_poi + 1
        if (val['salary'] != 'NaN'):
            num_real_salary = num_real_salary + 1

    print ('Number of POIs In The Dataset: %d' % num_poi)
    print ('Number of Non-POIs In The Dataset: %d' % (len(keys) - num_poi))
    print ('Number of People With Quantified Salary: %d' % num_real_salary)
    print


def featuresMissingPercentage(data_dict):
    entries = data_dict.values()
    total = len(entries)
    missingCounts = {}
    for key in entries[0].keys():
        missingCounts[key] = 0

    for value in entries:
        for k, val in value.items():
            if (val == 'NaN'):
                missingCounts[k] = missingCounts[k] + 1

    missingPercentage = {}
    missingValues = {}

    for k, v in missingCounts.items():
        percent = v / float(total)
        missingValues[k] = v
        missingPercentage[k] = percent

    print
    return missingPercentage, missingValues


def displayMissingFeatures(missingPercentage, missingValues):
    for k, v in missingPercentage.items():
        print missingValues[k], 'of \'' + k + '\' is missing; missing percentage is : %.2f' % v
    print


def removeFeaturesWithMissingValuesAboveXpercentage(allFeatures, missingPercentage, threshold):
    # select only the features where the mising percentage is less or equal to specified threshold
    newList = []
    for value in allFeatures:
        if (value != 'email_address' and float(missingPercentage[value]) <= threshold):
            newList.append(value)
    print
    return newList


# This function removes data point for which more x% e.g. 85% of the features are missing ...
def suspectDataPointForWhichMoreThanXpercentFeaturesIsNaN(data_dict, threshold):
    keys = data_dict.keys()
    num_features = len(data_dict[keys[0]].keys())
    data_dict_new = data_dict

    dataKeys = []
    for name, item in data_dict.items():
        num_nan = 0
        for key, value in item.items():
            if (value == 'NaN'):
                num_nan = num_nan + 1

        nanPercent = num_nan / float(num_features)
        if (nanPercent >= threshold):
            dataKeys.append(name)
            del data_dict_new[name]
    return dataKeys, data_dict_new


# ========= TASK 2 : FUNCTIONS FOR REMOVING OUTLIERS ===========

'''
	This plot gives visual view of the existence of outliers
'''


def PlotOutlier(data_dict, feature_x, feature_y):
    """ Plot with flag = True in Red """
    data = featureFormat(data_dict, [feature_x, feature_y, 'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            color = 'red'
        else:
            color = 'blue'
        plt.scatter(x, y, color=color)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()


def getFormattedFeaturesAndNameList(data_dict, features):
    featureList = {}
    nameList = {}
    for feature in features:
        featureList[feature] = []
        nameList[feature] = []

    for feature in features:
        for name, value in data_dict.items():
            for key, val in value.items():
                if (key in features and val != 'NaN'):
                    featureList[key].append(val)
                    nameList[key].append(name)

    return featureList, nameList


def findSuspectedOutliers(featureList):
    suspectedOutliers = {}
    for key in featureList.keys():
        suspectedOutliers[key] = set()

    for key, value in featureList.items():
        q1 = np.percentile(value, 25)
        q3 = np.percentile(value, 75)
        iqr = q3 - q1
        floor = q1 - 10 * iqr
        ceiling = q3 + 10 * iqr

        for x in value:
            if ((x < floor) | (x > ceiling)):
                suspectedOutliers[key].add(x)
    return suspectedOutliers


def findOutliers(data_dict, outlyingValues, threshold):
    outliers = {}
    outlyingKeys = outlyingValues.keys()

    for key in data_dict.keys():
        outliers[key] = 0

    for key, value in data_dict.items():
        for k, val in value.items():
            if ((k in outlyingKeys) and (val in outlyingValues[k])):
                outliers[key] = outliers[key] + 1
    filteredOutliers = {}
    total = 0
    for k, v in outliers.items():
        if (v > 0):
            filteredOutliers[k] = v
            total += v

    realOutliers = []
    for key, value in filteredOutliers.items():
        if (value / float(total) >= threshold):
            realOutliers.append(key)

    return realOutliers


# ========= TASK 3: CREATE NEW FEATURES(S) ===========
# from_poi_to_this_person_ratio and from_this_person_to_poi_ratio
def computeRatio(messages, allMessages):
    ratio = 0.
    if (messages == 'NaN' or allMessages == 'NaN'):
        return ratio
    ratio = messages / float(allMessages)
    return ratio


def createNewFeatures(my_dataset):
    for poi_name in my_dataset:
        data_point = my_dataset[poi_name]
        data_point['from_poi_to_this_person_ratio'] = computeRatio(data_point['from_poi_to_this_person'],
                                                                   data_point['to_messages'])
        data_point['from_this_person_to_poi_ratio'] = computeRatio(data_point['from_this_person_to_poi'],
                                                                   data_point['from_messages'])
    return my_dataset, ['from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio']


# ------------------------- FEATURE SELECTION METHOD -----------------------------------
# function using SelectKBest
def findKbestFeatures(data_dict, features_list, k):
    from sklearn.feature_selection import f_classif
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(f_classif, k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    # print "sorted_pairs", sorted_pairs
    k_best_features = dict(sorted_pairs[:k])

    return k_best_features


features_list = ['poi']  # You will need to use more features
email_features = ['email_address', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']
email_features.remove('email_address')  # remove email_address feature ...
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)  # returns a python dictionay ...

# TASK 1: SELECT WHAT FEATURES YOU'LL USE
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

# 1.1 : Combines both email and financial features together ...
all_features = email_features + financial_features

# 1.2 : This functions explore the dataset for some important quantifiable data
datasetExploration(data_dict)

# 1.3 This function analyses the data set for features with missing values
missingPercentage, missingValues = featuresMissingPercentage(data_dict)

# 1.4 Displays features with missing values and the percentage of missing values
displayMissingFeatures(missingPercentage, missingValues)

# 1.5 : Email and features for which more x percent e.g. 70% are missing were considered irrelevant and may be not used
# features_list = features_list + removeFeaturesWithMissingValuesAboveXpercentage(all_features, missingPercentage, 0.70)


# TASK 2: REMOVE OUTLIERS
'''
2.1 : Function to plot and give a visual view of the outliers
'''
PlotOutlier(data_dict, 'total_payments', 'total_stock_value')
# PlotOutlier(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi')
# PlotOutlier(data_dict, 'total_payments', 'total_stock_value')


'''
2.2 : Now detecting the outliers ...
'''
featureList, nameList = getFormattedFeaturesAndNameList(
    data_dict, all_features)
outlyingValues = findSuspectedOutliers(featureList)

# detected outliers are  ['TOTAL']
detectedOutliers = findOutliers(data_dict, outlyingValues, 0.50)


# other outliers detected by manual inspection of the key returned by
# are 'LOCKHART EUGENE E'
detectedOutliers.append('LOCKHART EUGENE E')

'''
2.2 : Now removing the outliers that were detected ...
'''
for outlier in detectedOutliers:
    data_dict.pop(outlier, 0)
'''
	Data points for which more than 85%  of the features are missing were suspected as outliers ...
	Manually inspect returned dataKeys i.e. ['WHALEY DAVID A', 'WROBEL BRUCE', 'LOCKHART EUGENE E', 'THE TRAVEL AGENCY IN THE PARK', 'GRAMM WENDY L']
'''
dataKeys, data_dict_modified = suspectDataPointForWhichMoreThanXpercentFeaturesIsNaN(
    data_dict, 0.85)


# TASK 3: CREATE NEW FEATURES(S)
# Store to my_dataset for easy export below.
my_dataset = data_dict
'''
	3.1 : Now creating new features ...
'''
my_dataset, new_features = createNewFeatures(my_dataset)
all_features = all_features + new_features

'''
	3.2 : Now Selecting Important Features Using SelectKBest Algorithm ...
'''
num_features = 10
selectedBestFeatures = findKbestFeatures(my_dataset, all_features, num_features)
selectedFeatures = ['poi'] + selectedBestFeatures.keys()

# ****** THESE FEATURES WERE MANUALLY BUT RATIONALLY SELECTED FOR REMOVAL ********
# selectedFeatures.remove('other')
# selectedFeatures.remove('from_this_person_to_poi')
# selectedFeatures.remove('from_poi_to_this_person')
# selectedFeatures.remove('from_messages')
# selectedFeatures.remove('loan_advances')
# selectedFeatures.remove('total_payments')
# **************************** UN-COMMENT TO REMOVE THEM ***************************

print "Selected Features ", selectedFeatures

'''
	3.3 : Now Extracting Required Features and Labels From Dataset Corpus
'''
data = featureFormat(my_dataset, selectedFeatures, sort_keys=True)
labels, features = targetFeatureSplit(data)
'''
3.4 : Now Performing Feature Scaling On Selected Features ...
'''
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Split selected features into training and test data ...
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

'''
	Define a dictionary of classifiers ...
'''
classifiers = {}


# Provided to give you a starting point. Try a variety of classifiers.

# NAIVE BAYES CLASSIFIER
def naive_bayes_classifier(features_train, features_test, labels_train, labels_test):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(features_train, labels_train)  # accept numpy array
    pred = classifier.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)

    # Compute the ratio of true positives relative to all positives (true + false)
    precision = precision_score(labels_test, pred)
    # Compute the ratio of true positives relative to true positives and false negatives
    recall = recall_score(labels_test, pred)

    print "Naive Bayes Accuracy :", accuracy
    print 'Precision :', precision
    print 'Recall :', recall
    return classifier


# SVM CLASSIFIER
def svm_classifier(features_train, features_test, labels_train, labels_test):
    from sklearn.svm import SVC
    classifier = SVC(kernel="rbf", C=1000)
    classifier.fit(features_train, labels_train)
    pred = classifier.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)

    # Compute the ratio of true positives relative to all positives (true + false)
    precision = precision_score(labels_test, pred)
    # Compute the ratio of true positives relative to true positives and false negatives
    recall = recall_score(labels_test, pred)

    print "SVM Accuracy :", accuracy
    print 'Precision :', precision
    print 'Recall :', recall
    return classifier


#  SVM WITH GRID SEARCH
def svm_grid_search(features_train, features_test, labels_train, labels_test):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    classifier = GridSearchCV(SVC(), parameters)
    classifier = classifier.fit(features_train, labels_train)
    pred = classifier.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)

    # Compute the ratio of true positives relative to all positives (true + false)
    precision = precision_score(labels_test, pred)
    # Compute the ratio of true positives relative to true positives and false negatives
    recall = recall_score(labels_test, pred)

    print "GridSearch Accuracy :", accuracy
    print 'Precision :', precision
    print 'Recall :', recall
    return classifier

# DECISION TREE


def decision_tree_classifier(features_train, features_test, labels_train, labels_test):
    from sklearn import tree
    import graphviz
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(features_train, labels_train)

    dot_data = tree.export_graphviz(classifier, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("enron")

    pred = classifier.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)

    # Compute the ratio of true positives relative to all positives (true + false)
    precision = precision_score(labels_test, pred)
    # Compute the ratio of true positives relative to true positives and false negatives
    recall = recall_score(labels_test, pred)

    print "Decision Tree Accuracy :", accuracy
    print 'Precision :', precision
    print 'Recall :', recall

    dot_data = tree.export_graphviz(classifier, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("poi")
    return classifier


# Adaboost Classifier
def adaboost_classfier(features_train, features_test, labels_train, labels_test):
    from sklearn.ensemble import AdaBoostClassifier
    classifier = AdaBoostClassifier(n_estimators=1000, random_state=202, learning_rate=1.0, algorithm="SAMME.R")
    classifier.fit(features_train, labels_train)
    pred = classifier.predict(features_test)

    accuracy = accuracy_score(labels_test, pred)

    # Compute the ratio of true positives relative to all positives (true + false)
    precision = precision_score(labels_test, pred)
    # Compute the ratio of true positives relative to true positives and false negatives
    recall = recall_score(labels_test, pred)

    print "Adaboost Accuracy :", accuracy
    print 'Precision :', precision
    print 'Recall :', recall
    return classifier


# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

# clf = naive_bayes_classifier(features_train, features_test, labels_train, labels_test)
#clf = svm_classifier(features_train, features_test, labels_train, labels_test)
# clf = svm_grid_search(features_train, features_test, labels_train, labels_test)
clf = decision_tree_classifier(features_train, features_test, labels_train, labels_test)
# clf = adaboost_classfier(features_train, features_test, labels_train, labels_test)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
