#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import pprint
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
finance_features_list = ['poi','salary', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock'] # You will need to use more features
email_features_list = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']


my_features = ['poi', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive']
### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


data_dict.pop("TOTAL", 0)
### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features, sort_keys = True)
labels, features = targetFeatureSplit(data)




data_points = len(data_dict)
poi_count = 0
non_poi_count = 0
print "Data points:\t", data_points
print "no of POIs:\t\t", poi_count
print "no of non POIs:\t", non_poi_count
print "POI ratio:\t\t", poi_count/data_points
print "Total features:\t", len(data_dict[data_dict.keys()[0]])
print "my_features:\t", len(my_features)
#print "Email features:\t", len(email_features)
print ""
'''
#code
def outlier_visualization(data, a, b, a_name, b_name, pos):
    plt.subplot(4,4,pos)
    f1 = []
    f2 = []
    y = []
    for point in data:
        f1.append(point[a])
        f2.append(point[b])
        #c = 'red' if point[0]==True else 'blue'
        #y.append(c)
    plt.scatter(f1, f2)

    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    plt.xlabel(a_name)
    plt.ylabel(b_name)
    plt.show()


def visualize_outliers():
    start = 1
    for i in range(2,len(my_features)):
        outlier_visualization(data, 1, i, 'salary', my_features[i], start)
        start += 1
    start = 10

visualize_outliers()
#end my code

'''

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

#clf = GaussianNB()
#clf = svm.SVC()

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_features)