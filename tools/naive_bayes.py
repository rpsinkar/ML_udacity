import sys
from time import time
from email_preprocess import preprocess
import numpy as np
from sklearn.naive_bayes import GaussianNB

features_train,features_test,labels_train, labels_test = preprocess()


clf = GaussianNB()

t0 = time()


clf.fit(features_train,labels_train)

print "training time:", round(time()-t0,3), "s"

t1 = time()

pred = clf.predict(features_test)
print "prediction time:", round(time()-t1,3), "s"

print "Accuracy: ",clf.score(features_test,labels_test)
