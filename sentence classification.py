__author__ = 'Sadhana'

# coding: utf-8



import sys
sys.path.append('C:\Users\Sadhana\Anaconda\pywsd')
from pywsd.similarity import *
from pywsd.lesk import *
import math
import pandas as pd
import preprocess
import feature_extractor
import frames
import features

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



#getting frames of all input sentences
fram=frames.get_frames()
print fram

#getting top 2 frames for each process
#import features
feat=features.get_features()
print feat



def features_extract(df,df2):
    #creating new data frame to include features
    df6=pd.DataFrame(columns=['Process','Feature','Label'])
    df6['Process']=df['Process']
    df6['Label']=df['Label']
    df6['Sentence']=df['Sentence']
    #df6=df6[df6['Process']!='accumulation'].reset_index()
    #print df6


    #setting feature=1 if any of the top 2 frames are present in each sentence
    for x in range(df6.index.get_values()[-1]+1):
        if x not in df6.index.get_values():
            continue
        pro=df6.loc[x,'Process']

        if(pro not in feat):
            continue
        else:
            d=set(feat[pro]) & set(fram[x])
            #print x
            if len(d)!=0:
                df6.loc[x,'Feature']=1
            else:
                df6.loc[x,'Feature']=0
    df6=df6.dropna()
    #print df6,len(df6)








    #getting the sense of the process in the corresponding sentence
    for j in range(df6.index.get_values()[-1]+1):
        if j not in df6.index.get_values():
            continue
        df6.loc[j,'Sense']=max_similarity(df6.loc[j,'Sentence'],df6.loc[j,'Process'])
    #print df6


    #getting the definition of the sense
    for i in range(df6.index.get_values()[-1]+1):
        if i not in df6.index.get_values():
            continue
        if df6.loc[i,'Sense'] is None:
            #print df6['Sentence'][i]
            continue

        df6.loc[i,'Definitiion']=df6.loc[i,'Sense'].definition()
    ##print df6



    #df7=df6[['Process','Definitiion','Sentence','Sense']]




    #df8 = pd.DataFrame(columns=['Process','Definitiion','Sentence','Sense'])
    #setting feature =1 if sense is appropriate to the domain
    for i in range(df6.index.get_values()[-1]+1):
        #print str(df6.loc[i,'Sense'])
        if i not in df6.index.get_values():
            continue
        if df6.loc[i,'Sense'] is None:
            df6.loc[i,'SenseFeature']=1

        elif df6.loc[i,'Sense'].name() in ['preoccupation.n.02','adaptation.n.01','circulation.n.04','seethe.v.02','switch.v.03','combustion.n.02','cross-pollination.n.02','condensation.n.01','combustion.n.02','decomposition.n.01','deposit.v.02','deposit.n.06','deposition.n.04','diffusion.n.02','clash.n.02','germination.n.02','gravity.n.03','graveness.n.01','hail.v.02']:
            df6.loc[i,'SenseFeature']=0
            #df8.loc[i]=df6.loc[i]

        else:
            df6.loc[i,'SenseFeature']=1



    #setting feature=1 if the sentence ends with ...
    for i in range(df6.index.get_values()[-1]+1):

        if i not in df6.index.get_values():
            continue
        if str(df6.loc[i,'Sentence']).endswith("...")==True:
            #print df6.loc[i,'Sentence']
            df6.loc[i,'Incomp']=0

        else:
            df6.loc[i,'Incomp']=1
    #print sum(df6['Incomp']),len(df6)

    #getting the score of AI2 classifier

    df2.columns=['Query','Process','Sentence','Overlap','Score','Label','CScore']
    df2=df2[['Process','Sentence','CScore']]
    df2=df2.sort(columns='Process').reset_index()

    result = pd.merge(df6, df2, on='Sentence')
    #print result,len(result),sum(result['Label'])



    #getting the score of AI2 classifier
    count=0
    co=0
    for i in range(result.index.get_values()[-1]+1):
        if result.loc[i,'Incomp']==0:
            co=co+1
            #print result.loc[i,'CScore']
            if result.loc[i,'CScore']>0.8:
                #print result.loc[i,'Sentence']
                count=count+1
    #print len(result['Incomp']),count,co,co-count,(co-count)/float(co)
    return result
# read data file
df = pd.read_csv("senwlabels.txt", sep="\t",header=0)
df2 = pd.read_csv("senaioutput.txt", sep="\t",header=None)

train=df[0:500]# modify train to train=df and test= the dataframe containing the test data for classifying with real time data
test=df[501:len(df)]

result=features_extract(train,df2)

t=features_extract(test,df2)

#creating feature vector
import numpy as np
f=result['Feature']
s=result['SenseFeature']
l=result['Label']
c=result['CScore']
ft=t['Feature']
st=t['SenseFeature']
lt=t['Label']
ct=t['CScore']
inc=result['Incomp']
#f=f.reshape(len(f),1)
#s=s.reshape(len(f),1)
#c=c.reshape(len(c),1)
#l=l.reshape(len(l),1)
l=np.ravel(l)
X = np.array([f, s,c]).T
Xt = np.array([ft, st,ct]).T
lt=np.ravel(lt)


from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
#Xtrn, Xtst, Ytrn, Ytst = train_test_split(X,l , test_size=0.3, random_state=0)
#std_scale = preprocessing.StandardScaler().fit(Xtrn)
#X_train_std = std_scale.transform(Xtrn)
#X_test_std = std_scale.transform(Xtst)
#print X_train_std
#print Xtrn


#classification
from sklearn.cross_validation import cross_val_score
clfs = [KNeighborsClassifier(), GaussianNB(), LogisticRegression(), RandomForestClassifier(n_estimators=100, n_jobs=2), AdaBoostClassifier(n_estimators=100)]
clf_names = ['Nearest Neighbors', 'Gaussian Naive Bayes', 'Logistic Regression', 'RandomForestClassifier', 'AdaBoostClassifier']

results = {}
for (i, clf_) in enumerate(clfs):

    clf = clf_.fit(X,l)
    preds = clf.predict(Xt)
    f=open('classifiedsentences'+clf_names[i]+'.txt', 'w+')
    for j in range(len(preds)):
        if preds[j]==1:
            f.write(t.loc[j,'Sentence']+'\n')



    precision = metrics.precision_score(lt, preds)
    recall = metrics.recall_score(lt, preds)
    f1 = metrics.f1_score(lt, preds)
    accuracy = accuracy_score(lt, preds)
    #scores = cross_val_score(clf, X, l, scoring='accuracy', cv=10)
    #scoremean=scores.mean()
    # report = classification_report(Ytst, preds)
    # matrix = metrics.confusion_matrix(Ytst, preds, labels=list(set(labels)))

    data = {'precision':precision,
            'recall':recall,
            'f1_score':f1,
            'accuracy':accuracy,
            # 'clf_report':report,
            # 'clf_matrix':matrix,
            'y_predicted':preds}
           #'Cross Validation Score':scoremean}

    results[clf_names[i]] = data

cols = ['precision', 'recall', 'f1_score', 'accuracy']#'Cross Validation Score']
print pd.DataFrame(results).T[cols].T








