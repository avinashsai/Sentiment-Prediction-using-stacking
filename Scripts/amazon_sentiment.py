import os
import re
import pandas as pd 
import sklearn
import numpy as np 
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import SVC
from nltk.corpus import stopwords

class stacking():
	def __init__(self,filename):

		self.filename = filename
		self.stopword = nltk.corpus.stopwords.words('english')
		self.labels = np.zeros(1000)

		self.kf = sklearn.model_selection.StratifiedKFold(n_splits=5)

	
	def fileopen(self):

		with open(self.filename,'r',encoding='latin1') as f:
			data = f.readlines()

		return data

	def cleansent(self,sentence):

		sentence = re.sub(r'[^\w\s]'," ",str(sentence))
		sentence = re.sub(r'[^a-zA-Z]'," ",str(sentence))
		sents = word_tokenize(sentence)
		new_sents = " "
		for i in range(len(sents)):
			if(sents[i].lower() not in self.stopword):
				new_sents+=sents[i].lower()+" "

		return new_sents

	def preprocess(self,data):

		corpus = []
		for i in range(1000):
			corpus.append(self.cleansent(data[i]))
			self.labels[i]=(int)(data[i][-2])

		return corpus

	def splitdata(self,corpus):

		xtrain,xtest,ytrain,ytest = train_test_split(corpus,self.labels,test_size=0.3,random_state=42)

		return xtrain,xtest,ytrain,ytest

	def features(self,xtrain,xtest,ytrain):


		vec = TfidfVectorizer(min_df=1,max_df=0.8,use_idf=True,sublinear_tf=True,stop_words='english')
		train_tf = vec.fit_transform(xtrain,ytrain)
		test_tf = vec.transform(xtest)

		chi = SelectKBest(chi2,k=300)
		train_f = chi.fit_transform(train_tf,ytrain)
		test_f = chi.transform(test_tf)

		return train_f,test_f

	def makepred(self,xtrain,xtest,ytrain):
		clf1 = MultinomialNB()
		clf1.fit(xtrain,ytrain)
		clf1p = clf1.predict(xtest)

		clf2 = SVC(kernel='linear',gamma=0.2)
		clf2.fit(xtrain,ytrain)
		clf2p = clf2.predict(xtest)

		return clf1p,clf2p

	def add_vals(self,train_meta,m1_pred,m2_pred,st,en):

		j = 0

		for i in range(st,en):
			train_meta[i][0] = m1_pred[j]
			train_meta[i][1] = m2_pred[j]
			j = j+1

		return train_meta

	def kfoldtrain(self,xtrain,xtest,ytrain):

		st=0
		en=0
		train_l = len(xtrain)
		train_meta = np.zeros((train_l,2))

		for train_index,test_index in self.kf.split(xtrain,ytrain):

			corpus_train = [xtrain[i] for i in train_index]
			corpus_test = [xtrain[i] for i in test_index]

			label_train,label_test = ytrain[train_index],ytrain[test_index]

			train_f,test_f = self.features(corpus_train,corpus_test,label_train)

			m1_pred,m2_pred = self.makepred(train_f,test_f,label_train)

			st = en
			en = st+len(test_index)

			train_meta = self.add_vals(train_meta,m1_pred,m2_pred,st,en)

		return train_meta

	def train(self,xtrain,xtest,ytrain):
		
		test_l = len(xtest)
		test_meta = np.zeros((test_l,2))
		
		train_f,test_f = self.features(xtrain,xtest,ytrain)

		m1_pred,m2_pred = self.makepred(train_f,test_f,ytrain)

		test_meta = self.add_vals(test_meta,m1_pred,m2_pred,0,test_l)

		return test_meta

	def actualtrain(self,train_meta,test_meta,ytrain,ytest):

		clf = LogisticRegression()
		clf.fit(train_meta,ytrain)
		clfp = clf.predict(test_meta)

		clfa = accuracy_score(clfp,ytest)
		clff1 = f1_score(clfp,ytest)
		clfcm = confusion_matrix(clfp,ytest)

		return clfa,clff1,clfcm



	
if __name__ == '__main__':

	dataset = '../dataset/amazon_cells_labelled.txt'
	
	stack = stacking(dataset)

	data = stack.fileopen()

	corpus = stack.preprocess(data)

	xtrain,xtest,ytrain,ytest = stack.splitdata(corpus)

	train_f,test_f = stack.features(xtrain,xtest,ytrain)

	train_meta = stack.kfoldtrain(xtrain,xtest,ytrain)

	test_meta = stack.train(xtrain,xtest,ytrain)

	acc_s,f1_s,cm_s  = stack.actualtrain(train_meta,test_meta,ytrain,ytest)

	print("Accuracy of stacked model :",acc_s)
	print("F1 Score of stacked model :",f1_s)
	print("confusion matrix:")
	print(cm_s)


	print("--------------------------------------------------------------")

	acc_o,f1_o,cm_o = stack.actualtrain(train_f,test_f,ytrain,ytest)

	print("Accuracy of original model :",acc_o)
	print("F1 Score of original model :",f1_o)
	print("confusion matrix:")
	print(cm_o)



