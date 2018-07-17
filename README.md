# Sentiment-Prediction-using-stacking

# Getting Started
Stacking is an Ensemble technique popularly used in Kaggle competitions. Inspired by this http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/ tutorial on Stacking, I implemented stacking from scratch and used it to predict sentiments of movie reviews.

There is a significant improvement in accuracy and F1 Score when stacking is used.

# Approach

Dataset is taken from Sentiment Labelled Sentences from UCI Repository.

First layer of Classifiers used: Support Vector Machines and Multinomial Naive Bayes

Second layer of Classifiers used: Logistic Regression

# Future Work

I am planning to extend to Neural Networks as well.

# Installation
**Scikit-learn**
**Numpy**
**NLTK**

# Running the code
To run this code,

Clone this repository to your system using

```
git clone https://github.com/avinashsai/Sentiment-Prediction-using-stacking.git

```
change the folder using

```
cd Scripts

```
Run 

```

python3 amazon_sentiment.py

```

If you have anything to add, give me a pull request
