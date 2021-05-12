## Naive Bayes Classifier for Hate Tweets

​	My implementation of Naive Bayes Classifier for detecting the hate tweets. For mathematical details, see the comments in scripts.



### Requirements

​	`pip install -r requirements.txt`



### Dataset

​	The training set is downloaded from [CLAWS](http://claws.cc.gatech.edu/covid/#dataset).



### How to run the code

​	Just run `python NBClassifier.py` if using the pretrained model file. Otherwise, run `python retrieve_tweets.py` to create your own dataset first.



### Performance

​	The model is trained on the COVID-HATE data (Jul 8 - Aug 7) and the testing accuracy is 42.29%. When removing the stopwords during training, the testing accuracy is 42.78%.
