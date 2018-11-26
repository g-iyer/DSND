"""train classifier python program based on the template
	This will get the required data from the database created earlier, tokenize it and have the ML model process it.
	The model will then be evaluated and saved
"""
	
import sys
# import libraries
# we need many functions for this work
import numpy as np
import pandas as pd
import re
import warnings
warnings.simplefilter('ignore')

# for packaging the model
import pickle
# for SQL operations
from sqlalchemy import create_engine

# for NL processing
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 

# for processing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


def load_data(database_filepath):
#    pass
# load data from database and return with X, Y  and category names
	engine = create_engine('sqlite:///' + database_filepath)
	df = pd.read_sql("SELECT * from msgs_cat_table", engine)
	X =  df['message']

	Y = df.drop(['id','message', 'original', 'genre'], axis = 1)
	category_names = list(Y.columns.values)
	return X,Y, category_names

#def tokenize(text):
#    pass
	
def tokenize(text):
    ''' Function to normalize (remove punctuation and make lowercase), tokenize and stem
        input is "text" string
        output is a list of strings containing normalized and stemmed tokens
    '''
# Remove any punctuation and make the text lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
# Tokenize
    tokens = word_tokenize(text)
# Lemmatize (using Lemmatize insted of stemming - as per reviewer comment)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
# Stem - (removed as per reviewer comment)
#    stemmer = PorterStemmer()
#    stop_words = stopwords.words("english")
#    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return clean_tokens



# Define performance metric for use in grid search scoring object
def f1_metric(y_true, y_pred):
    f1_list = []
#    print (np.shape(y_pred)[1])
    
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i],average='micro')
        f1_list.append(f1)
#        print (i, f1_list)
        
    score = np.median(f1_list)
    return score

def build_model():
	''' Function to build the model with the pipeline, parameters and scorer
		uses f1_metric as the scoring 
		Returns the model	
	'''
#    pass
	pipeline = Pipeline ([('vect', CountVectorizer (tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()), 
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))]) 
                     
	parameters = {'vect__min_df': [1, 5],
	              'tfidf__use_idf':[True, False],
	              'clf__estimator__n_estimators':[10, 25], 
	              'clf__estimator__min_samples_split':[2, 5, 10]}
	
	scores_p = make_scorer (f1_metric)
	
	cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scores_p, verbose=10)
	return cv
	
	
def model_metrics(actual, predicted, labels):
    '''Function to calculate the accuracy, precision, recall and f1 scores of the model's performance 
    	Input: Actual results, Predicted results and category labels
    	Output: array of metrics
    '''
    
    metrics_list = []
    
    for i in range(len(labels)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i],average='micro')
        recall = recall_score(actual[:, i], predicted[:, i], average ='micro')
        f1 = f1_score(actual[:, i], predicted[:, i], average='micro')
        
        metrics_list.append([accuracy, precision, recall, f1])
    
    
    metrics_list = np.array(metrics_list)
    metrics_pr = pd.DataFrame(data = metrics_list, index = labels, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return metrics_pr    

	
def evaluate_model(model, X_test, Y_test, category_names):
#    pass
# Test set (use test data on the model )
    predict_test = model.predict(X_test)
    
    metrics_test = model_metrics(np.array(Y_test), predict_test, category_names)
    
    print(metrics_test)

	

def save_model(model, model_filepath):
#    pass
    pickle.dump(model.best_estimator_,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()