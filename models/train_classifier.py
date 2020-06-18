import pickle
import re
import sys
import warnings

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")

nltk.download(['punkt', 'stopwords', 'wordnet'])

STOPWORDS = stopwords.words("english")
LEMMATIZER = WordNetLemmatizer()


def load_data(database_filepath):
    '''
    This function reades data from data base and returns features, labels & catagories

    Parameters
    ----------
        database_filepath: String
            The path to the data base to be read

    Returns
    -------
        X: pandas.DataFrame 
            The features of the data
        Y: pandas.DataFrame 
            The labels of the data
        catagories: List 
            Names of all catagorie present in the data set
    '''
    # Reading data from data base
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_management', engine)

    # Spltting data into X(features) and Y(labels)
    X = df['message']
    Y = df.drop(['id', 'original', 'genre', 'message'], axis=1)

    # Getting categories names
    categories = Y.columns.tolist()

    return X, Y, categories


def tokenize(text):
    '''
    This function takes in test, removes stock words, tokenizes and
        Lemmatizes it and return a list of words

    Parameters
    ----------
        text: String
            The string to tokenize

    Returns
    -------
        clean_tokens: List
            List containg tokenized string words
    '''
    # Removing non letter and number characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize
    tokens = word_tokenize(text)

    # remove stop words
    words = [t for t in tokens if t not in STOPWORDS]

    # Lemmatizing
    clean_tokens =  [LEMMATIZER.lemmatize(w) for w in words]

    return clean_tokens


def build_model():
    '''
    This function builds a model and return it

    Parameters
    ----------
        None:

    Returns
    -------
        model: Object
            A classifier fitted to the data
    '''
    # Building the model
    pipeline = Pipeline([
        ('cv', CountVectorizer(tokenizer=tokenize)),
        ('tfid', TfidfTransformer()),
        ('clf', RandomForestClassifier(n_estimators=10, n_jobs=-1))
    ])

    # initializing the parameters
    parameters = {
        'cv__ngram_range': ((1, 1), (1, 2)),
        'tfid__use_idf': [True, False],
    }
    
    # GridSearchCV to find the best combination of parameters for the model
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=1, n_jobs=-1) 

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function evaluates the model with data and displayes the results

    Parameters
    ----------
        model: Classifier
            The classifier to be evaluated
        X_test: pandas.DataFrome
            The features data frame, containg test data features
        Y_test: pandas.DataFrame
            The labels data frame, containg test data labels
        category_names: List
            A list containing names of different categories

    Returns
    -------
        None:
    '''
    # Converting Y_test into numpy.array 
    Y_test = Y_test.values
    # Predicting values of Y_test using X_test (features) 
    Y_pred = model.predict(X_test)

    # Calculating and printing classification_report for all catagories
    print("Classification Report:\n")
    print(classification_report(Y_test, Y_pred))


def save_model(model, model_filepath):
    '''
    This function pickles an object to filepath provided

    Parameters
    ----------
        model: Object
            This can be any object which is to be pickles
        model_filepath: String
            This is the location where the object pickle is to placed

    Returns
    -------
        None
    '''
    # Svaing our model in a file
    pickle.dump(model, open(model_filepath, 'wb'))


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

        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
