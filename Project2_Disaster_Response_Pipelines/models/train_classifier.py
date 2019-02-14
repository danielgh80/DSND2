import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier


def load_data(database_filepath):
    """Load cleaned data from database into dataframe.
    Args:
        database_filepath: String. It contains cleaned data table.
        table_name: String. It contains cleaned data from disastertable.
    Returns:
       X: numpy.ndarray. Disaster messages.
       y: numpy.ndarray. Disaster categories for each messages.
       cat: list. Disaster category names.
    """
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)

    connection = engine.raw_connection()
    table_name = str(engine.table_names()[0])

    df = pd.read_sql("SELECT * FROM '{}'".format(table_name), con=connection)

    cat = df.columns[4:]

    X = df['message'].values
    y = df[cat].values

    return X, y, cat


def tokenize(text, lemmatizer=WordNetLemmatizer()):
    """Tokenize text (a disaster message).
    Args:
        text: String. A disaster message.
        lemmatizer: nltk.stem.Lemmatizer.
    Returns:
        list. It contains tokens.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Detect and replace URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # Remove all non-alpha-numeric characters and tokenize text
    clean_tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))

    # Remove stopwords
    clean_tokens = [t for t in clean_tokens if t not in stopwords.words('english')]

    # Lemmatize tokens
    clean_tokens = [lemmatizer.lemmatize(t) for t in clean_tokens]

    return clean_tokens


def build_model():
    """
    Function: build model that consist of pipeline
    Args:
      N/A
    Return
      cv(model): Grid Search model
    """
    # Set pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))
    ])

    # Set parameters for gird search
    # Using all CPU cores, otherwise it takes forever...
    parameters = {
        'vect__min_df': [1, 10],
        'vect__lowercase': [True, False],
        'tfidf__smooth_idf': [True, False],
        'clf__estimator__min_samples_split': [2, 5],
        'clf__estimator__n_estimators': [10, 20]
    }

    # Set grid search
    cv = GridSearchCV(pipeline, param_grid = parameters, cv = 2, n_jobs = -1)

    return cv


def evaluate_model(model, X_test, y_test, cat):
    """Evaluate model
    Args:
        model: sklearn.model_selection.GridSearchCV.  It contains a sklearn estimator.
        X_test: numpy.ndarray. Disaster messages.
        y_test: numpy.ndarray. Disaster categories for each messages
        category_names: Disaster category names.
    """
    y_pred = model.predict(X_test)

    # Print accuracy, precision, recall and f1_score for each categories
    for i in range(0, len(cat)):
        print(cat[i])
        print("\tAccuracy: {:.4f}\t Precision: {:.4f}\t Recall: {:.4f}\t F1_score: {:.4f}".format(
            accuracy_score(y_test[:, i], y_pred[:, i]),
            precision_score(y_test[:, i], y_pred[:, i], average='weighted'),
            recall_score(y_test[:, i], y_pred[:, i], average='weighted'),
            f1_score(y_test[:, i], y_pred[:, i], average='weighted')
        ))


def save_model(model, model_filepath):
    """Save model
    Args:
        model: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
        model_filepath: String. Trained model is saved as pickel into this file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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
