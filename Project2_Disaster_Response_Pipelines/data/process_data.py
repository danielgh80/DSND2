import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
import sqlalchemy


def load_data(messages_filepath, categories_filepath):
    """
    Function: load_data from message and categories csv files and merge them
    Args：
      messages_filepath(str): messages file path
      categories_filepath(str): categories files path
    Return：
       df： merge messages and categories
    """
    # Load messages
    messages = pd.read_csv(messages_filepath)
    # Load categories
    categories = pd.read_csv(categories_filepath)
    # Merge datasets
    df = pd.merge(categories,messages,on='id',how='outer')
    return df


def clean_data(df):
    """
    Function: clean data
    Args:
        df(pd.dataframe):raw dataset
    Return:
        df(pd.dataframe):clean dataset
    """
    # Make dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)
    # Select the first row of the categories dataframe
    row = [col_.split('-')[0].strip() for col_ in list(categories.iloc[0])]
    # Extract a list of new column names for categories
    category_colnames = row
    # Rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str)
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x:int(x.split('-')[1].strip()))
        categories.head()
    # Drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # Drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Function: save clean data
    Args:
        df(pd.dataframe):clean dataset
    Return:
        N/A
    """
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('disaster_table', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
            .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
