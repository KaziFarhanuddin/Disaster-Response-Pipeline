import os
import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    This functio loads data and load data

    Parameters
    ----------
        messages_filepath: String 
            File name containing messages data set
        categories_filepath: String
            File name containing categories data set

    Returns
    -------
        df: pandas.DataFrame
            Dataframe containing message & categories
    '''
    # Reading values from csvs and merging them
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=['id', ])
    return df


def clean_data(df):
    '''
    This function takes in a dataframe and cleans its data and return a new dataframe

    Parameters
    ----------
        df: pandas.DataFrame
            The dataFrame containing data to be cleaned

    Returns
    -------
        df: pandas.DataFrame
            A new dataFrame containing cleaned data
    '''
    # Getting all categories
    categories = df['categories'].str.split(';', expand=True)
    category_colnames = categories.iloc[0].str[:-2]
    categories.columns = category_colnames

    # Setting each value to be the last character
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)

    # Droping categories columns and adding categories datafroame to df dataframe 
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Scalling related from 0-2 to 0-1
    df.replace({'related':2}, 1, inplace=True)

    # Removing duplicate rows
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
    This function takes in a dataframe an a path to database, 
        and saves the data into the database, it will also delete the database
        file it already present

    Parameters
    ----------
        df: pandas.DataFrame
            The dataFrame to be saved
        database_filename: String
            Path where to save the data

    Returns
    -------    
        None:
    '''

    # Saving cleaned data to data base
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_management', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
