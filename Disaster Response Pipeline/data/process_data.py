import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
#    pass
	""" Function to load messages and categories files & merge them
    	Takes 2 arguments 
		messages_filepath : location of the messages csv file
		categories_filepath : location of the categories file
		Outputs one dataset df (the merged output file)
	"""
	
	messages_ds = pd.read_csv(messages_filepath)
	categories_ds = pd.read_csv(categories_filepath)
	df = pd.merge(messages_ds, categories_ds, left_on='id', right_on='id', how='outer')
	return df

def clean_data(df):
#    pass
	""" Function takes as input the merged dataset from load_data, cleans it and returns the cleaned dataset
		Input: df dataset with categories and messages merged
		Returns: df cleaned dataset 
	"""
# split the categories by separator 
	categories = df.categories.str.split(pat=";", expand = True)
# take the first row as header	
	row=categories.loc[0]
	category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
	categories.columns = category_colnames
#	categories.related.loc[categories
	for column in categories:
    	# set each value to be the last character of the string
        	categories[column] = categories[column].transform(lambda x:x[-1:])
    	# convert column from string to numeric
        	categories[column] = pd.to_numeric(categories[column])
		
	df.drop('categories', axis=1, inplace= True)
	# concatenate the original dataframe with the new `categories` dataframe
	df = pd.concat([df,categories],axis=1)
	# drop duplicates
	df.drop_duplicates(inplace=True)
	return df
	

def save_data(df, database_filepath):
#    pass
	""" Function to save the merged and cleaned data to a SQL database table
	"""  
	engine = create_engine('sqlite:///'+ database_filepath)
	df.to_sql('msgs_cat_table', engine, index=False, if_exists='replace')

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