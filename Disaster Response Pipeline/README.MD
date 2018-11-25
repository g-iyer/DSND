## Hello and Welcome !
Here you will find my files and code for the Disaster Response Pipeline Project
This project combines multiple activities
  - ETL activities
  - ML activities (including NLP activities)
  - Deployment
    
## Background:
  This project is based on a project by FigureEight datasets that were collected for a disaster response project.
  This implementation merges, cleans and then creates a SQL database that will be used by a Machine Learning model to categorize the data.
  Following this a web application is used to solicit disaster messages and then suggests where to direct the aid. 
  
 ## Workspace (folder) structure:
 
  **/app** : containes the files for the implementation of the web application
 
 **/data** : containes the messages and categories (CSV) files and will contain the newly created SQL database file
 
 **/models** : containes the classifier code
  
## Instructions:
  From the root directory please execute these steps:
  1.  *python data/process_data.py data/disaster_message.csv data/disaster_categories.csv data/DR.db* 
  
      **Please note you may have to delete DR.db if it already exists**
      
      - This step will Extract, Transform and Load (ETL) the messages and categories that were collected as input to the project
      
      - The output will be a SQL table in a SQL database
      
  2.  *python models/train_classifier.py data/DR.db dr_project.sav*
  
      **Please use the same DB that was created in the last step**
      
      - This step will execute the Machine Learning (ML) part of the project by training the classifier model and saves it
      
  3.  *python app/run.py*
      
      - This program will execute the web app and use the trained model to classify operator input and display the interpreted data and action group.
 


(dataset_overview.jpg?raw="true")

