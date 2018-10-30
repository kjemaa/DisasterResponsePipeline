# Disaster Response Pipeline
Udacity project: Analyze disaster data fro Figure Eight and build a model for an API that classifies disaster messages.

### Installations
Python 3.7

libraries: 
- pandas
- numpy
- nltk
- sklearn
- sqlalchemy
- pickle
- re



### Project components: 
The project consists of three components

**1. ETL Pipeline**
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data 
- Stores it in a SQLite database

**2. ML Pipeline**
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

**3. Flask web app**
- categorize new messages
- visualisations of the training set genres and categories


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### File structure:
 Here's the file structure of the project:

**app** 

| - template
 
| |- master.html # main page of web app 

| |- go.html # classification result page of web app

|- run.py # Flask file that runs app 

**data**

|- disaster_categories.csv # data to process 

|- disaster_messages.csv # data to process 

|- process_data.py
 
|- DisasterResponse.db # database to save clean data to 

**models** 

|- train_classifier.py 

|- classifier.pkl # saved mode


|- README.md


### Resources
- Data has been provided by Figure Eight

- sklearn tutorials

http://scikit-learn.org/stable/modules/model_evaluation.html#classification-report

http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble

- pandas tutorials

http://pandas.pydata.org/Pandas_Cheat_Sheet.pdf



