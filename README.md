# DSND2

Repository of projects for Data Scientist Nanodegree Term 2

Since previous reviewers did not see the correct README.md file within the project's folder, I'm including it's content here.




# Data Scientist Nanodegree Term 2

## Writing a blog post: Applying Data Analysis and Machine Learning to AirBnB data from Madrid and Barcelona.

In this project I analyze consumer patterns for AirBnB listings in Madrid and Barcelona. I analyze the data and use serveral models to predict and find out:

  1. What moves users to choose from the wide range of rentals in offer?
  2. Do they look for the same things in different destinations?
  3. What city yields the most revenue?
  4. What are their revenue patterns?

## Summary of Results

   1. Guest seem to be mainly booking properties with Superhosts and a large number of amenities according to OLS correlations and Random Forest models.
   2. Yes. They look for the same things. According to the Random Forest models of both data sets, properties run by Superhosts and with a large number of amenities in the description are good predictors in both datasets.
   3. Barcelona has the higher average daily price with $92.27 and Madrid a lower one with $68.06.
   4. Both have a seasonal revenue pattern. In both summer has a higher revenue than in winter, but Madrid's pattern seems to be more linear and Barcelona has the larger differences between seasons.

## Link to the blog post

[Here is a blog post on medium](https://medium.com/@danielgh/what-do-travelers-look-for-when-booking-vacation-rental-382af4e61f30)

## Dependecies and packages

- Python 3
- Numpy
- Pandas
- Scikit learn
- seaborn
- matplotlib
- statsmodels
- IPython

## Repository content

- Jupyter notebook file 'airbnb_analysis.ipynb'.
- Folder 'data' to contain 'barcelona-airbnb-open-data' and 'madrid-airbnb-open-data'. Data can be downloaded from [here](https://cloud.insoft.es/s/acfq983SwrifgKB).
- Folder 'images' containing figures used in the in the Medium article.
- MIT License file.

## Source data

Updated files used in this project can be downloaded from:

- [Inside Airbnb](http://insideairbnb.com/get-the-data.html)




# Disaster Response Pipeline Project


## Installation

I used python 3.6 and the libraries are:

nltk==3.4
numpy==1.15.4
pandas==0.23.4
plotly==3.4.2
scikit-learn==0.20.2
scipy==1.2.0
sklearn==0.0
SQLAlchemy==1.2.15


## Project Motivation

Project code is deployed a program as a web application which is part of the Udacity Data Scientist Nanodegree program. The project uses Tweets and SMS's from real live disaster situations that have been collected and labelled by Figure Eight. The data is analyzed to build a model for an API that classifies disaster messages.

A machine learning pipeline to label these and new messaages into the appropiate categories in order to foward each message to an appropriate agency/department to handle the situation. The data is first divided into a training and a test set, then, a machine learning pipeline that uses NLTK, scikit-learn and GridSearchCV to produce a final model that is saved as a pickle file.

The web application uses bootstap and Flask. There, an emergency worker can input a new message and get it labeled for several categories.

Also, the application displays visualizations for the data.


## File Descriptions

- \
	- README.md
	- ETL_Pipeline Preparation.ipynb
	- ML:Pipeline Preparation.ipynb
	- LICENSE.txt.
- \app
	- run.py
	- \templates
	   - go.html
	   - master.html
- \data
	- DisasterResponse.db
	- disaster_categories.csv
	- disaster_messages.csv
	- process_data.py
- \models
	- classifier.pkl The file can be downloaded from [here](https://cloud.insoft.es/s/NdHtSL7cRKGFM7d)
	- train_classifier.py


## Instructions:

     1. Run the following commands in the project's root directory to set up your database and model.

         - To run ETL pipeline that cleans data and stores in database
             `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
         - To run ML pipeline that trains classifier and saves
             `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

     2. Run the following command in the app's directory to run your web app.
         `python run.py`

     3. Go to http://0.0.0.0:3001/



# Dog Breed Detector Project

This is a repository of the Capstone project for the Term 2 of the Data Scientist Nanodegree.

## Contents

The following are the files and folders for this project.

app
bottleneck_features
dog_app.html
dog_app.ipynb
dog_images
extract_bottleneck_features.py
haarcascades
images
LICENSE
project.md
README.md


## App requirements

- At 1st, this is my python info:

     Python 3.6


- Requirements:

 Flask==1.0.2
 Flask-RESTful==0.3.6
 h5py==2.8.0
 image==1.5.24
 Keras==2.2.0
 matplotlib==2.2.2
 mistune==0.8.3
 nltk==3.3
 numpy==1.14.5
 opencv-python==3.4.1.15
 Pillow==5.1.0
 pip==18.1
 scikit-learn==0.19.1
 scipy==1.1.0
 tensorflow-gpu==1.8.0
 tqdm==4.23.4

## Running the app

- In order to run the app, make sureall dependencies are intalled and run the following command within the /app/ directory:

     python run.py

- The app will be then accesible via http://localhost:5000

- Upload any image located at /app/upload/ and the image will tell you if it's a dog, a human or something else and what breed of dog it looks closer to.
