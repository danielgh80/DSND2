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


Package                 Version
----------------------- ---------
Flask                   1.0.2
Flask-RESTful           0.3.6
h5py                    2.8.0
image                   1.5.24
Keras                   2.2.0
matplotlib              2.2.2
mistune                 0.8.3
nltk                    3.3
numpy                   1.14.5
opencv-python           3.4.1.15
Pillow                  5.1.0
pip                     18.1
scikit-learn            0.19.1
scipy                   1.1.0
tensorflow-gpu          1.8.0
tqdm                    4.23.4

## Running the app

- In order to run the app, make sureall dependencies are intalled and run the following command within the /app/ directory:

python run.py

- The app will be then accesible via http://localhost:5000

- Upload any image located at /app/upload/ and the image will tell you if it's a dog, a human or something else and what breed of dog it looks closer to.
