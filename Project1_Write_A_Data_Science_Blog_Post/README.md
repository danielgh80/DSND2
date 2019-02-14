# Data Scientist Nanodegree Term 2 Blog Post
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

# Link to the blog post

[Here is a blog post on medium](https://medium.com/@danielgh/what-do-travelers-look-for-when-booking-vacation-rental-382af4e61f30)

# Dependecies and packages

- Python 3
- Numpy
- Pandas
- Scikit learn
- seaborn
- matplotlib
- statsmodels
- IPython

# Repository content

- Jupyter notebook file 'airbnb_analysis.ipynb'.
- Folder 'data' to contain 'barcelona-airbnb-open-data' and 'madrid-airbnb-open-data'. Data can be downloaded from [here](https://cloud.insoft.es/s/acfq983SwrifgKB).
- Folder 'images' containing figures used in the in the Medium article.
- MIT License file.

# Source data

Updated files used in this project can be downloaded from:

- [Inside Airbnb](http://insideairbnb.com/get-the-data.html)
