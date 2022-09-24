# Netflix-Movies-and-TV-Shows-Clustering-
This project is a part of "Unsupervised Machine Learning” curriculum as capstone projects at AlmaBetter School 
I have clustered similar movies and TV Shows available on Netflix taking into account of attributes like Description, Cast, Director, Genre etc of a particular movie/show.
💾 Project Files Description

This Project includes 1 colab notebook, 1 technical documentation as well as 1 presentation:
Executable Files:

    NETFLIX MOVIES AND TV SHOWS CLUSTERING.ipynb - Includes all functions required for classification operations.

Output:

    Google Colab - All the outputs are visible in the provided colab notebook. 

Input Files:

    NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv - Input dataset having information about different shows/movies available on Netflix.




    

📖Introduction
Netflix, the world’s largest on-demand internet streaming media and online DVD movie rental service provider.it Founded August 29, 1997, in Los Gatos, California by Marc and Reed. It has 69 million members in over 60 countries enjoying more than 100 million hours of TV shows and movies per day Netflix is the world’s leading internet entertainment service with enjoying TV series, documentaries, and feature films across a wide variety of genres and languages. I was curious to analyze the content released in Netflix platform which led me to create these simple, interactive, and exciting visualizations and find similar groups of people.
📖 Problem Statement
This dataset consists of tv shows and movies available on Netflix as of 2019. The dataset is collected from Flixable which is a third-party Netflix search engine.
In 2018, they released an interesting report which shows that the number of TV shows on Netflix has nearly tripled since 2010. The streaming service’s number of movies has decreased by more than 2,000 titles since 2010, while its number of TV shows has nearly tripled. It will be interesting to explore what all other insights can be obtained from the same dataset.

Attribute Information
    show_id : Unique ID for every Movie / Tv Show

    type : Identifier - A Movie or TV Show

    title : Title of the Movie / Tv Show

    director : Director of the Movie

    cast : Actors involved in the movie / show

    country : Country where the movie / show was produced

    date_added : Date it was added on Netflix

    release_year : Actual Releaseyear of the movie / show

    rating : TV Rating of the movie / show

    duration : Total Duration - in minutes or number of seasons

    listed_in : Genere

    description: The Summary description



Integrating this dataset with other external datasets such as IMDB ratings, rotten tomatoes can also provide many interesting findings.

📖 Data Summary
Netflix is by far the most widely used media and video streaming service. It includes over 8000+ movies with tv shows worldwide. Currently, Netflix has more than 200 million subscribers worldwide. NETFLIX is also the most widely utilized entertainment platform worldwide. It offers a vast library of films and TV series that may be seen at any time through internet services.
Netflix is without a doubt the most used media and video streaming service. There are more than 8000 films and television programmers included. At the moment, Netflix has over 200 million subscribers worldwide. Additionally, NETFLIX is the most widely used entertainment service worldwide. Through online services, it offers a vast library of movies and TV series that may be accessed at any time. 
The tabular dataset includes listings for all Netflix movies and TV shows, together with information about the actors, directors, ratings, release year, duration, and other factors. It has 7787 rows and 12 columns.
Our project's main objective is to build a model that can cluster similar data by matching text-based features.
According to the problem statement, the question arises that, understanding what type of content is available in different countries and Is Netflix increasingly focused on TV rather than movies in recent years we have to do clustering on similar content by matching text-based features. For that we have used K-means Clustering.
To obtain relevant data free of NAN values (null values), we will manipulate the raw data. We will next look at the dataset's summary statistics. With feature engineering, feature scaling, and the removal of unnecessary columns, we prepared a dataset. To use the model in the study, the data was transformed into a common scalar form.
We executed the exploratory data analysis. Reviewing all the data processing then the model was trained to form Collections. In which we remark as below- k-means clustering model gave insights of silhouette analysis consisting of 2,3,4,5,6 clusters.
      For n_clusters = 2 The average silhouette score is 0.42541313028836003
  For n_clusters = 3 The average silhouette score is: 0.3940500353920696
           For n_clusters = 4 The average silhouette score is: 0.38498095158183726
             For n_clusters = 5 The average silhouette score is: 0.3962372504377786
      For n_clusters = 6 The average silhouette score is: 0.392565886828632

In the end, we plot boxplot to predict the hypothesis -
•	After clustering, we can state that the number of TV series that have been released over the past few years is not increasing, which is our alternative hypothesis.
•	Our second alternative hypothesis is the number of TV shows added to Netflix is high.
We calculated that whereas movies make up 97.2 percent of the total, TV series only make up 2.8 percent. When compared to films made in the last 10 years, Netflix has added a lot more movies and TV episodes in the past years, but the numbers are still small. In many nations, people choose to watch movies over TV programming.


Dash Web App:

NetflixRecommender recommends Netflix movies and TV shows based on a user's favourite movie or TV show. It uses a Natural Language Processing (NLP) model and a K-Means Clustering model to make these recommendations. These models use information about movies and TV shows such as their plot descriptions and genres to make suggestions. The motivation behind this project is to develop a deeper understanding of recommender systems. Specifically, thinking about how companies like Netflix and YouTube create algorithms to tailor content based on user interests and behaviour. I created a Dash web app that utlizes my model to provide film recommendations based on a user's favourite movie or TV show.

App
Exploring Solutions:

Having a deeper understanding of what problem we are trying to solve, what the users’ needs, and frustrations are, and what the goals are for achieving the best possible solution for both for the business as well as the user, I began by listing out the possible solutions that were arrived from the research.

    Improve rating system: Use the star rating rather than a thumbs up and thumbs down rating system to help guide in decision making when selecting a film.
    Separate recently watched: Hide the movies and TV-Shows on a separate page so users don’t have to scroll through those already seen. — users have to do more searching
    Randomize a Movie: When users are unsure of what to choose, Netflix will randomly select something to watch based on their viewing history.
    Show popular/trending films: Create a category which showcases only trending content.
    Connect with Friends: It was proven that users watch shows and movies based on friend recommendations so this may be useful for keeping users locked into Netflix for longer.
    Organizing films by the mood: Alongside the genres filter, it may be possible to organize content based on the mood that is experienced after watching the film.

Steps involved:

The full code for this article can be found here. It is implemented in Python and different clustering algorithms are used. Below is a brief description of the general approach that I employed:

    Data cleaning and pre-processing: Here I checked and dealt with missing and duplicate variables from the data set as these can grossly affect the performance of different machine learning algorithms (many algorithms do not tolerate missing data).
    Exploratory Data Analysis: Here I wanted to gain important statistical insights from the data and the things that I checked for were the distributions of the different attributes, correlations of the attributes with each other and the target variable and I calculated important odds and proportions for the categorical attributes.
    Clustering: Clustering or cluster analysis is a machine learning technique, which groups the unlabelled dataset. It can be defined as "A way of grouping the data points into different clusters, consisting of similar data points. The objects with the possible similarities remain in a group that has less or no similarities with another group." It does it by finding some similar patterns in the unlabelled dataset such as shape, size, colour, behaviour, etc., and divides them as per the presence and absence of those similar patterns. It is an unsupervised learning method; hence no supervision is provided to the algorithm, and it deals with the unlabeled dataset. After applying this clustering technique, each cluster or group is provided with a cluster-ID. ML systems can use this id to simplify the processing of large and complex datasets.

Conclusion

In conclusion, tailored recommendations can be made based on information about movies and TV shows. In addition, similar models can be developed to provide valuable recommendations to consumers in other domains. It will solve for improved movie and TV-Show selection times with a considerable growth in satisfaction of the content being consumed leading to more user engagement and greater trust in Netflix recommendations.

1.	Most films were released in the years 2018, 2019, and 2020.
2.	TV shows account for 2.8 percent of the total, while movies account for 97.2 percent.
3.	Dramas is a genre that is mostly watched on Netflix and as per audience preference international movies are mostly watched.
4.	The largest count of Netflix content is made with a “TV-14” rating,
5.	The United States, India, the United Kingdom, Canada, and Egypt are the top five producer countries.
6.	Netflix has added a lot more movies and TV episodes in the previous years, but the numbers are still low when compared to movies released in the last ten years.
7.	Movies are mostly watched in various countries rather than TV shows.
8.	We performed data engineering to remove the unnecessary variables and to convert the data into standardized form into scalar.
9.	Implemented model is based on the K-means clustering algorithm consisting of 2,3,4,5,6 clusters.
Silhouette Analysis score for K-means :
•	For n_clusters = 2, silhouette score Is 0.42541313028836003
•	For n_clusters = 3, silhouette score is 0.3940500353920696
•	For n clusters = 4, silhouette score is 0.38498095158183726
•	For n_clusters = 5, silhouette score is 0.3962372504377786
•	For n_clusters = 6, silhouette score is 0.3925658868286329
11.	After clustering, we can say that our alternative hypothesis is that the number of TV shows launched in the previous few years is NOT growing.
12.	Our second alternative hypothesis is the number of TV shows added to Netflix is higher.

