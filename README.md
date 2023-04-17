# Steam-Game-Recommendation-System


## I. Introduction
The gaming industry is a rapidly growing market with billions of dollars in revenue and millions of players worldwide. With the advent of digital platforms like Steam, it is easier for users to find and purchase the game they wish to play. However, with the increasing variety of games on the market, the lack of personalization and discovery of new games has become a problem for many users. Game recommendation systems can potentially increase user engagement with the platform and increase game sales. By providing personalized recommendations, users can be presented with games that match their interests and taste better. In addition, the system can be used to identify games that users might not otherwise have considered, giving them a wider choice of games to play.


## II.Dataset processing and analysis
In this project, we use  two different datasets extracted from Steam, which are available for free on Kaggle.
### [Dataset](https://github.com/L408987636/Steam-Game-Recommendation-System/tree/main/dataset)
The first dataset(steam-200k.csv) is the user behavior dataset, which contains 200,000 rows, including 5,155 unique games and 12,393 unique users. It has four attributes, user id, game name, behavior ('purchase' or 'play') and value of behavior. 

The second dataset is the game dataset(steam_games.csv), which comprises a list of games and their corresponding information, such as their descriptions, the URL that leads to the Steam store, the game title, a brief description, recent reviews, all reviews, release date, developer, publisher, popular tags, game details, languages, achievements, genre , game description, description of mature content, minimum requirements to run the game, recommended requirements, original price, and price with a discount. This dataset includes a total of 51,920 games. 

### [Processing And Analysis](https://github.com/L408987636/Steam-Game-Recommendation-System/tree/main/data_analysis)
There is some redundancy in this dataset. For convenient, we convert four attributes three attributes, user id, game name and play hours.  We also delete the rows of ‘purchase’ behavior because every row has the same value. Each row in the new user behavior dataset represents a unique user-game behavior. Total number of rows reduced from 200,000 to 128,804. 

After that, Players with less than 10 games, games that are played by less than 30 players, and games without playing time are removed. 

For data partitioning, we use 80% of dataset for training, 20% of dataset for testing. 
## III. Algorithms Implementation

### [KNN](https://github.com/L408987636/Steam-Game-Recommendation-System/tree/main/KNN)
The KNN-Basic algorithm is based on users’ ratings of the items. In many real-world situations, however, we must rely on implicit ratings, such as how many times a user has played a game in the dataset we obtained. We assumed that the time users spend playing video games is directly proportional to their preferences. Before applying the KNN-Basic algorithm, we convert the users’ playtime into explicit ratings from 1 to 5 according to the time spent on each game. The Python function pd.cut() is used to achieve this task. 

<img width="416" alt="image" src="https://user-images.githubusercontent.com/71811311/232437326-bb6095cf-bac5-451a-bfd5-3fbc215e3aa0.png">

<img width="177" alt="image" src="https://user-images.githubusercontent.com/71811311/232437351-d9c44770-67dd-49b8-be1f-08af9ffcc5d7.png"><img width="202" alt="image" src="https://user-images.githubusercontent.com/71811311/232437362-76ac7736-8154-405f-baa4-dcb30bf56ea2.png">


For the KNN-Basic implementation, we used the external Surprise library and chose to use the cosine distance. The similarity matrix can be directly calculated using the build in function in the Surprise library.

<img width="207" alt="image" src="https://user-images.githubusercontent.com/71811311/232437541-7b4eef75-2a65-4d28-a41c-edc4ade4495e.png">

 
A function was built to perform the recommendation of top N games for a user. The idea is to rank all the games according to their predicted high to low scores and select the top N games to our recommendation list after removing the games that a user already owned. Below is an example of the recommendation result for user 5250 choosing k=10 and N=10.

 <img width="255" alt="image" src="https://user-images.githubusercontent.com/71811311/232437566-c98681f8-15a8-4cdf-8bef-053ec47a2d1f.png">
 
<img width="256" alt="image" src="https://user-images.githubusercontent.com/71811311/232437586-57ed8d1d-b0e4-410d-bd99-4cad87e04f1b.png">

 

### [SVD](https://github.com/L408987636/Steam-Game-Recommendation-System/tree/main/SVD)
Singular Value Decomposition (SVD) is one of the most widely used Unsupervised learning algorithms. The key idea of SVD algorithm is to map users and items to a common latent factor space of dimensionality k, so that user-item interactions are modeled as inner products in that space. The latent space attempts to explain ratings by characterizing products and users into factors automatically inferred from user feedback (Rosaria Bunga, Fernando Batista1, and Ricardo Ribeiro).

Before training the model, the first step is to process the dataset. As steam games are refundable within two hours of purchase, user data with less than two hours of play time needs to be deleted. Users and games will be renumbered in order to facilitate the correspondence between users and games, and thus the creation of the rating matrix.
The processed dataset is shown as follow:

<img width="416" alt="image" src="https://user-images.githubusercontent.com/71811311/232437825-bb90e0cc-657f-42e2-a22b-a8d01a3a6cb8.png">


In this part the data ‘loghrs’ is chosen to be the rating for the users to games. The code to make matrix is shown as follow:

<img width="416" alt="image" src="https://user-images.githubusercontent.com/71811311/232437852-100d147e-6709-488c-a6dc-a693ae27f1fb.png">

Splitting the data set into a test set and a training set, and then zeroes out the user-game pairs for the test set in the rating matrix. Only use the train set to train the model. The code is shown as follow

<img width="382" alt="image" src="https://user-images.githubusercontent.com/71811311/232438098-12799bde-9b6e-4042-81f7-8f01696782e6.png">

<img width="390" alt="image" src="https://user-images.githubusercontent.com/71811311/232438134-82642bdc-db6a-4d36-88c6-462ad2d63571.png">


Aftering 200 times iteration the SSE got the smallest. The picture below shows rate of change of the SSE and RMSE.

<img width="386" alt="image" src="https://user-images.githubusercontent.com/71811311/232438177-17d8e27f-4a0c-41c4-8961-98423ac69065.png">


Now the vacant values in the rating matrix are filled with the predicted scores. Sorting the scores from highest to lowest will get a table of users preferences for the games
Then the games with the highest scores can be recommended to users.

### [ALS](https://github.com/L408987636/Steam-Game-Recommendation-System/tree/main/ALS)
ALS is an iterative optimization process based on matrix factorization, in which we try to get closer and closer to a factorized representation of our original data at each iteration.

Before the implementation of ALS algorithm, we used the Python function parse.csr_matrix() to transform the user-game matrix with playtime into a sparse matrix.
Initially, we implemented the ALS functions based on the above theory directly. However, the computation cost is very high. Finding the final solution took more than half an hour, even using only half of the dataset. To reduce the running time, we optimized our algorithm with the help of an external library called Implicit. We can build the ALS model using the built-in function in the Implicit library directly. After fitting the model, we can get the user-factor and item-factor matrices through model.user_factors and model.item_factors. 

<img width="416" alt="image" src="https://user-images.githubusercontent.com/71811311/232438278-185a9093-396f-4f80-b0db-b6275c166a64.png">

For the recommendation process, we can use the recommend function built by ourselves or directly use the model.recommend function in the Implicit library. The two methods will give the exactly same result.

<img width="476" alt="image" src="https://user-images.githubusercontent.com/71811311/232438629-c2978525-2407-44b6-979f-97025e2e9db2.png">

### [Content-based Recommendation](https://github.com/L408987636/Steam-Game-Recommendation-System/tree/main/content_based) 

Content-based recommendation algorithm is an item-based methodology. It uses multiple features of items as its input for analysis. For every game and its feature, we implement is as a high-dimension vector according to the words in the features. The idea is similar from NLP word-embedding process. One game corresponds to one vector. We can build a similarity matrix based on the vectors.

First, we need to do the data pre-processing to our datasets. We matched up the games both in user dataset and game dataset and filtered the unmatchable ones. The names are compared with pure letters without any symbol or capital. Only games that were both seen in user dataset and game dataset were be accepted for further research.

<img width="416" alt="image" src="https://user-images.githubusercontent.com/71811311/232439047-73fe4a16-dc2a-43e9-8d31-87f83540d23f.png">


Then, we implemented the similarity matrix mentioned above using the following codes.

<img width="416" alt="image" src="https://user-images.githubusercontent.com/71811311/232439069-60fad8da-4ec0-45fd-9fd7-43f64f1c4e85.png">


For every game in the test set that one user has, we generated a group of recommended games. Then, if one game is recommended more than one time, it would be weighted as higher probability that the user may like it. In the end, we output the required number of games based on the process above. Here’s the core codes of recommendation.

<img width="416" alt="image" src="https://user-images.githubusercontent.com/71811311/232439094-20c42880-13c0-41a7-ab15-9e4d68b1702d.png">

Here's a clip of the results of recommendation for each user when k=5.

<img width="416" alt="image" src="https://user-images.githubusercontent.com/71811311/232439156-dd1d9354-46e7-49f1-b03d-6ea9dbaed919.png">


## IV. Results

<img width="416" alt="image" src="https://user-images.githubusercontent.com/71811311/232439189-b704bfa9-f1d8-4fc5-a213-cf6de0062c5e.png">


There are total 5 collaborative filtering recommendation systems for the video game domain in this project, the table above shows the project results. A comparison of the results in the table shows the ALS algorithm achieved the best results. It performs well in all areas. But the presented algorithms didn’t achieve desirable result.
