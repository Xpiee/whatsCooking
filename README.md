# What’s Cooking? (Kaggle Competition)

## 1.1	Background and Objective
What’s Cooking is a competition hosted by Kaggle. In this competition, we need to predict the category of a dish’s cuisine based on the dish’s ingredients. The dataset is provided by Yummly. The provided dataset includes recipe id, the type of cuisine, and the ingredients required for cooking that recipe. The data is stored in JSON format. The objective of this competition is to train a machine learning algorithm that can predict the type of cuisine by using the ingredients of that cuisine. Looking at the problem statement, we can understand that the given task is a Multi-class classification i.e., there are more than 2 categories to predict. This multi-class classification would require text processing and analysis.

Based on my preliminary analysis, I think that classification algorithms like SGD classifier, Logistic Regression, and Neural Networks would be a good choice of algorithms for predicting multiple categories. Also, we can use SVMs to predict the classes by converting the given problem into one vs all classes problem.


## 1.2	Data Description
As mentioned earlier, the dataset is provided by Yummly has different recipes in different rows. The training dataset has 39774 different recipes, while the test dataset has only 9944 different recipes. Also, the dataset has three attributes:
1. Id: This is a unique identifier for each recipe.
2. Cuisine: This column contains the cuisine type of each recipe. This is our target variable, as we need to predict this in the test dataset.
3. Ingredients: This column contains a list of ingredients required for each dish. We would use this column to train our machine learning algorithm and predict target variable.


## 1.3	Exploratory Data Analysis and Data Pre-Processing

### EDA
Now, let’s jump onto the exploratory data analysis and try to identify important patterns and extract valuable insights from the dataset. These insights would help us design our strategies to pre-process the text given in the column ‘ingredients’. Since, the dataset did not have any missing or null values so, we don’t have to deal with any missing values.

There are 20 unique cuisines in the training dataset. These cuisines along with their distribution, in terms of count, is shown in the figure below:
![Fig. 1: Bar graph showing distribution of Cuisines vs Count](images/Cuisine_Distribution.png)
**Figure 1: Bar graph showing distribution of Cuisines vs Count**


We can see from the above bar graph that majority of the recipes given in the training dataset belongs to either ‘Italian’ or ‘Mexican’ cuisines. Hence, we can conclude that our dataset is imbalanced. Now, let us look at the distribution of ingredient lengths. 

![Bar graph showing distribution of Ingredients vs length](images/ingre_dish.png)


Before doing some more analysis, let us first perform the data cleaning on the train and test dataset to remove unwanted strings and special characters (discussed in next section).
After the data cleaning, let us check what are the top ingredients in all cuisines. 


![Graph showing distribution of top 10 ingredients from all Cuisines](images/ingre_cuisine.png)

### Data Pre-Processing (Data Cleaning)

We can divide our data cleaning into 8 categories. They are listed below:
1. Special Character Exceptions: After going through the train and test datasets, I found that there are certain special words/characters that we would need to remove first before applying any other data cleaning because then we would not be able to spot those special words/characters in the datasets. Some of these special words are ‘\xada’, ‘Ã£’, etc. 
2. Certain Phrases: In the train and test datasets, I identified that there are some words that have similar meaning but are written in different forms e.g., "sun dried" and "sundried" are same. Hence, we can replace one of to another so that there is no discrepancy in the datasets. 
3. Handling Plural Words: In the train and test datasets, there are so many ingredients with their plural forms. Hence, we identified some of those and replaced those since they mean the same ingredients. For example, "drumsticks" and "drumstick".
4. Brand Names: This step was very important because the other text processing techniques can not remove brand names associated with the ingredients in the datasets. I could not identify any other automated method to identify and remove them so, I did it manually. Fortunately, I identified one kernel [1] that had mentioned almost every brand name in the datasets; however, some of them were still missing, so I added them in the code. Some of the brand names associated with the ingredients are kraft, and Tabasco. 
5. Unnecessary Keywords: There were a lot of unnecessary keywords associated with the ingredients in the train and test datasets. These keywords could be removed without the loss of any important information. Some of the examples of these keywords are “drained and chopped”, “thawed”, “firmly packed”, etc.
6. Measurements: I also removed the measurements associated with different ingredients. 
7. Phrases with Similar Meaning: There are a lot of ingredients mentioned with different names but can be replaced with the one main ingredient e.g., “green onion”, "red onion",  and "purple onion" can be replaced with “onion” since they are the same ingredients.
8. Usual Special Characters: Finally, removing the usual special characters in the datasets with space or “no space”.

### Count Vectorizer and TF-ID Vectorizer

Both Count Vectorizer and TFID Vectorizers are the methods for converting the textual data into numerical feature vectors such that these vectors having numerical values can then be used to model machine learning algorithms. Count Vectorizer is a simple vectorizer that converts the textual data into vectors by counting the number of times a word has appeared in the document. Hence, Count Vectorizers are not suitable for a dataset with imbalanced word counts as it gives bias to the most frequent words, ultimately ignoring the rare words.
On the other hand, TFID Vectorizer also converts the textual data into vectors but instead of taking the count of words into account, it considers overall weightage of the word in the document and ultimately helps us to handle rare words present in the documents as well. This could be the main reason why Count Vectorizer did not perform well on this dataset while we could get a way better Kaggle score with TFID Vectorizer.


#### Count Vectorizer

For training our model with Count Vectorizer, I first separated each ingredient by a pipeline i.e., '|' and then for the ‘token_pattern’ parameter of the Count Vectorizer used the pipeline as the separator. By doing so, the count vectorizer considered each ingredient as an individual word (i.e., it considered ‘black olives’ as one ingredient instead of considering two different ingredients ‘black’ and ‘olives’). For the parameter ‘vocabulary’ I used the complete set of ingredients present the training dataset. I made the ‘binary’ parameter as True since, I needed that all non-zero ingredient count should be labeled as 1.  I did not use any other parameter such as “stop_words” or “ngram_range” as they were not required for the current problem. 
With these parameters, I created a vectorized matrix with columns as vectorized features and rows as the number of dishes. Then instead of using this matrix for modelling, I created a sparse matrix from the vectorized matrix by using library csr_matrix from Scipy Sparse. The csr_matrix converts the vectorized matrix into sparse matrix this makes the processing and training very fast.
After training a random forest with best grid searched parameters, I could only score 0.69569 on Kaggle. This score was not satisfactory. So, I decided to move to TFID Vectorizer.

#### TFID Vectorizer

For training with TFID Vectorizer, I used the default parameters of the vectorizer with default value of ‘token_pattern’ and the default value range of parameter ‘ngram_range’ i.e, (1, 1). Default values of the ‘token_pattern’ and ‘tokenizer’ means that my ingredients would be divided on the basis of each word. Also, ‘ngram_range’ of (1, 1) means that only unigrams would be extracted from the ingredients list. This does make sense since we do not need n-grams for this problem statements. For TFID vectorizer, I did not use any stop words as well. Also, very important to mention, I have taken the ‘binary’ parameter as True, this does not mean that the outputs will only have 0/1 values, instead only the tf-term (i.e., term-frequency) in the tf-idf would be binary. The parameters of TFID Vectorizer are mentioned below:

'TfidfVectorizer(analyzer='word', binary=True, decode_error='strict',
                dtype=<class 'numpy.float64'>, encoding='utf-8',
                input='content', lowercase=True, max_df=1.0, max_features=None,
                min_df=1, ngram_range=(1, 1), norm='l2', preprocessor=None,
                smooth_idf=True, stop_words=None, strip_accents=None,
                sublinear_tf=False, token_pattern='(?u)\\b\\w\\w+\\b',
                tokenizer=None, use_idf=True, vocabulary=None)'
