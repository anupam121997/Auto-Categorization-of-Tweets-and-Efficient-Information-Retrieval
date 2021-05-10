# IR2021_Project_30

1.	Downloaded the tweets for the 5 different categories like education, entertainment, politics, sports, and technology using “tweet_download.py”.
2.	Pre-processed the tweets and then removed the duplicate tweets which are further merged to form the dataset using “Preprocess.ipynb” and “Datagen.ipynb”.
3.	Also, create the preprocessed dataset for each category using “datagen_education.ipynb” and similar for other categories.
4.	We have applied different Machine Learning models using tf-idf vectorizer and chose the model which is giving the best accuracy. In our case, the best model is SVM with an accuracy of 87.199% using the file “SVM.ipynb”.
5.	Apart from, tf-idf vectorizer, we have also used Word2Vec and applied different Machine Learning models using “word2vec.ipynb” but it resulted in a decreased accuracy.
6.	We have also tuned the different hyper-parameters of SVM using “Grid_search_SVM.ipynb”.
7.	Generate the TF-IDF Matrix for each of the 5 different categories separately, which require the construction of “docId_tokens”, “document_Id_name_dictionary”, and “inverted_index” using “dict_cons.ipynb”.
8.	Using these dictionaries, we have constructed the “term_freq_matrix” using “count_matrix.ipynb” which is further used to construct the “tf_idf_vector” using “tf_idf_generator.ipynb”.
9.	For, BM25 we have constructed the matrix which is different from TF-IDF using “BM_Matrix.ipynb” for each category separately.
