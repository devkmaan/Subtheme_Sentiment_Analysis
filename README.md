### Subtheme Sentiment Analysis Assignment Documentation 


## Assignment Overview
The provided document outlines the steps taken to develop a subtheme sentiment analysis model using deep learning techniques. The project's primary goal is to predict the sentiment (positive or negative) of textual reviews. 
## Structure and Organization
The document is well-organized with clear sections: Introduction, Data Collection, Data Pre-processing, Model Development, Model Evaluation, Deployment, and Conclusion
## 1. Introduction
Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) task that involves identifying and classifying subjective information in text data. It is widely used in various applications such as customer feedback analysis, social media monitoring, and market research. This project aims to develop a sentiment analysis model to predict whether a textual review expresses a positive or negative sentiment. The model is built using deep learning techniques and implemented with TensorFlow and Keras.
## 2. Data Collection
•	The dataset is available as name ‘Evaluation-dataset.csv’, which was provided in at with the assignment.
•	The dataset is loaded in Notebook using pandas methods as follows
 ![image](https://github.com/devkmaan/Subtheme_Sentiment_Analysis/assets/140909236/fe1cc351-5d14-40e4-a393-585c895b8885)


## 3. Data Pre-processing
•	Tokenization:  the process of breaking down text into individual units such as words or subwords, which are then used as input for the model to analyze sentiment. This step is crucial for transforming raw text into a format that can be effectively processed by machine learning algorithms.
•	Padding: Ensures that all input sequences have the same length by adding special tokens to shorter sequences. This standardization allows the model to process batches of data efficiently, regardless of the original sequence lengths
•	Encoding: Involves converting tokens into numerical representations that can be processed by machine learning models. This step typically uses techniques such as word embeddings or one-hot encoding to transform textual data into vectors.
•	Splitting: Divide the dataset into training and testing sets.
 ![image](https://github.com/devkmaan/Subtheme_Sentiment_Analysis/assets/140909236/ec6a9e39-bf3a-4cea-898f-418344e6fd2d)

![image](https://github.com/devkmaan/Subtheme_Sentiment_Analysis/assets/140909236/6908840a-b441-43d8-8dc6-c669bb786c6a)

 



## 4. Model Defining
•	Motivation: Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks, are effective for sequence data like text. The model architecture is designed to capture the temporal dependencies in the data and perform binary classification.
Approach:
•	Embedding Layer: Converts integer representations of words into dense vectors of fixed size.
•	LSTM Layer: Captures long-term dependencies in the text.
•	Batch Normalization: Normalizes activations to improve training stability and performance.
•	Dropout: Prevents overfitting by randomly dropping units during training.
•	Dense Layers: Further layers for nonlinear transformations and classification.
 ![image](https://github.com/devkmaan/Subtheme_Sentiment_Analysis/assets/140909236/8eff3d3d-974f-4930-9434-77080f250a3e)

## 5. Model Compiling
•	Compiled themodel using RMSprop optimizer in order to increase accuracy.
•	Binary Cross entropy Loss function has been used in order for better metrics analysis
•	Accuracy was used as the metrics 
 
![image](https://github.com/devkmaan/Subtheme_Sentiment_Analysis/assets/140909236/4d13f962-5f56-47e8-a44d-e8783529f42b)



## 6. Model Training
•	Model was trained at first having set the batch size as 64 and number of epochs as 10.
![image](https://github.com/devkmaan/Subtheme_Sentiment_Analysis/assets/140909236/6349e0a2-42a1-4fdc-84bd-736622233934)

## 7. Model Evaluation
•	Evaluate the trained model on the testing dataset.
•	The Accuracy came out to be 83.78 % with Loss = 0.4601
 ![image](https://github.com/devkmaan/Subtheme_Sentiment_Analysis/assets/140909236/edffe189-d3b7-4f1c-b35b-995114f06ec4)

## 8. Model Training (By changing Parameters)
•	Model was trained at first having set the batch size as 32 and number of epochs as 25 
![image](https://github.com/devkmaan/Subtheme_Sentiment_Analysis/assets/140909236/dce2b6fb-940b-4bd0-b74d-7b353ebc8e3c)

## 9. Model Evaluation (By changing Parameters)
•	Evaluate the trained model on the testing dataset.
•	The Accuracy came out to be 83.31 % with Loss = 0.4441, thus we can observe that our loss decreases by changing parameters.
 ![image](https://github.com/devkmaan/Subtheme_Sentiment_Analysis/assets/140909236/079c06ef-a8a3-435c-8dbd-9da8cecbe822)



## 10. Model Shortcomings
1.	Limited Feature Representation:
o	The current feature engineering might not be sufficient to capture the complexity of the data, leading to suboptimal model performance.
2.	Hyperparameters:
o	Default hyperparameters might not be well-suited for the given dataset, leading to suboptimal performance.
3.	Data Imbalance:
o	The dataset is imbalanced, the model might be biased towards the majority class, affecting the accuracy and overall performance as the ratio of positive to negative subtheme is 4.4 which leads to less perfect model.
## 11. Proposed Improvements
1.	Improve Feature Engineering:
o	Use more sophisticated feature extraction techniques like TF-IDF or word embeddings to better represent the textual data.
2.	Hyperparameter Tuning:
o	Use techniques like grid search or random search to find the best hyperparameters for the model.
3.	Handle Class Imbalance:
o	Use techniques like oversampling, undersampling, or class weighting to address class imbalance.
4.	Cross-Validation:
o	Implement cross-validation to ensure that the model's performance is consistent across different subsets of the data.
5.	Advanced Models:
o	Consider using more advanced models such as ensemble methods (Random Forest, Gradient Boosting) or deep learning models (LSTM, BERT for text data).
6.	Learning Rate Schedulers:
o	Use learning rate schedulers to adjust the learning rate during training for better convergence.
## 12. Conclusion
The overall approach ensures that the text data is clean and uniformly formatted before being fed into a machine learning model. The tokenization and padding steps convert text into a suitable numerical format for the RNN. The chosen model architecture (LSTM) is well-suited for handling sequential data and is designed to prevent overfitting and enhance training stability. This methodical process aims to achieve robust and accurate predictions for the binary classification task at hand.
