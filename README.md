# NIC-case-study
# Credit Card Fraud Detection
This Jupyter Notebook contains a comprehensive analysis of credit card transactions data to detect fraudulent activities using Nature Inspired Computing (NIC) techniques. The notebook covers data preprocessing, feature selection, and classification using various machine learning algorithms.

# Data Preprocessing
Two CSV files, train_identity.csv and train_transaction.csv, are merged  into a single dataframe df. Then we performed data preprocessing, including handling missing values, encoding categorical variables, and scaling numerical variables.

# Feature Selection
The notebook uses a genetic algorithm to select the most relevant features from the preprocessed data. It defines a fitness function that evaluates the accuracy of a classification model using the selected features. The genetic algorithm iteratively selects the best features based on the fitness function.

Additionally, the notebook uses Particle Swarm Optimization (PSO) to optimize the feature selection process. It defines a binary PSO algorithm to select the most relevant features from the preprocessed data. The PSO algorithm iteratively updates the position of particles based on the fitness function.

# Classification
We used various classification models, including Decision Trees, Random Forest, AdaBoost, and Support Vector Machines (SVM), to evaluate the performance of the selected features.
# Results
The notebook prints the accuracy scores for each classification algorithm using the selected features. It also prints the selected features and their corresponding importance scores.

# Dependencies
The notebook requires the following Python libraries:

pandas
numpy
scikit-learn
seaborn
matplotlib
plotly
pyswarms
Usage
To run the notebook, clone the repository and open it in a Jupyter Notebook environment. Ensure that all the required libraries are installed. The notebook can be executed by running each cell in order.
