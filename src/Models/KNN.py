import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist
import numpy as np

class KNN(BaseEstimator, ClassifierMixin):
    def __init__(self) -> None:
        """
        attributes:
            k (integer): banyaknya nearest-neighbour
            distance_metric (enum: ['1', '2', '3', 'manhattan', 'euclidean', 'minkowski']): metode perhitungan distance
            p (integer): nilai p untuk minkowski distance
            data (DataFrame): data yang digunakan sebagai data train
        """
        self.k = 5
        self.distance_metric = "3"

        if ((self.distance_metric == "minkowski") or (self.distance_metric == "3")):
            self.p = 10

        ### end of method ###


    def calculateManhattanDistance(self, test_data: np.ndarray, train_data: np.ndarray):
        distance = cdist(test_data, train_data, metric='cityblock')

        return distance
        ### end of method ###


    def calculateEuclideanDistance(self, test_data: np.ndarray, train_data: np.ndarray):
        distance = cdist(test_data, train_data, metric='euclidean')

        return distance
        ### end of method ###


    def calculateMinkowskiDistance(self, test_data: np.ndarray, train_data: np.ndarray):
        distance = cdist(test_data, train_data, metric='minkowski', p=self.p)

        return distance
        ### end of method ###


    def calculateDistance(self, test_data: np.ndarray, train_data: np.ndarray):
        if ((self.distance_metric == "manhattan") or (self.distance_metric == "1")):
            return self.calculateManhattanDistance(test_data, train_data)
        elif ((self.distance_metric == "euclidean") or (self.distance_metric == "2")):
            return self.calculateEuclideanDistance(test_data, train_data)
        elif ((self.distance_metric == "minkowski") or (self.distance_metric == "3")):
            return self.calculateMinkowskiDistance(test_data, train_data)
        ### end of method ###


    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        train_data = pd.concat([X, y], axis=1)
        # sample_fraction = min(1, 10000 / len(train_data))
        # train_data = train_data.groupby('attack_cat', group_keys=False).apply(lambda x: x.sample(frac=sample_fraction)).reset_index(drop=True)
        self.data = train_data

        ### end of method ###


    def predict(self, data_to_predict: pd.DataFrame):
        class_predictions = []
        features = self.data.columns.difference(["attack_cat"])
        train_data = self.data[features].values
        train_labels = self.data['attack_cat'].values
        test_data = data_to_predict[features].values

        distances = self.calculateDistance(test_data, train_data)

        for i in range(len(test_data)):
            k_nearest_indices = distances[i].argsort()[:self.k]
            k_nearest_labels = train_labels[k_nearest_indices]
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            class_predictions.append(unique[np.argmax(counts)])

        return class_predictions
        ### end of method ###

# classifier = KNN()
# pipe = run_pipeline(classifier=classifier)


### Unit Test ###


# data = pd.DataFrame({
#     'outlook': ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy", "sunny", "overcast", "overcast", "rainy"],
#     'temp': ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot", "mild"],
#     'humidty': ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"],
#     'windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
#     'play': [False, False, True, True, True, False, True, False, True, True, True, True, True, False]
# })

# data_to_predict = pd.DataFrame({
#     'outlook': ["sunny", "overcast", "rainy", "sunny", "overcast", "rainy"],
#     'temp': ["cool", "mild", "hot", "mild", "hot", "cool"],
#     'humidty': ["high", "normal", "normal", "normal", "high", "high"],
#     'windy': [True, False, True, False, True, False]
# })

# print(data.isnull().sum())





# train_data = pd.read_csv("data/preprocessed_train_data.csv")
# val_data = pd.read_csv("data/preprocessed_validation_data.csv")
# test_data = pd.read_csv("data/preprocessed_test_data.csv")

# # exclude id column
# train_data = train_data.drop(columns=["id"])
# val_data = val_data.drop(columns=["id"])
# test_data = test_data.drop(columns=["id"])

# print("train_data length: ", len(train_data))
# print("val_data length: ", len(val_data))
# print("test data length: ", len(test_data))

# # take 10000 training data with distributed attack_cat
# sample_fraction = min(1, 1000 / len(train_data))
# train_data = train_data.groupby('attack_cat', group_keys=False).apply(lambda x: x.sample(frac=sample_fraction)).reset_index(drop=True)
# # take 5 samples from val_data
# # sample = val_data.head(200)

# # start timer
# import time
# start_time = time.time()




### USING KNN CLASS ###

# knn = KNN(test_data)
# prediction = knn.predict(sample)

# # calculate accuracy
# accuracy = (prediction == sample['attack_cat']).mean()
# print(f"Accuracy: {accuracy}")





### USING SCIKIT-LEARN ###

# # use scikit knn to predict
# from sklearn.neighbors import KNeighborsClassifier
# knn_scikit = KNeighborsClassifier(n_neighbors=5)

# # Prepare the data for scikit-learn
# X_train = train_data.drop(columns=["attack_cat"])
# y_train = train_data["attack_cat"]
# # X_val = sample.drop(columns=["attack_cat"])

# # Fit the scikit-learn KNN model
# knn_scikit.fit(X_train, y_train)

# # Predict using the scikit-learn KNN model
# # scikit_prediction = knn_scikit.predict(X_val)
# scikit_prediction = knn_scikit.predict(test_data)

# # print accuracy
# # accuracy = (scikit_prediction == sample['attack_cat']).mean()
# # print(f"Accuracy: {accuracy}")




# # end timer
# print("--- %s seconds ---" % (time.time() - start_time))



# # export the prediction to csv with only id and attack_cat
# # ex:
# # id, attack_cat
# # 0, Normal
# # 1, Exploits

# result = pd.DataFrame({
#     "id": test_data.index,
#     # "attack_cat": prediction
#     "attack_cat": scikit_prediction
# })

# result.to_csv("data/knn_lib_test_prediction.csv", index=False)