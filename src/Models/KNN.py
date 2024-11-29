import pandas as pd

class KNN:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

        print("Jumlah neighbour (k, integer)")
        
        while True:
            try:
                k_input = int(input(">>> "))
                if k_input <= 0:
                    raise ValueError
                break
            except ValueError:
                print("Input tidak valid. Masukkan angka bulat positif.")

        self.k = k_input

        print("Metrik jarak:")
        print("1. Euclidean")
        print("2. Manhattan")
        print("3. Minkowski")
        print("Pilih metrik jarak (nama atau angka):")
        
        while True:
            try:
                distance_metric_input = input(">>> ")
                distance_metric_input = distance_metric_input.lower()
                
                if distance_metric_input not in ["1", "2", "3", "euclidean", "manhattan", "minkowski"]:
                    raise ValueError
                break
            except ValueError:
                print("Input tidak valid. Pilih metrik jarak yang tersedia.")

        self.distance_metric = distance_metric_input

        if ((self.distance_metric == "minkowski") or (self.distance_metric == "3")):
            print("Masukkan nilai p untuk metrik Minkowski:")
            while True:
                try:
                    p_input = int(input(">>> "))
                    if p_input <= 0:
                        raise ValueError
                    break
                except ValueError:
                    print("Input tidak valid. Masukkan angka bulat positif.")
            
            self.p = p_input
        
        ### end of method ###
    
    def calculateMinkowskiDistance(self, data1: pd.Series, data2: pd.Series, p: int):
        distance = 0
        weight = 1

        for i in range(len(data1)):
            if (pd.api.types.is_numeric_dtype(data1.iloc[i])):
                distance += abs(data1.iloc[i] - data2.iloc[i]) ** p
            else:
                if (data1.iloc[i] != data2.iloc[i]):
                    distance += 1 * weight

        return distance ** (1/p)
        ### end of method ###

    def calculateEuclideanDistance(self, data1: pd.Series, data2: pd.Series):
        return self.calculateMinkowskiDistance(data1, data2, 2)
        ### end of method ###
    
    def calculateManhattanDistance(self, data1: pd.Series, data2: pd.Series):
        return self.calculateMinkowskiDistance(data1, data2, 1)
        ### end of method ###
    
    def calculateDistance(self, data1: pd.Series, data2: pd.Series):
        if ((self.distance_metric == "euclidean") or (self.distance_metric == "1")):
            return self.calculateEuclideanDistance(data1, data2)
        elif ((self.distance_metric == "manhattan") or (self.distance_metric == "2")):
            return self.calculateManhattanDistance(data1, data2)
        elif ((self.distance_metric == "minkowski") or (self.distance_metric == "3")):
            return self.calculateMinkowskiDistance(data1, data2, self.p)
        ### end of method ###

    def predict(self, data_to_predict: pd.DataFrame):
        class_predictions = []

        for index_predict, row_predict in data_to_predict.iterrows():
            # distance format: [[data_index, distance, class], ...]
            distances = []
            for index, row in self.data.iterrows():
                distance = self.calculateDistance(row_predict, row)
                distances.append([index + 1, distance, row['play']])
            
            distances = sorted(distances, key=lambda x: x[1])

            k_nearest_neighbours = distances[:self.k]
            print(k_nearest_neighbours)

            class_frequency = {}
            for neighbour in k_nearest_neighbours:
                if (neighbour[1] in class_frequency):
                    class_frequency[neighbour[2]] += 1
                else:
                    class_frequency[neighbour[2]] = 1
            
            class_predictions.append(max(class_frequency, key=class_frequency.get))
        
        return class_predictions
        ### end of method ###


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

# kNN_model = KNN(data)
# predictions = kNN_model.predict(data_to_predict)
# print(predictions)