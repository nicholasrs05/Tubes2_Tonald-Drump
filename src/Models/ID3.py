import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional,List
from sklearn.base import BaseEstimator, ClassifierMixin


class ID3DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.tree: Optional[Dict[str, Any]] = None
        self.label: str = ""

    def calculate_entropy(self, data: pd.DataFrame) -> float:
        _, counts = np.unique(data.iloc[:, -1], return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))


    def calculate_information_gain(self, data: pd.DataFrame, feature: str) -> float:
        total_entropy = self.calculate_entropy(data)
        unique_values = data[feature].unique()
        weighted_entropy = 0

        for value in unique_values:
            subset = data[data[feature] == value]
            weighted_entropy += (len(subset) / len(data)) * self.calculate_entropy(subset)

        return total_entropy - weighted_entropy


    def find_best_splits(self, data: pd.DataFrame, feature: str, max_splits: int) -> List[float]:
        sorted_data = data.sort_values(feature)
        unique_values = sorted_data[feature].unique()

        print(f"Unique value length of column '{feature}' is {len(unique_values)}")

        if len(unique_values) <= 1:
            return []

        max_splits = min(max_splits, len(unique_values) - 1)

        split_points = []
        last_split = None
        for i in range(1, max_splits + 1):
            quantile = i / (max_splits + 1)
            split_value = np.quantile(unique_values, quantile)

            if last_split is None or not np.isclose(split_value, last_split):
                split_points.append(split_value)
                last_split = split_value

        split_points = sorted(set(split_points))

        if len(split_points) < 1:
            return []

        return split_points



    def calculate_information_gain_on_split(self, data: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> float:
        total_entropy = self.calculate_entropy(data)
        left_weight = len(left) / len(data)
        right_weight = len(right) / len(data)
        return total_entropy - (left_weight * self.calculate_entropy(left) + right_weight * self.calculate_entropy(right))

    def convert_numeric_to_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in data.columns[:-1]:
            if pd.api.types.is_numeric_dtype(data[column]):
                print(f"Processing numeric column: {column}")

                unique_values = data[column].unique()
                if len(unique_values) < 20:
                    max_split = len(unique_values)
                else:
                    max_split = int(len(data) / 100)
                min_val = data[column].min()
                max_val = data[column].max()

                split_points = self.find_best_splits(data, column, max_splits=max_split)
                if split_points:
                    bins = [min_val] + split_points + [max_val]
                    bins = sorted(set(bins))
                    labels = [f'<= {split_points[0]}'] + \
                            [f'{split_points[i]} - {split_points[i + 1]}' for i in range(len(split_points) - 1)] + \
                            [f'> {split_points[-1]}']

                    data[column] = pd.cut(data[column], bins=bins, labels=labels, include_lowest=True)

        return data


    def best_feature_to_split(self, data: pd.DataFrame) -> str:
        gains = {feature: self.calculate_information_gain(data, feature) for feature in data.columns[:-1]}
        return max(gains, key=gains.get)

    def build_tree(self, data: pd.DataFrame) -> Dict[str, Any]:
        target = data.columns[-1]

        if len(data[target].unique()) == 1:
            return data[target].iloc[0]

        if len(data.columns) == 1:
            return data[target].mode().iloc[0]

        best_feature = self.best_feature_to_split(data)
        print(f"Feature {best_feature} processed")

        tree = {best_feature: {}}

        majority_class = data[target].mode().iloc[0]

        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value].drop(columns=[best_feature])
            tree[best_feature][value] = self.build_tree(subset)

        tree[best_feature]["unknown"] = majority_class

        return tree

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        self.label = 'attack_cat'

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            print(f"Converted X to pandas DataFrame. Type of X: {type(X)}; Shape of X: {X.shape}")

        if len(X) != len(y):
            raise ValueError("The number of rows in X and the number of elements in y must be the same.")

        data = X.copy()
        print(f"Target column 'attack_cat':\n{y.head()}")

        try:
            data[self.label] = y.values
            print(f"Data after adding target column:\n{data.head()}")
        except Exception as e:
            print(f"Error occurred while adding target column: {e}")

        processed_data = self.convert_numeric_to_categorical(data)

        print("Building tree")

        columns = [col for col in processed_data.columns if col != self.label]
        columns.append(self.label)

        processed_data = processed_data[columns]
        print(f"Columns after reordering: {processed_data.columns}")

        self.tree = self.build_tree(processed_data)


    def predict_sample(self, tree: dict, sample: pd.Series) -> Any:
        while isinstance(tree, dict):
            feature = next(iter(tree))
            value = sample.get(feature)

            if isinstance(value, pd.Series):
                value = value.iloc[0]

            if pd.isna(value):
                return "default_class"

            branches = tree[feature]

            found_match = False
            for key in branches:
                if isinstance(key, str) and ' - ' in key:
                    lower_bound, upper_bound = key.split(' - ')
                    lower_bound = float(lower_bound)
                    upper_bound = float(upper_bound)

                    if isinstance(value, (int, float)):
                        if lower_bound <= value <= upper_bound:
                            tree = branches[key]
                            found_match = True
                            break
                    else:
                        continue

                elif isinstance(key, str) and any(op in key for op in ['<=', '>=', '<', '>', '==', '!=']):
                    operator, threshold = key.split(' ', 1)
                    threshold = float(threshold) if operator not in ['==', '!='] else threshold

                    if isinstance(value, (int, float)):
                        if operator == '<=':
                            if value <= threshold:
                                tree = branches[key]
                                found_match = True
                                break
                        elif operator == '<':
                            if value < threshold:
                                tree = branches[key]
                                found_match = True
                                break
                        elif operator == '>=':
                            if value >= threshold:
                                tree = branches[key]
                                found_match = True
                                break
                        elif operator == '>':
                            if value > threshold:
                                tree = branches[key]
                                found_match = True
                                break

                        elif operator == '==':
                            if value == threshold:
                                tree = branches[key]
                                found_match = True
                                break
                        elif operator == '!=':
                            if value != threshold:
                                tree = branches[key]
                                found_match = True
                                break
                    else:
                        continue

                elif isinstance(value, (str, int, float)):
                    if value == key:
                        tree = branches[key]
                        found_match = True
                        break
                else:

                    continue

            if not found_match:
                if "unknown" in branches:
                    # return "Normal"
                    return branches["unknown"]
                else:
                    return "default_class"

        return tree



    def predict(self, df: pd.DataFrame) -> pd.Series:
      # Ensure df is a pandas DataFrame
      if isinstance(df, np.ndarray):
          df = pd.DataFrame(df)
          print(f"Converted df to pandas DataFrame. Type of df: {type(df)}; Shape of df: {df.shape}")

      # Apply the prediction function row-wise
      predictions = df.apply(lambda row: self.predict_sample(self.tree, row), axis=1)

      return predictions


    def save_model(self, file_path: str) -> None:
        if self.tree is None:
            raise ValueError("Model is not trained yet, cannot save.")
        with open(file_path, 'w') as f:
            json.dump(self.tree, f, indent=4)

    def load_model(self, file_path: str) -> None:
        with open(file_path, 'r') as f:
            self.tree = json.load(f)

# if __name__ == "__main__":
#     df = pd.DataFrame({
#         'Temperature': [85, 80, 83, 70, 68, 65, 72, 75, 77, 81, 71, 73, 78, 69],
#         'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 60, 65, 75, 80, 70, 80],
#         'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'True'],
#         'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
#     })

#     tree_model = ID3DecisionTree()
#     tree_model.train_model(df, label='Play')
#     print(f"Trained Tree:\n")
#     print_tree(tree_model.tree)

#     tree_model.save_model("decision_tree.json")

#     new_tree_model = ID3DecisionTree()
#     new_tree_model.load_model("decision_tree.json")
#     print(f"\nLoaded Tree:\n")
#     print_tree(new_tree_model.tree)