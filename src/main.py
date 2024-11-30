from sklearn.model_selection import train_test_split
import pandas as pd
from pipelines import create_preprocessing_pipeline, handleMissingValue

df = pd.read_csv("data/merged_train_data.csv")
df = df.rename(columns={"attack_cat ": "attack_cat"})

metadata = pd.read_csv("data/metadata.csv")
metadata = metadata.rename(columns={"Type ": "Type"})

metadata["Type"] = metadata["Type"].str.lower()
metadata[["Name", "Type"]]

# Drop column with name = label in metadata
if "label" in metadata["Name"].values:
    metadata = metadata[metadata["Name"] != "label"]

numeric_columns = metadata.loc[
    (metadata["Type"] == "float") | (metadata["Type"] == "integer")
]["Name"]
categorical_columns = metadata.loc[
    (metadata["Type"] != "float") & (metadata["Type"] != "integer")
]["Name"]


df = df.drop(columns=["label"])
train_set, val_set = train_test_split(
    df, test_size=0.33, random_state=42, stratify=df["attack_cat"]
)


preprocessing_pipeline = create_preprocessing_pipeline(
    train_set, numeric_columns, categorical_columns
)

# Save the preprocessing pipeline as csv
pd.DataFrame(preprocessing_pipeline).to_csv(
    "data/preprocessing_pipeline.csv", index=False
)
