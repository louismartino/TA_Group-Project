import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import spacy
import os
import warnings
warnings.filterwarnings("ignore")


class FeatureEngineering:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.target_column = "state"
        self.features_scale = ["duration", "CCI_index"]
        self.features_no_scale = []

        print("Initialization complete. Ready to perform feature engineering.\n")


    def train_val_test_split(self, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(len(self.df))
        train_end = int(train_size * len(self.df))
        val_end = train_end + int(val_size * len(self.df))

        train_indices = shuffled_indices[:train_end]
        val_indices = shuffled_indices[train_end:val_end]
        test_indices = shuffled_indices[val_end:]

        self.df_train = self.df.iloc[train_indices].reset_index(drop=True)
        self.df_val = self.df.iloc[val_indices].reset_index(drop=True)
        self.df_test = self.df.iloc[test_indices].reset_index(drop=True)

        print("\nData split into train, validation and test sets.\n")


    def target(self):
        self.df["target"] = self.df[self.target_column].apply(lambda x: 1 if x == "successful" else 0)

        print(f"Target variable '{self.target_column}' transformed into binary 'target' column.")


    def dummies(self):
        self.df = pd.get_dummies(self.df, columns=["category.parent_name"], prefix="cat", drop_first=False, dtype=int)
        self.features_no_scale.extend([col for col in self.df.columns if col.startswith("cat_")])

        self.df = pd.get_dummies(self.df, columns=["country"], prefix="country", drop_first=False, dtype=int)
        self.features_no_scale.extend([col for col in self.df.columns if col.startswith("country_")])

        print("Categorical variables 'category.parent_name' and 'country' transformed into dummy variables.")


    def words_number(self):
        self.df["blurb_length"] = self.df["blurb"].apply(lambda x: len(str(x).split()))
        self.features_scale.append("blurb_length")

        print("Feature 'blurb_length' created.")


    def sentiment_analysis(self):
        """
        score between +0.05 and +1.0: positive sentiment
        score between -0.05 and +0.05: neutral sentiment
        score between -1.0 and -0.05: negative sentiment
        """
        analyzer = SentimentIntensityAnalyzer()
        self.df["sentiment_score"] = self.df["blurb"].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
        self.features_scale.append("sentiment_score")

        print("Feature 'sentiment_score' created using VADER sentiment analysis.")


    def vocab_level(self):
        self.df["readability_score"] = self.df["blurb"].apply(lambda x: textstat.flesch_reading_ease(str(x)))
        self.features_scale.append("readability_score")

        print("Feature 'readability_score' created.")


    def name_blurb_similarity(self):
        nlp = spacy.load("en_core_web_md")
        names = self.df['name'].astype(str).tolist()
        blurbs = self.df['blurb'].astype(str).tolist()

        similarities = []
        for doc_name, doc_blurb in zip(nlp.pipe(names), nlp.pipe(blurbs)):
            similarities.append(doc_name.similarity(doc_blurb))

        self.df['name_blurb_similarity'] = similarities
        self.features_scale.append("name_blurb_similarity")

        print("Feature 'name_blurb_similarity' created.")
    

    def goal(self):
        self.df["log_goal"] = np.log1p(self.df["goal_usd"])
        self.features_scale.append("log_goal")

        self.df["CCI_per_goal"] = self.df["CCI_index"] / self.df["goal_usd"]
        self.features_scale.append("CCI_per_goal")

        print("Features 'log_goal' and 'CCI_per_goal' created.")


    def z_score_log_goal(self):
        stats = self.df_train.groupby("category.name")["log_goal"].agg(["mean", "std"]).reset_index()

        mean_dict = stats.set_index("category.name")["mean"].to_dict()
        std_dict = stats.set_index("category.name")["std"].to_dict()

        def apply_zscore(target_df):
            m = target_df["category.name"].map(mean_dict)
            s = target_df["category.name"].map(std_dict).replace(0, 1).fillna(1)
            return (target_df["log_goal"] - m) / s

        self.df_train["z-score_log_goal"] = apply_zscore(self.df_train)
        self.df_val["z-score_log_goal"] = apply_zscore(self.df_val)
        self.df_test["z-score_log_goal"] = apply_zscore(self.df_test)
        self.features_no_scale.append("z-score_log_goal")

        print("Feature 'z-score_log_goal' created.")


    def save(self):
        path = "../data/features/"
        os.makedirs(path, exist_ok=True)

        self.df_train.to_csv(f"{path}train.csv", index=False)
        self.df_val.to_csv(f"{path}val.csv", index=False)
        self.df_test.to_csv(f"{path}test.csv", index=False)

        with open(f"{path}features_scale.txt", "w") as f:
            for feature in self.features_scale:
                f.write(feature + "\n")

        with open(f"{path}features_no_scale.txt", "w") as f:
            for feature in self.features_no_scale:
                f.write(feature + "\n")

        print(f"\nProcessed dataframes saved to '{path}' directory.")


    def feature_engineering(self):

        # Apply all feature engineering steps (not depending on train-val-test split)
        self.target()
        self.dummies()
        self.words_number()
        self.sentiment_analysis()
        self.vocab_level()
        self.name_blurb_similarity()
        self.goal()
        
        # Split the data into train-val-test sets
        self.train_val_test_split()

        # Apply all feature engineering steps (DEPENDING on train-val-test split)
        self.z_score_log_goal()

        # Save the processed dataframes
        self.save()


if __name__ == "__main__":
    df = pd.read_csv("../data/dataset.csv")
    fe = FeatureEngineering(df)
    fe.feature_engineering()