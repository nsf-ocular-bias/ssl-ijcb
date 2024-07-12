import pickle 
import numpy as np

from sklearn.linear_model import LogisticRegression

import pandas as pd
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, equalized_odds_ratio, selection_rate
# from fairlearn.widget import FairlearnDashboard

def evaluate_model(data_test, data_train, df_train, df_test, attrs):
    data_train = data_train.cpu().numpy()
    data_test = data_test.cpu().numpy()
    
    X_test = data_test

    N_SAMPLES = 5000

    scores = {}

    sensitive = (df_test['Male'].values + 1) / 2

    for attr in attrs:
        if "Attractive" not in attr:
            continue
        # if attr != "Male":
        #     continue

        y_train = df_train[attr].values
        y_test = df_test[attr].values

        y_train = (y_train + 1) / 2
        y_test = (y_test + 1) / 2


        # Sample N_SAMPLES random samples
        indices = np.random.choice(data_train.shape[0], N_SAMPLES, replace=False)
        X_train = data_train[indices]
        y_train = y_train[indices]

        clf = LogisticRegression(max_iter=10000)
        clf.fit(X_train, y_train)

        acc = clf.score(X_test, y_test) * 100

        preds = clf.predict(X_test)

        # Demographic Parity
        dp_diff = demographic_parity_difference(y_test, preds, sensitive_features=sensitive)
        dp_ratio = demographic_parity_ratio(y_test, preds, sensitive_features=sensitive)

        # Equalized Odds
        eo_diff = equalized_odds_difference(y_test, preds, sensitive_features=sensitive)
        eo_ratio = equalized_odds_ratio(y_test, preds, sensitive_features=sensitive)

        # Selection Rate
        sr = selection_rate(y_test, preds, pos_label=1)

        print(f"Attribute: {attr}, Accuracy: {acc}")

        scores[f"{attr}_acc"] = acc

        scores[f"{attr}_dp_diff"] = dp_diff
        scores[f"{attr}_dp_ratio"] = dp_ratio

        scores[f"{attr}_eo_diff"] = eo_diff
        scores[f"{attr}_eo_ratio"] = eo_ratio

        scores[f"{attr}_sr"] = sr


    return scores




def main():
    df = pd.read_csv("../datasets/celeba/list_attr_celeba.csv")
    partition = pd.read_csv("../datasets/celeba/list_eval_partition.csv")

    df_train = df[partition["partition"] == 0]
    df_val = df[partition["partition"] == 1]
    df_test = df[partition["partition"] == 2]

    print(df_train.shape)
    print(df_val.shape)
    print(df_test.shape)

    attrs = df.columns[1:].tolist()

    with open("feats_simclr.pkl", "rb") as f:
        data_test = pickle.load(f)

    with open("feats_simclr_train.pkl", "rb") as f:
        data_train = pickle.load(f)


    simclr = evaluate_model(data_test, data_train, df_train, df_test, attrs)

    with open("feats_vicreg.pkl", "rb") as f:
        data_test = pickle.load(f)

    with open("feats_vicreg_train.pkl", "rb") as f:
        data_train = pickle.load(f)

    vicreg = evaluate_model(data_test, data_train, df_train, df_test, attrs)

    with open("feats_byol.pkl", "rb") as f:
        data_test = pickle.load(f)

    with open("feats_byol_train.pkl", "rb") as f:
        data_train = pickle.load(f)

    byol = evaluate_model(data_test, data_train, df_train, df_test, attrs)

    with open("feats_barlow.pkl", "rb") as f:
        data_test = pickle.load(f)

    with open("feats_barlow_train.pkl", "rb") as f:
        data_train = pickle.load(f)

    barlow = evaluate_model(data_test, data_train, df_train, df_test, attrs)

    with open("feats_107.pkl", "rb") as f:
        data_test = pickle.load(f)

    with open("feats_107_train.pkl", "rb") as f:
        data_train = pickle.load(f)

    supcon = evaluate_model(data_test, data_train, df_train, df_test, attrs)

    with open("feats_129.pkl", "rb") as f:
        data_test = pickle.load(f)

    with open("feats_129_train.pkl", "rb") as f:
        data_train = pickle.load(f)

    supcon2 = evaluate_model(data_test, data_train, df_train, df_test, attrs)

    # Create a DataFrame
    df = pd.DataFrame([simclr, vicreg, byol, barlow, supcon, supcon2])

    df.to_csv("results.csv")

    print(df)

if __name__ == "__main__":
    main()
