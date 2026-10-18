import pandas as pd

pd.set_option("display.max_columns", None)
def load_and_prepare():
    df = pd.read_csv("data/churn.csv")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    df = pd.get_dummies(df, drop_first=True)
    return df

if __name__=="__main__":
    df=load_and_prepare()
    print(df.shape)