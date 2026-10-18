import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Data preprocessing ----------------
df = pd.read_csv("data/churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print(df["TotalCharges"].dtype)
print(df["TotalCharges"].isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print(df["Churn"].value_counts())

# ---------------- EDA charts ----------------
print(df.groupby("Contract")["Churn"].value_counts(normalize=True))

plt.figure()
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by contract type")
plt.savefig("reports/churn_by_contract.png")

plt.figure()
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Tenure by churn")
plt.savefig("reports/tenure_by_churn.png") 