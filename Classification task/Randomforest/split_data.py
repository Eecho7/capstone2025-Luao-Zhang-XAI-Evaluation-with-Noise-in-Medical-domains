import pandas as pd

df = pd.read_csv("/Users/luao/Desktop/XAI PROJECT/rf/Training.csv")
X = df.drop(columns=["prognosis"])
y = df[["prognosis"]]

X.to_csv("X_train.csv", index=False)
y.to_csv("y_train.csv", index=False)

print(" Copied data to X_train.csv and y_train.csv")