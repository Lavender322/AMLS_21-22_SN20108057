import pandas as pd

# Create Task A label
df = pd.read_csv("/dataset/label.csv")
df.loc[df["label"]=="meningioma_tumor", "label"] = "tumor"
df.loc[df["label"]=="glioma_tumor", "label"] = "tumor"
df.loc[df["label"]=="pituitary_tumor", "label"] = "tumor"
df.to_csv("/dataset/label_task_A.csv", index=False)