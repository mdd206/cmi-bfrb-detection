import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

train = pd.read_csv("data/raw/train.csv",
                    usecols=["sequence_id", "subject", "gesture"])
seq = train.drop_duplicates("sequence_id").reset_index(drop=True)

# 1. Hold out 20% of subjects as the test set (one of 5 group folds)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(seq, y=seq["gesture"], groups=seq["subject"]))
seq["split"] = "train"
seq.loc[test_idx, "split"] = "test"

# 2. Split the remaining 80% into 5 folds for validation (test rows keep fold = -1)
seq["fold"] = -1
dev = seq[seq["split"] == "train"]
splits = sgkf.split(dev, y=dev["gesture"], groups=dev["subject"])
for fold, (_, val_idx) in enumerate(splits):
    seq.loc[dev.index[val_idx], "fold"] = fold

seq.to_csv("folds/folds.csv", index=False)
print("Saved folds/folds.csv:", seq.shape)
