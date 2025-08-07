from typing import Optional
from collections import Counter

import numpy as np
from datasets import Dataset as HF_Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from utils import write_results


def train_classifier(
    dataset: HF_Dataset,
    cls: object, # Sklearn classifier
    res_dict: dict,
    dst_file: str,
    imb_method_class: Optional[object] = None, # Imbalance learn class not initialized
):
    # 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    skf.get_n_splits(dataset)

    for i, (train_index, test_index) in enumerate(skf.split(dataset["features"], dataset["label"])):
        res_dict["FOLD"] = i + 1

        # Split data
        X_train = np.array(dataset["features"])[train_index]
        y_train = np.array(dataset["label"])[train_index]
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            random_state=42,
            stratify=y_train,
        )
        X_test = np.array(dataset["features"])[test_index]
        y_test = np.array(dataset["label"])[test_index]

        # Normalization
        def min_max_normalization(X: np.ndarray):
            X_min = X.min(axis=1, keepdims=True)
            X_max = X.max(axis=1, keepdims=True)
            return (X - X_min) / (X_max - X_min)

        X_train = min_max_normalization(X_train)
        X_val = min_max_normalization(X_val)
        X_test = min_max_normalization(X_test)

        # Imbalance
        if imb_method_class is not None:
            imb_method_config = {
                "sampling_strategy": "not majority",
                "random_state": 42,
                "n_neighbors": min(Counter(y_train).values()) - 1,
            }
            if "SMOTE" in imb_method_class.__name__:
                imb_method_config["k_neighbors"] = imb_method_config["n_neighbors"]
                del imb_method_config["n_neighbors"]
            imb_method = imb_method_class(**imb_method_config)
            X_train, y_train = imb_method.fit_resample(X_train, y_train)

        # Train and test classifier
        ### Only for GMM
        if cls.__class__.__name__ == "GaussianMixture":
            cls.means_init = np.array([X_train[y_train == i].mean(axis=0) for i in range(cls.n_components)])

        cls.fit(X_train, y_train)
        y_val_pred = cls.predict(X_val)
        y_val_pred_prob = cls.predict_proba(X_val)
        y_test_pred = cls.predict(X_test)
        y_test_pred_prob = cls.predict_proba(X_test)

        # Compute metrics
        val_P, val_R, val_F1, _ = precision_recall_fscore_support(
            y_true=y_val, y_pred=y_val_pred, average="macro"
        )
        test_P, test_R, test_F1, _ = precision_recall_fscore_support(
            y_true=y_test, y_pred=y_test_pred, average="macro"
        )
        res_dict.update(
            {
                "val_F1": val_F1,
                "test_F1": test_F1,
            }
        )

        # Write results to CSV file
        write_results(dst_results_file=dst_file, res_dict=res_dict)