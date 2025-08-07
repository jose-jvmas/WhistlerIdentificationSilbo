import argparse

from sklearn.svm import SVC
from datasets import Dataset as HF_Dataset
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

from train_classifier import train_classifier


CLS_METHODS = {
    GaussianMixture: {
        "covariance_type": ['spherical', 'tied'] 
    },
    KNeighborsClassifier: {
        "n_neighbors": [1, 3, 5]
    },
    SVC: {
        "kernel": ["rbf", "poly"]
    },
    RandomForestClassifier: {
        "n_estimators": [100, 500],
    },
    MLPClassifier: {
        "solver": ["lbfgs", "sgd"],
    },
}
IMB_METHODS = [None, SMOTE, BorderlineSMOTE, ADASYN]


def perform_experiments(in_dataset: HF_Dataset, args: argparse.Namespace, res_file: str):
    """Perform experiments for each classifier and imbalance method"""
    res_dict = {"encoder": args.encoder, "encoder_conf": args.param}

    for cls, cls_data in CLS_METHODS.items():
        res_dict["cls"] = cls.__name__

        for param, conf_values in cls_data.items():
            for single_conf_value in conf_values:
                res_dict["param"] = single_conf_value

                # Get classifier
                cls_config = {param: single_conf_value}
                if cls == KNeighborsClassifier:
                    pass
                elif cls == MLPClassifier:
                    cls_config.update(
                        {
                            "alpha": 0.0001,
                            "hidden_layer_sizes": 32,
                            "batch_size": 16,
                            "max_iter": 1000,
                            "random_state": 42,
                        }
                    )
                elif cls == SVC:
                    cls_config.update({
                        "max_iter": 1000,
                        "random_state": 42,
                    })
                elif cls == GaussianMixture:
                    cls_config.update(
                        {
                            "n_components": len(set(in_dataset['label'])),
                        }
                    )
                else:
                    cls_config["random_state"] = 42
                classifier = cls(**cls_config)

                # Train classifier for each imbalance method
                for imb_method_class in IMB_METHODS:
                    res_dict["ImbMethod"] = imb_method_class.__name__ if imb_method_class is not None else "-"
                    train_classifier(
                        dataset=in_dataset,
                        cls=classifier,
                        res_dict=res_dict,
                        imb_method_class=imb_method_class,
                        dst_file=res_file,
                    )
