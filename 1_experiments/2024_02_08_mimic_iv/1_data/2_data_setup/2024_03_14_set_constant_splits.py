import numpy
import random
import pandas as pd




if __name__ == '__main__':
    # Set random seed
    random.seed(0)
    numpy.random.seed(0)

    train_val_test_split = [0.8, 0.1, 0.1]

    constants_path = "/home/makaron1/uc2_nsclc/2_experiments/2024_02_08_mimic_iv/1_data/0_final_data/constants.csv"

    # Load constants
    constants = pd.read_csv(constants_path)


    # Split data
    n = len(constants)
    indices = list(range(n))
    random.shuffle(indices)

    train_indices = indices[:int(n * train_val_test_split[0])]
    val_indices = indices[int(n * train_val_test_split[0]):int(n * (train_val_test_split[0] + train_val_test_split[1]))]
    test_indices = indices[int(n * (train_val_test_split[0] + train_val_test_split[1])):]

    # assert that they do not overlap
    assert len(set(train_indices).intersection(set(val_indices))) == 0
    assert len(set(train_indices).intersection(set(test_indices))) == 0
    assert len(set(val_indices).intersection(set(test_indices))) == 0

    # assert that they make the full set
    assert len(set(train_indices).union(set(val_indices)).union(set(test_indices))) == n

    # Save splits
    constants.loc[train_indices, "dataset_split"] = "TRAIN"
    constants.loc[val_indices, "dataset_split"] = "VALIDATION"
    constants.loc[test_indices, "dataset_split"] = "TEST"

    # Save constants
    constants.to_csv(constants_path, index=False)

    



