# Project Environment Setup

## Main Environment
The version requirements are not strict. If any issues arise during execution, you can change the package versions. The `requirements.txt` file contains more detailed version dependency information. You can use `conda install` to quickly set up the environment (though it might not always work).

- **Python**: 3.11.7
- **PyTorch**: 2.2.1
- **NumPy**: 1.24.3
- **scikit-learn**: 1.1.3
- **Matplotlib**: 3.8.0

## Framework
![img_3.png](img_3.png)
## Parameter Declarations
- `patch_size`: Size of the pixel neighborhood block.
- `class_num`: Number of classes in the dataset.
- `pca_components`: Number of PCA components after dimensionality reduction, which is also the input dimension for the network.
- `test_ratio`: Proportion of test data per class; `1 - test_ratio` is the proportion of training data.

## Comparison Methods
Comparison methods are stored in the `compare_method` folder. The training methods are the same as those used in the main program.
## Hyperparameter Settings
In the **Hyperparameter Settings** section, you can modify the training parameters.

## Dataset Selection
In the **Dataset selection** section, you can change the dataset.

## Important Notes
- The `loss`, `model`, `net`, and `utils` folders are source code directories. If using PyCharm, right-click on these folders → **Mark Directory as** → **Sources Root**. This prevents issues where Python files cannot be read.
- The `main.py` file is the main program used for pre-training and saving models.
- Use `finetune.py` for fine-tuning training. The results are saved in the `results` folder.
- If you encounter GPU memory issues, reduce the `batch_size`.

## Directory Structure