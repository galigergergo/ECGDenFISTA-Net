# Sparse dictionary learning neural networks for ECG signal denoising

This is the code base for the paper **Sparse dictionary learning neural networks for ECG signal denoising**.

## K-SVD dictionary learning

To learn a new dictionary with K-SVD, run `KSVD_dictionary-learning.ipynb`.

The dictionary used for sparse approximation in the paper was saved in the `data/dictionary_BW_real_data.npy` file.

## FISTA-Net sparse approximation

Before running the FISTA-Net training notebooks, start an MLflow Tracking server with the following command:
```shell
mlflow server --host 127.0.0.1 --port 8080
```
The tracking server can now be accessed from any browser at `http://localhost:8080`. If MLflow is not installed, simply install it with `pip install mlflow`.

To retrain the FISTA-Net model from the paper:
1. first, run `FISTA-Net_pre-training.ipynb` to pre-train the model for 3000 epochs;
2. then, run `FISTA-Net_transfer-learning` to fine-tune the model for the remaining 17000 epochs.

NOTE: Don't forget to stop the kernel after pre-training before starting the transfer learning training process when training on GPU to free up resources.

The pre-trained model used in the paper can be found in `models/FISTA-Net_ep20000`.
