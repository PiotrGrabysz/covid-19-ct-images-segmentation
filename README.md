# COVID-19 CT Image Segmentation

This project applies deep learning techniques to segment radiological features in CT scans of patients with COVID-19.
The model distinguishes between ground-glass opacities and consolidations in axial lung slices. 

This project is a final project in the Udacity AWS Machine Learning Engineer Nanodegree course, which I participated in.
The subject one the project was chosen by myself, and it was developed as part of a 
[Kaggle competition](https://www.kaggle.com/competitions/covid-segmentation/overview).

**The reports required by the Udacity course are in [reports/](reports).**

## ⚡ Technology Stack

- **[PyTorch](https://pytorch.org/)**: Deep learning framework used to implement and train the model.
- **[Lightning](https://lightning.ai/)**: High-level interface for PyTorch, enabling cleaner code.
- **[segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)**: 
Provides a flexible and powerful U-Net implementation with pretrained backbones (EfficientNet used in this project).
- **[Albumentations](https://albumentations.ai/)**: Fast and flexible library for image augmentation.
- **[Typer](https://typer.tiangolo.com/)**: Tool for building great and easy CLIs.

Other dependencies:
- `numpy`: For loading data.
- `matplotlib`: For data visualization.
- `boto3` and `sagemaker`: Used to interface with AWS SageMaker and S3.

## 📦 Infrastructure and Environment

I used AWS SageMaker AI with PyTorch estimator in the script mode to train the model.
Because the Docker image managed by AWS doesn't contain the other libraries I need (like `ligthing`), 
I created my custom Docker image and pushed it to a private ECR (Elastic Container Registry).
The instructions on how to do it are [here](docker/README.md).

## 🚀 Setup and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/covid19-ct-segmentation.git
cd covid19-ct-segmentation
```

### 2. Get the competition data

You can use Kaggle CLI to download data. Firstly you need to get your Kaggle API key and save it as `kaggle.json`.
Then move it to `/home` directory,so that you don't store it in your project.

```bash
mkdir -p /home/sagemaker-user/.config/kaggle
mv kaggle.json /home/sagemaker-user/.config/kaggle/kaggle.json
chmod 600 /home/sagemaker-user/.config/kaggle/kaggle.json
```

Install the Kaggle CLI:
```bash
pip install kaggle
```

Fetch the competition data:

```bash
kaggle competitions download -c covid-segmentation
```

Extract data from the archive:

```bash
mkdir covid-segmentation-data
unzip covid-segmentation.zip -d covid-segmentation-data && rm covid-segmentation.zip
```

### 3. Run notebooks

I run all the code with SageMaker AI Studio. I used `ml.t3.large` instance with `Python 3` kernel.

- The Exploratory Data Analysis is in [notebooks/Exploratory data analysis.ipynb](notebooks/Exploratory%20data%20analysis.ipynb).
- Data processing and uploading to S3 is in [notebooks/Prepare data.ipynb](notebooks/Prepare%20data.ipynb).
- Model training is in [notebooks/Train.ipynb](notebooks/Train.ipynb).
- [notebooks/Deploy_and_run_inference.ipynb](notebooks/Deploy_and_run_inference.ipynb) shows how to deploy an inference
endpoint, obtain test predictions and send them to Kaggle for evaluation.

The code for training and inference, used by Jupyter Notebooks, is in `src/`. 
The code is break down into small, logical components:
```
src/
├── data/
    ├── augmentation.py
    ├── data_loaders.py
    └── dataset.py
├── loss.py
├── metrics.py
├── model.py
└── train.py
```

