# Writing style of Generative AI

The goal of the project is to analyze the large language models' writing style novelty. For that we created our own dataset, called Authors' Writing Style, that consists of human authors books and corresponding generated stories. In our work we use different methods like PCA, Classification or Entropy Analysis to explore the tendencies of human authors and the models. The process is divided into steps, which are covered by the notebooks from the root directory. More about them are explained later. For more detailed information feel free to check the [Thesis Paper](thesis.pdf).

## Setup

The setup is only possible with an access to `curie.compute.dtu.dk` server. The first step is to copy the datasets to `res` directory. Corresponding files can be found at `/home/s222914/GenAIStyle/res_v3` in `curie.compute.dtu.dk` directory. Make sure that python is installed (tested on version 3.12.1) and create virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
```

After that install required python packages:

```sh
pip install -r requirements.txt
```

Run the notebooks in the indicated order and follow their further setup instructions. 

## Notebooks

- **0_words_distribution.ipynb** loads all the datasets and calculates global word distribution.

- **1_writing_style_books_info.ipynb** explores the dataset of books from [Guthenberg](https://aclanthology.org/E14-3011/) dataset and selects ten authors, which are analyzed.

- **2_generator.ipynb** generates short stories in the writing style of ten, previously chosen authors using different language models.

- **3_writing_style_dataset.ipynb** performs PCA and Entropy Analysis on Authors' Writing Style dataset.

- **4_writing_style_pca_classification.ipynb** runs various classification experiments for PCA feature set on Authors' Writing Style dataset.

- **5_writing_style_all_features_classification.ipynb** runs various classification experiments for full feature set on Authors' Writing Style dataset.

- **6_daigt_dataset.ipynb** runs performs PCA on DAIGT-V4 dataset.

- **7_daigt_pca_classification.ipynb** runs various classification experiments for PCA feature set on DAIGT-V4 dataset.

- **8_daigt_all_features_classification.ipynb** runs various classification experiments for full feature set on DAIGT-V4 dataset.