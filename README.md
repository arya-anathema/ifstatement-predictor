# ifstatement-predictor
The script loads a pretrained CodeT5 transformer model, which is designed to be able to interpret and generate code. It then fine-tunes the model with the specific objective of predicting if statements for python methods. Three corpora `test_output.csv` , `train_output.csv` and `valid_output.csv` of preprocessed Python methods, created by the preprocessing.py script, are given to the model to facilitate the fine-tuning process. After fine-tuning, several evaluation metrics are calcuated to assess model utility, including BLEU-4 score, exact if statement matches, and F1 score. Evaluation metrics are stored in the `testset-results.csv` file.

WARNING: attempting to run the preprocessing.py script will not work on non-windows systems. Please use the already processed output files.

# Installation:
1. Install [python 3.9+](https://www.python.org/downloads/) locally
2. Clone the repository to your workspace:  
```shell
~ $ git clone https://github.com/arya-anathema/ifstatement-predictor.git
```
3. Navigate into the repository:
```shell
~ $ cd ifstatement-predictor
~/ifstatement-predictor $
```
4. Set up a virtual environment and activate it:
```shell
~/ngram-recommender $ python -m venv ./venv/
```
- For macOS/Linux:
```shell 
~/ifstatement-predictor $ source venv/bin/activate
(venv) ~/ifstatement-predictor $ 
```
- For Windows:
```shell
~\ifstatement-predictor $ .\venv\Scripts\activate.bat
(venv) ~\ifstatement-predictor $ 
```

5. To install the required packages: 
```shell
(venv) ~/ifstatement-predictor $ pip install -r requirements.txt
```
# Running the Program
1. Generate new JSON files based on the contents of the folder `output_csv`:
```shell
python ifstatement-predictor.py
```
2. By default, `ifstatement-predictor.py` will log to Weights & Biases. The documentation for setup can be found [here](https://docs.wandb.ai/quickstart/). You must login using your own account before logging to wandb. 
    - If you want to disable wandb logging, comment out lines 68 and 69.

3. Our best-performing model can be downloaded [here](https://wmedu-my.sharepoint.com/:u:/g/personal/wjsanders_wm_edu/EYHBxWgIfi5Etl3fJwrgV3ABd3Ao5pvcjqeG9rKTpevjng?e=KIVIjE). This link expires on May 5th, 2025, but the same model can be generated with the data provided in the repo.

# Report

The assignment report is available in the file `Writeup.md`.
