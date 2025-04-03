# Assignment 2 Writeup

Kayla Anderson - Whitt Sanders - Arya Ray-Shryock

### Dataset Preparation
To prepare the dataset for finetuning, we created a preprocessing file. Our file takes a csv as input, strips all newlines from the cleaned_method column, and replaces the text representation of the tab spacing in the methods with a \<tab\> token to preserve python indentation. Since the values in the target_block column were already tokenized, we could not simply run a replace function on the cleaned_method column. To handle this, we created a list of all the values in the target_block column and stripped them of spaces. Then, we used a regex function to search each method for the target agnostic of spaces, and replaced the first instance of the target block with a \<IF-STMT\> token. After these processing steps, we saved the output in a new csv.

### Fine-tuning

### Evalutation Results
