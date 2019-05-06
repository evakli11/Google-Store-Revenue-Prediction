# How to set up
git clone CS539_ML to preferred directory

## Requirements
Make sure that processed_train_df.csv and rnn_train.csv is in the same directory. The data can be found at https://drive.google.com/drive/folders/1CP4uT9kd24L2ACueXX4nvA7YnJDVVXc1

## Running Customer-Based Model (RNN - Weighted Classified Subnetwork for Regression)
On bash, run python customer_based_model.py in the directory

## Running Visit-based Model (Pre-classified_Regression)
1. Open a terminal to directory and run "python visit_based_model.py"
2. The program will output results from each baseline and then the results from the proposed model
3. You may edit the classifier/regression model combination by editing the sections highlighted by the comments
