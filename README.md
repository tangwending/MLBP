# MLBP
Identifying bioactive peptide function using multi-label deep learning


## Introduction
Motivation: 

The bioactive peptide has wide functions, such as lowering blood glucose levels and reducing inflammation. Meanwhile, computational methods such as machine learning are becoming more and more important for peptide functions prediction. Most of the previous studies concentrate on the single-functional bioactive peptides prediction. However, the number of multi-functional peptides is on the increase, therefore novel computational methods are needed.

Results: 

In this study, we develop a method MLBP (Multi-Label deep learning approach for determining the multi-functionalities of Bioactive Peptides), which can predict multiple functions including anti-cancer, anti-diabetic, anti-hypertensive, anti-inflammatory and anti-microbial simultaneously. MLBP model takes the peptide sequence vector as input to replace the biological and physiochemical features used in other peptides predictors. Using the embedding layer, the dense continuous feature vector is learnt from the sequence vector. Then, we extract convolution features from the feature vector through the convolutional neural network layer, and combine with the bidirectional gated recurrent unit layer to improve the prediction performance. The 5-fold cross-validation experiments are conducted on the training dataset, and the results show that Accuracy and Absolute true are 0.695 and 0.685, respectively. On the test dataset, Accuracy and Absolute true of MLBP are 0.709 and 0.697, with 5.0% and 4.7% higher than those of the suboptimum method, respectively. The results indicate MLBP has superior prediction performance on the multi-functional peptides identification.

Note:

For the peptides with length ≤ 45, the prediction results of MLBP are accurate. If the peptides with length > 45, the prediction results for ACP, ADP, AHP, AIP and AMP are just for reference only

![draft](./figures/framework.jpg)


## Related Files

#### MLBP

| FILE NAME           | DESCRIPTION                                                  |
| :------------------ | :----------------------------------------------------------- |
| main.py             | the main file of MLBP predictor (include data reading, encoding, and data partitioning) |
| train.py            | train model |
| model.py            | model construction |
| test.py             | test model result |
| evaluation.py       | evaluation metrics (for evaluating prediction results) |
| data                | data         |
| BiGRU_base          | models of MLBP           |


## Installation
- Requirement
  
  OS：
  
  - `Windows` ：Windows7 or later
  
  - `Linux`：Ubuntu 16.04 LTS or later
  
  Python：
  
  - `Python` >= 3.6
  
- Download `MLBP`to your computer

  ```bash
  git clone https://github.com/xialab-ahu/MLBP.git
  ```

- open the dir and install `requirement.txt` with `pip`

  ```
  cd MLBP
  pip install -r requirement.txt
  ```


## Run MLBP on a new test fasta file
```shell
python predictor.py --file test.fasta --out_path result
```

- `--file` : input the test file with fasta format

- `--out_path`: the output path of the predicted results


## Contact
Please feel free to contact us if you need any help.

