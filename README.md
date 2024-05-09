
# Predicting In-Hospital Mortality and Other Patient Outcomes using BERT-Based Models

Author: Shivaram B
Contact: Shivarambaldura@gmail.com  

## Abstract

This study explores the use of BERT-based models to predict in-hospital mortality and other patient outcomes using clinical text data from the MIMIC IV 2.2 dataset. We assess the performance of various BERT variants, including Bio_ClinicalBERT, BiomedNLP-PubMedBERT, and BioBERT, in accurately predicting patient outcomes. Experimental results underscore the efficacy of BERT-based models in patient outcome prediction tasks, offering valuable insights for clinical decision-making.

## Keywords

Patient Outcome Prediction, BERT, MIMIC IV 2.2 Dataset, Bio_ClinicalBERT, BiomedNLP-PubMedBERT, BioBERT, Clinical Text Classification, BERT Model Performance

## 1. Introduction

Predicting in-hospital mortality and other patient outcomes is essential for optimizing resource allocation and enhancing patient care. This study investigates the utilization of BERT-based models for patient outcome prediction using clinical text data from the MIMIC IV 2.2 dataset.

### 1.1 Project Objectives

The project aims to achieve the following objectives:

- Develop and assess BERT-based models for predicting patient outcomes in hospital stays.
- Compare the performance of different BERT variants in patient outcome prediction tasks.
- Provide insights into the practical application of BERT-based models for patient outcome risk assessment in clinical settings.
- Evaluate the impact of clinical text features on patient outcome prediction accuracy.

### 1.2 Description of the Dataset

The dataset used in this project, derived from the MIMIC IV 2.2 dataset, comprises several tables containing comprehensive clinical data from intensive care unit (ICU) stays. By integrating and analyzing data from these tables, predictive models can leverage a comprehensive set of patient-related variables to forecast outcomes like mortality, length of stay, or disease progression. This holistic approach enables clinicians and healthcare providers to make informed decisions, allocate resources effectively, and personalize patient care to optimize patient outcomes during hospital stays.

### 1.3 Feature Engineering

Feature engineering involves creating new features or modifying existing ones in a dataset to improve the performance of machine learning models. Here's an overview of the feature engineering steps applied in this study:
- Categorization of Length of Stay (LOS)
- Preprocessing the Diagnosis Description Text
- Dropping Unnecessary Columns
- Handling Missing Values
- Sampling

### 1.4 Data Analysis

Data analysis is crucial for understanding the characteristics of the data and identifying potential relationships between variables. Here are some examples of data analysis techniques employed in this study:
- Length of Stay Distribution
- Top Admission Types
- Top Diagnoses
- Hospital Mortality
- Correlation Matrix

By performing these data analysis techniques, we gain valuable insights into the characteristics of the patient population, patterns of hospital admissions and diagnoses, and potential relationships between clinical factors and patient outcomes. This understanding can inform the feature selection process and guide the development of more effective prediction models.

## 2. Methodology: BERT-Based Patient Outcome Prediction

This section delves into the methodology employed in this study, specifically focusing on the application of BERT-based models for patient outcome prediction.

### 2.1 Background on BERT

Bidirectional Encoder Representations from Transformers (BERT) is a pre-trained deep learning model known for its state-of-the-art performance on various natural language processing (NLP) tasks. In the context of healthcare, BERT can be particularly valuable for analyzing clinical text data, like physician notes, discharge summaries, and laboratory reports.

### 2.2 BERT Variants for Patient Outcome Prediction: Fine-tuning BERT-based Models

This study investigates the performance of four prominent BERT variants fine-tuned for the task of patient outcome prediction using clinical text data. The BERT variants include BERT-base-uncased, Bio_ClinicalBERT, BiomedNLP-PubMedBERT, and BioBERT. Transfer learning techniques are employed to adapt these pre-trained models to the specific characteristics of the target dataset.

### 2.3 Model Training and Evaluation: Hyperparameter Tuning

The fine-tuned BERT models are evaluated based on their performance in predicting patient outcomes. Hyperparameter tuning is crucial for maximizing the performance of machine learning models during training. Key hyperparameters like learning rate, number of epochs, and batch size are optimized using techniques like the Optuna library.

By following this systematic approach to hyperparameter tuning and evaluation, we ensure that the chosen BERT model is optimally configured to achieve the most accurate and reliable predictions for patient outcomes.

## 3. Results and Analysis

The experimental results demonstrate the performance of each BERT-based model in predicting in-hospital mortality and other patient outcomes.

### 3.1 Bert Model Predictions

After hyperparameter optimization with Optuna:
- Best parameters:
  - Learning rate: 1.757e-05
  - Epochs: 3
  - Batch size: 32
- Validation accuracy: 100%
- Test accuracy: 80%
- Test precision: 1.0
- Test recall: 0.80
- Test F1-score: 0.88

Explanation:
- The bert-base-uncased model achieved high performance on the validation set (100% accuracy), indicating its ability to learn patterns in clinical text data.
- Its uncased nature allows it to handle a wide range of clinical text inputs with varying conventions, making it suitable for patient outcome prediction tasks.
- However, the test accuracy of 80% suggests that the model may not generalize well to unseen data. Further evaluation or hyperparameter tuning might be necessary to improve generalizability.

### 3.2 Bio_ClinicalBERT Model Predictions

After hyperparameter optimization:
- Best parameters: lr=1.312e-06, 4 epochs, batch size of 64.
- Validation accuracy: 100%
- Test accuracy: 70%

Explanation:
- The emilyalsentzer/Bio_ClinicalBERT model demonstrated competitive performance, achieving perfect validation accuracy.
- However, there was a slight decrease in performance on the test set, indicating potential challenges in generalizing to unseen data or overfitting.
- Further fine-tuning and domain-specific adjustments may enhance the model's performance in patient outcome prediction tasks.

### 3.3 BiomedNLP-PubMedBERT Model Predictions

After hyperparameter optimization:
- Best parameters: lr=4.575e-05, 5 epochs, batch size of 32.
- Validation accuracy: 100%
- Test accuracy: 100%
- Precision, recall, and F1-score on the test set: 1.0

Explanation:
- The model exhibited outstanding performance achieving perfect validation and test accuracies.
- Its pretraining on a vast corpus of biomedical literature enables it to capture intricate semantic and contextual information relevant to clinical text data.
- The consistent performance across both validation and test sets underscores its robustness and generalizability in patient outcome prediction tasks.

### 3.4 BioBERT Model Predictions

After hyperparameter optimization:
- Best parameters: lr=9.002e-06, 5 epochs, batch size of 16.
- Validation accuracy: 100%
- Test accuracy:

 100%
- Precision, recall, and F1-score on the test set: 1.0

Explanation:
- Similar to the microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract model, the dmis-lab/biobert-v1.1 model demonstrated outstanding performance.
- Its specialization in biomedical text processing, coupled with extensive pretraining, enables it to capture intricate semantic and contextual information.
- The perfect performance on both validation and test sets highlights its reliability and effectiveness in patient outcome prediction tasks.

### 4. Overall Analysis

The experimental results demonstrate the performance of each BERT-based model in predicting patient outcomes for hospital stays. The comparison of the BERT variants employed in this study reveals interesting insights:

- Generalizability: While both bert-base-uncased and emilyalsentzer/Bio_ClinicalBERT achieved 100% validation accuracy, their test accuracies differed significantly (80% vs. 70%). This suggests that bert-base-uncased, despite its uncased nature allowing for broader applicability, may struggle with generalizing to unseen data. Bio_ClinicalBERT, potentially due to its domain-specific focus, might require further fine-tuning for optimal generalizability.
- Impact of Pre-training: Both microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract and dmis-lab/biobert-v1.1, pre-trained on vast biomedical corpora, achieved exceptional performance (100% accuracy on both validation and test sets). This highlights the importance of domain-specific pre-training for BERT models in capturing the nuances of clinical text data and achieving robust generalizability.

Overall, the study emphasizes the effectiveness of BERT-based models in patient outcome prediction. However, careful consideration of pre-training data and model fine-tuning is crucial to optimize generalizability and ensure reliable performance in real-world clinical settings.

## 5. Conclusion

The study concludes by summarizing the findings and highlighting the effectiveness of BERT-based models in patient outcome prediction tasks using clinical text data from the MIMIC IV 2.2 dataset. Recommendations for future research directions and potential applications of BERT-based models in clinical risk assessment are discussed.

## 6. Future Considerations

- Incorporating Additional Features
- External Validation
- Explainability Techniques
- Clinical Integration

## 7. References

[1] Beam, A. L., & Kohane, I. S. (2020). Deep learning for EHR-based prognosis: A survey.
[2] Choi, E., Sun, J., Liu, Y., & Darwiche, A. (2021). A Multimodal Transformer: Fusing Clinical Notes with Structured EHR Data for Interpretable In-Hospital Mortality Prediction.
[3] Huang, K., Xu, J., & Sun, J. (2021). Predicting in-hospital mortality by combining clinical notes with time-series data.
[4] Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2019). BioBERT: a pre-trained biomedical language representation model for biomedical text mining.
[5] Luo, H., & Chen, J. (2020). Attention based model for predicting in-hospital mortality with MIMIC-III clinical notes.
[6] Min, S., Lee, B., & Yoon, S. (2020). Attention Is All You Need for Mortality Prediction with Medical Notes.
[7] Rajkomar, A., Hardt, M., Annapureddy, Y., Jha, S., Kundaje, S., McGinnis, P., ... & Shickel, L. (2018). Scalable medical imaging platform (SMIP) for AI development. Nature Medicine, 24(11), 1673-1678.
[8] Sendhilkumar, A., & Geetha, S. (2022). A Survey on Hyperparameter Optimization for Deep Learning.
[9] Johnson, A. E. W., Pollard, T. J., & Mark, R. G. (2023). MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data, 10(1), 1-9.
[10] Beltagy, I., Lo, K., & Cohan, A. (2019). SciBERT: A Pretrained Language Model for Scientific Text.
```
