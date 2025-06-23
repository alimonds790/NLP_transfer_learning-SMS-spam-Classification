# Deep Learning for SMS Spam Detection
**Author:** Ali Saleh

# 1. Problem Identification and Goal

SMS spam is a persistent issue for mobile users, ranging from annoyances to serious threats. Its volume and evolving nature make manual filtering impractical.

The primary goal of this project is to develop, train, and evaluate deep learning models capable of accurately classifying SMS messages as either spam or not.


## Dataset Description and Exploratory Data Analysis

**Data Source:** The dataset used in this project is the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository.

**Description:** This dataset consists of 5,572 SMS messages in English. Each message is tagged as either "ham" (legitimate) or "spam". It is a widely used benchmark dataset for text classification tasks.


Initial Exploratory Data Analysis (EDA) revealed several key characteristics of the dataset:
- **Class Imbalance:** The dataset is highly imbalanced, with a significant majority of 'ham' messages (approx. 86.6%) compared to 'spam' messages (approx. 13.4%). This highlights the need for appropriate evaluation metrics (precision, recall, F1-score for spam) and stratified data splitting.
- **Message Length and Word Count:** Spam messages tend to be longer and have a higher word count on average than ham messages, suggesting these features could be useful for classification.
- **Common Words:** Word cloud analysis showed distinct vocabularies for each class. Spam messages frequently contain words related to promotions and offers ("free", "call", "txt", "claim", "prize"), while ham messages use more conversational terms ("get", "go", "know", "love", "good").




## Data Cleaning and Preprocessing

For this project, leveraging the power of a pre-trained Transformer model like BERT (specifically, DistilBERT), the traditional text cleaning steps often performed in NLP tasks (like removing punctuation, lowercasing, stemming, or removing numbers) were intentionally **omitted**.

### Decision on Text Cleaning

**BERT tokenizers are pretrained on raw, untampered text.** They are designed to handle various linguistic nuances, including punctuation, capitalization, and even the presence of URLs, emails, and numbers. Stripping these elements can actually disrupt the learned patterns and potentially reduce the model's effectiveness. BERT's WordPiece tokenization also handles unknown words effectively, making aggressive cleaning unnecessary and potentially harmful.

### Label Encoding

The target variable, 'label' (which is categorical: 'ham' or 'spam'), needs to be converted into a numerical format for the model. This was done using **Label Encoding**, mapping 'ham' to 0 and 'spam' to 1.

### Data Splitting

To ensure robust model evaluation and prevent data leakage, the dataset was split into three sets:

- **Training Set:** Used to train the model.
- **Validation Set:** Used during training to monitor performance and tune hyperparameters, helping to prevent overfitting.
- **Test Set:** An unseen dataset used only for the final evaluation of the trained model's performance.

The data was split using `train_test_split` from scikit-learn. An initial split allocated 20% of the data to the test set. The remaining 80% was then further split, allocating 10% of that portion (8% of the total data) to the validation set, leaving 72% for training. **Stratification** was applied during both splits to maintain the original class distribution of 'ham' and 'spam' in each subset, which is crucial given the dataset's imbalance.

### Tokenization with BERT

Transformer models like BERT require input text to be processed into a specific numerical format through tokenization. This project used the **`BertTokenizer`** loaded from the pre-trained **`distilbert-base-uncased`** model.

The tokenization process involves:
1.  Breaking down the text into tokens according to the tokenizer's vocabulary.
2.  Adding special tokens required by BERT (e.g., `[CLS]` at the beginning and `[SEP]` at the end).
3.  Converting tokens into their corresponding numerical IDs.
4.  Padding or truncating sequences to a fixed maximum length (`MAX_LENGTH`, set to 128) to ensure uniform input size.
5.  Creating attention masks to differentiate between actual tokens and padding tokens.

A custom function, `bert_encode_texts`, was used to apply this process to each message in the training, validation, and test sets, generating `input_ids` and `attention_masks` tensors for each set.

### Creating TensorFlow Datasets

For efficient training with TensorFlow, the tokenized data (`input_ids` and `attention_masks`) and the numerical labels were converted into **`tf.data.Dataset`** objects.

The `create_tf_dataset_for_bert` function facilitated this conversion. Using `tf.data.Dataset` provides several benefits:
-   **Batching:** Data is grouped into batches (`BATCH_SIZE`, set to 32) for efficient processing by the model.
-   **Shuffling:** The training dataset is shuffled to ensure the model does not learn the order of samples.
-   **Prefetching:** Data batches are prefetched, allowing the next batch to be loaded while the current batch is being processed by the GPU, significantly improving training speed by reducing idle time.

These datasets (`train_dataset_bert`, `val_dataset_bert`, `test_dataset_bert`) are now ready to be fed directly into the BERT model for fine-tuning and evaluation.



## Model Description and Fine-tuning

### Chosen Model: DistilBERT

For this SMS spam detection task, we utilized **DistilBERT (specifically `distilbert-base-uncased`)**, a smaller, faster, and lighter version of BERT. DistilBERT retains approximately 95% of BERT's performance while being 40% smaller and 60% faster. This makes it an excellent choice for tasks where computational resources or inference speed are considerations, without significant sacrifice in accuracy.

### Fine-tuning Process

The pre-trained DistilBERT model, configured for sequence classification with a single output neuron for binary classification, was fine-tuned on our specific SMS spam dataset.

#### Optimizer, Loss Function, and Learning Rate

-   **Optimizer:** The **AdamW** optimizer was used, which is a variant of Adam that includes weight decay regularization, commonly recommended for training Transformer models.
-   **Loss Function:** Since the model outputs a single logit for binary classification, **`BinaryCrossentropy`** with `from_logits=True` was used as the loss function.
-   **Initial Learning Rate:** An initial learning rate of **`2e-5`** was set, a typical small value for fine-tuning large pre-trained models.

#### Learning Rate Scheduling

To further optimize the training process and potentially improve convergence, a **Polynomial Decay learning rate scheduler** was implemented. This scheduler gradually decreases the learning rate over the training steps from the initial learning rate (`2e-5`) down to a very small end learning rate (`1e-6`) over a specified number of decay steps (1000). This approach helps the model make larger updates early in training and smaller, more refined updates later.

#### Freezing Lower Layers

During fine-tuning, it is common practice to freeze the lower layers of the pre-trained model and only train the upper layers and the newly added classification head. This leverages the general language understanding learned by the lower layers during pre-training while allowing the model to adapt to the specific task with the upper layers. Based on research suggesting freezing the first 6 layers can be effective [4], the first **6 layers** of the DistilBERT encoder were frozen (`layer.trainable = False`).

#### Addressing Class Imbalance with Class Weights

Given the significant imbalance in the dataset (many more 'ham' messages than 'spam'), training directly could lead the model to be biased towards the majority class ('ham'). To mitigate this, **class weights** were calculated and applied during training. The weights are inversely proportional to the class frequencies, giving more importance to the minority class ('spam') during the loss calculation.

The formula used for calculating class weights is:
`weight for class c = total_samples / (num_classes * count_of_class_c)`

These calculated weights were passed to the `fit` method using the `class_weight` parameter.

#### Callbacks

To enhance the training process, the following Keras callbacks were used:

-   **`EarlyStopping`:** Monitored the validation loss (`val_loss`) and stopped training if the loss did not improve for a specified number of epochs (`patience=1`). It also restored the model weights from the epoch with the best validation loss.
-   **`ModelCheckpoint`:** Saved the model weights (`save_weights_only=True`) whenever the validation accuracy (`val_accuracy`) improved, ensuring that the best performing model during fine-tuning was saved. The checkpoint file was named `./bert_spam_detector_checkpoint.weights.h5`.

#### Number of Epochs

Following recommendations from the original BERT paper [3], fine-tuning was performed for **3 epochs (`NUM_EPOCHS_BERT = 3`)**. This limited number of epochs is typically sufficient for fine-tuning pre-trained Transformer models on downstream tasks.



## Model Evaluation

The fine-tuned DistilBERT model was evaluated on the unseen test dataset to assess its performance. The key metrics examined were overall accuracy, precision, recall, and F1-score, with a particular focus on the 'spam' class due to the dataset's imbalance.

### Test Set Performance Metrics

The evaluation on the test set yielded the following results:

-   **Overall Accuracy:** 0.9839 (98.39%) - Represents the proportion of correctly classified messages (both ham and spam) out of the total messages in the test set. While high, this metric can be misleading in imbalanced datasets.
-   **Precision (Spam):** 0.9338 (93.38%) - This is the ratio of correctly predicted spam messages (True Positives) to the total number of messages predicted as spam (True Positives + False Positives). A high precision for spam means that when the model says a message is spam, it is highly likely to be correct, minimizing false alarms (ham messages incorrectly classified as spam).
-   **Recall (Spam):** 0.9463 (94.63%) - This is the ratio of correctly predicted spam messages (True Positives) to the total number of actual spam messages (True Positives + False Negatives). A high recall for spam means the model is effective at identifying most of the actual spam messages, minimizing the number of spam messages that slip through (false negatives).
-   **F1-score (Spam):** 0.9400 (94.00%) - The F1-score is the harmonic mean of precision and recall. It provides a balanced measure of the model's performance on the spam class, especially useful when there is an uneven class distribution. A high F1-score indicates a good balance between precision and recall.




## Conclusion and Future Work

The fine-tuned DistilBERT model achieved excellent performance on the unseen test dataset, with a high overall accuracy of 98.39%. More importantly, considering the class imbalance, the model demonstrated strong performance on the minority "spam" class:

*   **Precision (Spam):** 0.9338 - This means that when the model predicts a message is spam, it is correct about 93.4% of the time, minimizing false positives.
*   **Recall (Spam):** 0.9463 - This indicates that the model successfully identifies about 94.6% of all actual spam messages, minimizing false negatives.
*   **F1-score (Spam):** 0.9400 - This is a harmonic mean of precision and recall, providing a balanced measure of the model's effectiveness on the spam class.

The confusion matrix visually confirms these results, showing a low number of misclassifications for both ham and spam. The class weighting during training likely contributed to the strong performance on the spam class despite the dataset's imbalance.

Overall, the fine-tuned DistilBERT model is highly effective for this SMS spam detection task, demonstrating the power of transfer learning for NLP.

While the current model performs well, here are some potential areas for future exploration and improvement:

1.  **Explore Other Pre-trained Models:** Experiment with other state-of-the-art Transformer models (e.g., RoBERTa, ELECTRA, or even larger BERT variants if computational resources allow) to see if further performance gains can be achieved.
2.  **Hyperparameter Tuning:** Conduct more extensive hyperparameter tuning for the finetuning process, including learning rate schedules, batch size, optimizer parameters, and the number of unfrozen layers.
3.  **Data Augmentation:** Implement data augmentation techniques specific to text data (e.g., synonym replacement, random insertion, deletion, or swapping of words) to potentially increase the size and diversity of the training data, especially for the minority class.
4.  **Ensemble Methods:** Combine predictions from multiple models (e.g., different Transformer architectures or traditional ML models) to potentially improve robustness and performance.


## References

1. Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.

2. user702846. (2022, August 9). Why there is no preprocessing step for training BERT? Data Science Stack Exchange. Retrieved June 23, 2025, from https://datascience.stackexchange.com/questions/113359/why-there-is-no-preprocessing-step-for-training-bert

3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre‑training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL‑HLT), Volume 1 (Long and Short Papers) (pp. 4171–4186). Association for Computational Linguistics.

4. Lee, J., Tang, R., & Lin, J. (2019, November 8). What Would Elsa Do? Freezing Layers During Transformer Fine‑Tuning [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1911.03090
