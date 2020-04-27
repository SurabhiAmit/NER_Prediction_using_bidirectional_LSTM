# NER_Prediction_using_bidirectional_LSTM
Prediction of NER tags for twitter tweets using bidirectional LSTM


### Process flow:
1. Collected data on tweet tokens and corresponding tags.
2. Cleaned the url tokens and username tokens and built vocabulary dictionary.
3. Performed analysis on the data to ensure data quality
4. Using tensorflow library functions, placeholder tensors were declared.
5. The RNN network was built using LSTM cells together with dropout wrappers as a regularization method to reduce overfitting.
6. A dense layer of LSTM cells was added on top of the concatenated forward and backward LSTM cells, to perform regression over the RNN.
7. Softmax was applied on the output of RNN and argmax was taken over the softmax output to get the best predictable tag.
8. Cross-entropy loss function was used for the RNN and it was ensured that the loss function does not operate over the padding tokens created during batch generation for training purposes (all sequences within a batch need to have the same length).
9. Adam optimizer was used and the gradients were clipped appropriately to eliminate exploding gradients.
10. While training the model, it was ensured that the actual sequenece lengths of different batches were provided so as to skip computations for padded tokens.
11. Hyperparameter tuning was performed and the best found values are used in the project. 
