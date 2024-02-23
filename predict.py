import mlflow
import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.datasets import reuters
logged_model = 'runs:/962f25d4372445f8ab3177c9551b92c5/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

max_words = 1000
batch_size = 32
epochs = 5

print("Loading data...")
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words, test_split=0.2)
tokenizer = Tokenizer(num_words=max_words)
x_test = tokenizer.sequences_to_matrix(x_test, mode="binary")
print(x_test.shape)
# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(x_test))