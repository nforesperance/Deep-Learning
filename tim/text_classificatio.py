import tensorflow as tf
from tensorflow import keras
import numpy

# Movies reviews
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=88000)

# Interger Encoded (from keras github)
# A dictionary mapping words to an integer index
_word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# this function will return the decoded (human readable) reviews  
def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])

# Preprocessing Data (Due to varying lenghts)
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# Defining our model
# We have pick 16 as our the dimension of the word vector
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16)) 
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# Training 
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
results = model.evaluate(test_data, test_labels)
print(results)

# Saving a model to avoid training every time
model.save("model.h5")
model = keras.models.load_model("model.h5")

def review_encode(s):
	encoded = [1]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded
with open("test.txt", encoding="utf-8") as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = review_encode(nline)
        
        # make the data 250 words long
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) 
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])