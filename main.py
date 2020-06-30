
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model 

# Part 1 - Preprocessing 
# Importing the dataset
def load_data(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file.readlines():
            split_raw = line.split(' ')
            data.append(split_raw)
    x = []
    for row in data:
        x.append(list(map(float, row[0:2])))
    return x

def split(data):
    x = []
    y = []
    for row in data:
        x.append(row[0])
        y.append(row[1])
    return np.array(x), np.array(y)


data = load_data('dane7.txt')

np.random.shuffle(data)
train_data = data[0:60]
test_data = data[60:]

X_train, Y_train = split(train_data)
X_test, Y_test = split(test_data)

# Part 2 - Model Creation  
# Initialising the ANN
classifier = Sequential()
classifier.add(Dense(50, input_dim = 1))
classifier.add(Activation('relu'))
classifier.add(Dense(45, input_dim = 50))
classifier.add(Activation('relu'))
classifier.add(Dense(1, input_dim = 45))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 1, nb_epoch = 200)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# model visualization
plot_model(classifier,to_file="model_visualization.png",show_shapes = True,show_layer_names = True)

# H5 file
json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(json)
classifier.save_weights("model.h5")
print("model saved")

# Plotting results
plt.plot(X_test, Y_test, 'bo')
plt.plot(X_test, Y_pred, 'ro')
plt.xlabel("blue - Ytest")
plt.xlabel("blue - Xtest")
plt.savefig("model_plot_test_pred.png")
plt.show()

