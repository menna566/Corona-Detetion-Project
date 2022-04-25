import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

Dir = r'C:\Users\Data\Dataset'
Data = []
res = []
label = ["Negative", "Positive"]
for c in label:
    path = os.path.join(Dir, c)
    cnum = label.index(c)
    for img in os.listdir(path):
        pat_img = np.array(cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE))
        Data.append(pat_img.flatten())
        res.append(cnum)

Data = np.array(Data)
res = np.array(res)

#splitting Data
x_train, x_test, y_train, y_test = train_test_split(Data, res, test_size=0.25)

print("-------First Model---------")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(80, activation="softmax"),
    tf.keras.layers.Dense(1)
])

print("...Compiling the model...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="mean_squared_error",
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=5, epochs=25)

print("...Evaluating the accuracy...")
loss, acc = model.evaluate(x_test,  y_test)
print('Test accuracy : ', acc)

predictions = model.predict(x_test)
print("Predicted output = ", np.argmax(predictions[0]))
print ("Actual output = ", y_test[0])

print("-------Second Model---------")
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)
])

print("...Compiling the model...")
model2.compile(optimizer="Adam",
               loss="mean_squared_error",
               metrics=['accuracy'])

model2.fit(x_train, y_train, batch_size=2, epochs=30)

print("...Evaluating the accuracy...")
loss2, acc2 = model2.evaluate(x_test,  y_test)
print('Test accuracy : ', acc2)

predictions2 = model2.predict(x_test)
print("Predicted output = ", np.argmax(predictions2[0]))
print ("Actual output = ", y_test[0])

print("The accuracy of the first model = ", acc)
print("And accuracy of the Second model = ", acc2)
