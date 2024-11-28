from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
from tf_keras.models import Sequential
from tf_keras.layers import LSTM,Dense
from tf_keras.callbacks import TensorBoard
import numpy as np
import os

categories = np.array(['Distracted','Attentive'])
sequences = 30
sequence_length = 30
log_dir = os.path.join('TensorBoard_logs1')
Input_data_path = os.path.join("inputData1")
tb_callback = TensorBoard(log_dir=log_dir)

index_label = {label:num for num ,label in enumerate(categories)}

sequences_data = []
labels = []
for categorie in categories:
    for sequence in range(sequences):
        frame_points = []
        for frame in range(1,sequence_length+1):
            res = np.load(os.path.join(Input_data_path,categorie,str(sequence),"{}.npy".format(frame)))
            frame_points.append(res)
        sequences_data.append(frame_points)
        labels.append(index_label[categorie])

X = np.array(sequences_data)
y = to_categorical(labels).astype(int)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.05)

np.save('X_train1.npy', X_train)
np.save('X_test1.npy', X_test)
np.save('y_train1.npy', y_train)
np.save('y_test1.npy', y_test)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation = 'relu', input_shape=(30,1662)))#(batch_size,frames,64)
model.add(LSTM(128, return_sequences=True, activation='relu'))#(batch_size,frames,128)
model.add(LSTM(64, return_sequences=True, activation='relu'))#(batch_size,frames,64)
model.add(LSTM(64, return_sequences=False, activation='relu'))#(batch_size,64)
model.add(Dense(64, activation='relu'))#(batch_size,64)
model.add(Dense(32, activation='relu'))
model.add(Dense(categories.shape[0],activation='softmax'))#probabilities

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])

model.fit(X_train,y_train,epochs=900,callbacks=[tb_callback])
model.summary()
model.save('training_data1.h5')



