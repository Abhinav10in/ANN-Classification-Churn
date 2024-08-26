import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import datetime

data = pd.read_csv("Churn_Modelling.csv")

##Preprocess the data
#Drop irrevalent data
data = data.drop(['RowNumber','CustomerId','Surname'], axis=1)

##Encode categorical variables 
Label_encoder = LabelEncoder()
data['Gender'] = Label_encoder.fit_transform(data['Gender'])

##One hot encoding 

one_hot = OneHotEncoder()
geo_encoder = one_hot.fit_transform(data[['Geography']])

geo_encoded_df=pd.DataFrame(geo_encoder.toarray(),columns=one_hot.get_feature_names_out(['Geography']))


data = pd.concat([data.drop('Geography',axis=1),geo_encoded_df],axis=1)

##Save the encoders and scaler
"""
with open('label_enconder_gender.pkl','wb') as file:
    pickle.dump(Label_encoder,file)

with open('one_hot.pkl','wb') as file:
    pickle.dump(one_hot,file)

"""    
    
##Divide the dataset

X = data.drop('Exited',axis=1)
y=data['Exited']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# with open('scale.pkl','wb')  as file:
    # pickle.dump(scaler,file)

###################ANN Implementation #####################
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

###Build the model
model = Sequential([
    Dense(64,activation = 'relu',input_shape= (X_train.shape[1],)), # First hidden layer
    Dense(32,activation = 'relu'),
    Dense(1,activation = 'sigmoid')
]
)

#optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
loss = tf.keras.losses.BinaryCrossentropy()
#There are multiple other optimizers


###Compile the model 
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])

#Setup the tensorboard
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard

log_dir = "logs/fit"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback = TensorBoard(log_dir=log_dir,histogram_freq=1)

##setup early stopping
early_stopping_callback = EarlyStopping(monitor = 'val_loss',patience=10,restore_best_weights=True)

##Training the model
history = model.fit( X_train,y_train,validation_data=(X_test,y_test),epochs=100,
                    callbacks = [tensorflow_callback,early_stopping_callback])

model.save('model.h5')
#h5 is compatible with keras


"""
#magic command to visualize the logs 
%load_ext tensorboard
%tensorboard --log_dir logs/fit
"""


