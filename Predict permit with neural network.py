# # Prédiction de l'obtention d'un permis avec un réseau de neurones artificiel

# ### 1. Nettoyage du dataset


# Importation des librairies

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Importtion du dataset

df_permit = pd.read_csv('dataset/Building_Permits.csv')

# Suppression des catégories inutiles

df_permit = df_permit.drop(['Permit Number'],axis=1)
df_permit = df_permit.drop(['Block'],axis=1)
df_permit = df_permit.drop(['Lot'],axis=1)
df_permit = df_permit.drop(['Street Number'],axis=1)
df_permit = df_permit.drop(['Street Number Suffix'],axis=1)
df_permit = df_permit.drop(['Unit'],axis=1)
df_permit = df_permit.drop(['Unit Suffix'],axis=1)
df_permit = df_permit.drop(['Description'],axis=1)
df_permit = df_permit.drop(['Permit Expiration Date'],axis=1)
df_permit = df_permit.drop(['Estimated Cost'],axis=1)
df_permit = df_permit.drop(['Existing Use'],axis=1)
df_permit = df_permit.drop(['Existing Units'],axis=1)
df_permit = df_permit.drop(['Plansets'],axis=1)
df_permit = df_permit.drop(['Location'],axis=1)
df_permit = df_permit.drop(['Record ID'],axis=1)
df_permit = df_permit.drop(['Filed Date'],axis=1)

#Remplacement des données manquantes avec des zéros (voir paramètre du fillna)

df_permit['Number of Existing Stories'] = df_permit['Number of Existing Stories'].fillna(0)
df_permit['Number of Proposed Stories'] = df_permit['Number of Proposed Stories'].fillna(0)
df_permit['Structural Notification'] = df_permit['Structural Notification'].fillna(0)
df_permit['Voluntary Soft-Story Retrofit'] = df_permit['Voluntary Soft-Story Retrofit'].fillna(0)
df_permit['Fire Only Permit'] = df_permit['Fire Only Permit'].fillna(0)
df_permit['TIDF Compliance'] = df_permit['TIDF Compliance'].fillna(0)
df_permit['Site Permit'] = df_permit['Site Permit'].fillna(0)
df_permit['Street Suffix'] = df_permit['Street Suffix'].fillna(0)
df_permit['Existing Construction Type'] = df_permit['Existing Construction Type'].fillna(0)
df_permit['Proposed Construction Type'] = df_permit['Proposed Construction Type'].fillna(0)
df_permit['Existing Construction Type Description'] = df_permit['Existing Construction Type Description'].fillna(0)
df_permit['Proposed Construction Type Description'] = df_permit['Proposed Construction Type Description'].fillna(0)
df_permit['Proposed Use'] = df_permit['Proposed Use'].fillna(0)
df_permit['Revised Cost'] = df_permit['Revised Cost'].fillna(np.nanmedian(df_permit['Revised Cost']))
df_permit['Proposed Units'] = df_permit['Proposed Units'].fillna(np.nanmedian(df_permit['Proposed Units']))
df_permit = df_permit.dropna(subset=['First Construction Document Date'])
df_permit = df_permit.dropna(subset=['Neighborhoods - Analysis Boundaries'])
df_permit = df_permit.dropna(subset=['Zipcode'])
df_permit = df_permit.dropna(subset=['Issued Date'])
df_date = df_permit[['Permit Creation Date','Current Status Date','Issued Date','Completed Date']]
df_date['Permit Creation Date'] = pd.to_datetime(df_date['Permit Creation Date'])
df_date['Current Status Date'] = pd.to_datetime(df_date['Current Status Date'])
df_date['Issued Date'] = pd.to_datetime(df_date['Issued Date'])
df_date['Completed Date'] = df_date['Completed Date'].fillna(np.NaN)
df_date['Completed Date'] = pd.to_datetime(df_date['Completed Date'])

df_date2=df_date
df_date2['Issued Date'] = (df_date['Issued Date'] - df_date['Permit Creation Date']).dt.days
df_date2['Current Status Date'] = (df_date2['Current Status Date'] - df_date2['Permit Creation Date']).dt.days
df_date2['Completed Date'] = (df_date2['Completed Date'] - df_date2['Permit Creation Date']).dt.days




# In[5]:


#On rajoute nos données modifiées dans le dataset initial 

a = np.array(df_date2['Completed Date'].values.tolist())
df_date2['Completed Date'] = np.where(a > 5000, 9999, a).tolist()

df_permit['Current Status Date'] = df_date2['Current Status Date']
df_permit['Issued Date'] = df_date2['Issued Date']
df_permit['Completed Date'] = df_date2['Completed Date']

df_permit.head()


# In[6]:


#One hot encoder : on convertit les données de type string en integer pour les passer dans notre réseau de neurones

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X_1 = LabelEncoder()

df_permit_encoded = df_permit
df_permit_encoded['Permit Type Definition'] =labelencoder_X_1.fit_transform(df_permit_encoded['Permit Type Definition'])
df_permit_encoded['Street Name'] =labelencoder_X_1.fit_transform(df_permit_encoded['Street Name'])
df_permit_encoded['Street Suffix'] =labelencoder_X_1.fit_transform(df_permit_encoded['Street Suffix'].astype(str))
df_permit_encoded['Current Status'] =labelencoder_X_1.fit_transform(df_permit_encoded['Current Status'].astype(str))
df_permit_encoded['Existing Construction Type Description'] =labelencoder_X_1.fit_transform(df_permit_encoded['Existing Construction Type Description'].astype(str))
df_permit_encoded['Structural Notification'] =labelencoder_X_1.fit_transform(df_permit_encoded['Structural Notification'].astype(str))
df_permit_encoded['TIDF Compliance'] =labelencoder_X_1.fit_transform(df_permit_encoded['TIDF Compliance'].astype(str))
df_permit_encoded['Site Permit'] =labelencoder_X_1.fit_transform(df_permit_encoded['Site Permit'].astype(str))
df_permit_encoded['Neighborhoods - Analysis Boundaries'] =labelencoder_X_1.fit_transform(df_permit_encoded['Neighborhoods - Analysis Boundaries'].astype(str))
df_permit_encoded['Proposed Construction Type Description'] =labelencoder_X_1.fit_transform(df_permit_encoded['Proposed Construction Type Description'].astype(str))
df_permit_encoded['Number of Proposed Stories'] =labelencoder_X_1.fit_transform(df_permit_encoded['Number of Proposed Stories'].astype(str))
df_permit_encoded['Voluntary Soft-Story Retrofit'] =labelencoder_X_1.fit_transform(df_permit_encoded['Voluntary Soft-Story Retrofit'].astype(str))
df_permit_encoded['Fire Only Permit'] =labelencoder_X_1.fit_transform(df_permit_encoded['Fire Only Permit'].astype(str))
df_permit_encoded['Revised Cost'] =labelencoder_X_1.fit_transform(df_permit_encoded['Revised Cost'].astype(str))
df_permit_encoded['Proposed Use'] =labelencoder_X_1.fit_transform(df_permit_encoded['Proposed Use'].astype(str))
df_permit_encoded.head()


#Probleme : la majorité des données ne mènent pas à l'obtention d'un permis. Dans notre cas, il faut avoir 
# 50% de permis validés et 50% de permis refusés pour optimiser les perf de notre réseau de neurones

df_permit_validated = df_permit_encoded.loc[df_permit_encoded['Site Permit'] == 1]
df_permit_refused = df_permit_encoded.loc[df_permit_encoded['Site Permit'] == 0]

df_permit_encoded_equal = df_permit_validated.append(df_permit_refused.iloc[:3199,])
df_permit_encoded_equal.sample(frac=1)

df_permit_encoded_equal = df_permit_encoded_equal.drop(['Permit Creation Date'],axis=1)
df_permit_encoded_equal = df_permit_encoded_equal.drop(['First Construction Document Date'],axis=1)
df_permit_encoded_equal = df_permit_encoded_equal.drop(['Completed Date'],axis=1)

issuedate = df_permit_encoded_equal['Issued Date']

df_permit_encoded_equal.head()


# # Prediction de site permit

#On normalise les données pour diminuer le temps de traitement de notre algo et augmenter ses perf

from sklearn import preprocessing

X = df_permit_encoded_equal.drop(['Site Permit'],axis=1)
Y = df_permit_encoded_equal['Site Permit']


min_max_scalerX = preprocessing.MinMaxScaler()
X_scaled = min_max_scalerX.fit_transform(X)

min_max_scalerY = preprocessing.MinMaxScaler()
Y_scaled = min_max_scalerY.fit_transform(np.array(Y).reshape(-1, 1))


# Création de l'algo 

from sklearn.model_selection import train_test_split

#On sépare le dataset en 4 dataset pour l'entrainer et le tester

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

# Initialising the ANN : création des différentes couches
classifier = Sequential()

classifier.add(Dense(units=15, activation='relu', input_dim=23))
classifier.add(Dropout(0.2))

classifier.add(Dense(units=50, activation='relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(units=15, activation='relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(1, activation='sigmoid'))


#Sélection de la méthode d'optimisation

classifier.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

callbacks = [EarlyStopping(monitor='loss', patience=2)]

# serialize model to JSON
model_json = classifier.to_json()
with open("sitepermit.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("sitepermit.h5")
print("Saved model to disk")

history = classifier.fit(X_train, y_train,
                             batch_size=10,
                             epochs=50,
                             callbacks=callbacks,
                             validation_data=(X_test, y_test))

import matplotlib.pyplot as plt

print(history.history.keys())

# summarize history for loss
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

scores_train = classifier.evaluate(X_train, y_train, verbose=0)
scores_test = classifier.evaluate(X_test, y_test, verbose=0)

print("scores_train = ", scores_train[1] * 100, '%')
print("scores_test = ", scores_test[1] * 100, '%')





# # Predict the permit issue times

# In[65]:


#On normalise les données pour diminuer le temps de traitement de notre algo et augmenter ses perf

from sklearn import preprocessing

X2 = df_permit_encoded_equal.drop(['Site Permit','Issued Date'],axis=1)
Y2 = df_permit_encoded_equal['Issued Date']


min_max_scalerX2 = preprocessing.MinMaxScaler()
X_scaled2 = min_max_scalerX2.fit_transform(X2)

min_max_scalerY2 = preprocessing.MinMaxScaler()
Y_scaled2 = min_max_scalerY2.fit_transform(np.array(Y2).reshape(-1, 1))



# In[127]:


# Création de l'algo 

from sklearn.model_selection import train_test_split

#On sépare le dataset en 4 dataset pour l'entrainer et le tester

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_scaled2, Y_scaled2, test_size=0.1)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

# Initialising the ANN : création des différentes couches

model = Sequential()

model.add(Dense(units=15, activation='relu', input_dim=22))
model.add(Dropout(0.2))

model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=15, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))


#Sélection de la méthode d'optimisation

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.0005),
              metrics=['acc'])

callbacks = [EarlyStopping(monitor='loss', patience=2)]


history = model.fit(X_train2, y_train2,
                             batch_size=5,
                             epochs=100,
                             callbacks=callbacks,
                             validation_data=(X_test2, y_test2))


import matplotlib.pyplot as plt

print(history.history.keys())

# summarize history for loss
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

scores_train = model.evaluate(X_train2, y_train2, verbose=0)
scores_test = model.evaluate(X_test2, y_test2, verbose=0)

print("scores_train = ", scores_train[1])
print("scores_test = ", scores_test[1])

prediction = model.predict(X_test2)
df_predicted = pd.DataFrame(min_max_scalerY2.inverse_transform(prediction))
df_test = pd.DataFrame(y_test2)
df_test = pd.DataFrame(min_max_scalerY2.inverse_transform(df_test))
result = pd.concat([df_test,df_predicted],axis=1)
result.columns = ['y_test','predicted']
print(np.abs(np.mean(result['y_test']-result['predicted'])))



model_json = model.to_json()
with open("delaypermit.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("delaypermit.h5")
print("Saved model to disk")
