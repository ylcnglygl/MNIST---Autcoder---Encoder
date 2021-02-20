from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras import Input, Model
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train,_),(X_test,_) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.


X_train = np.reshape(X_train, (-1, 28, 28, 1))
X_test = np.reshape(X_test, (-1, 28, 28, 1))

#ENCODING
input_img = Input(shape=(28,28,1))
x = Conv2D(16,(3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same', name='encoder')(x)

#DECODING
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(X_train, X_train, epochs=3, batch_size=256, callbacks=None, shuffle=True, )

autoencoder.save('autoencoder.h5')
autoencoder.summary()



encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
encoder.save('encoder.h5')




encoder=load_model('encoder.h5')
autoencoder=load_model('autoencoder.h5')
#query image
query = X_test[8000]
plt.imshow(query.reshape(28,28), cmap='gray')


codes = encoder.predict(X_train)
query_code = encoder.predict(query.reshape(1,28,28,1))

n_neigh = 10
codes = codes.reshape(-1, 4*4*8); print(codes.shape)
query_code = query_code.reshape(1, 4*4*8); print(query_code.shape)

nbrs = NearestNeighbors(n_neighbors=n_neigh).fit(codes)
distances, indices = nbrs.kneighbors(np.array(query_code))
print(distances)
print("Indices:", indices)
closest_images = X_train[indices]
closest_images = closest_images.reshape(-1,28,28,1); print(closest_images.shape)
#plt.imshow(query.reshape(28,28), cmap='gray')

plt.figure(figsize=(20, 6))
for i in range(n_neigh):
    # display original
    ax = plt.subplot(1, n_neigh, i+1)
    plt.imshow(closest_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()
plt.savefig('nearestneighbors.png')

