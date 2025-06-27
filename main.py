# Gerekli kütüphaneleri import ediyoruz.
from keras.layers import Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Conv2DTranspose, Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers.legacy import Adam
from matplotlib import pyplot as plt
import numpy as np
import os

# Optimizer için gerekli parametreleri giriyoruz.
learning_rate = 0.0002
beta_1 = 0.5

# Adam optimizer'ı tanımlıyoruz.
optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1)

# Gerekli parametreleri tanımlıyoruz.
imgHeight, imgWeight, channels, = 64, 64, 3
latent_dim = 100
datasetPath = "sample_data"
epochs = 1000
batch_size = 64
save_interval = 100

# Örnek verileri yükleme ve normalizasyon
def loadImage():
    images = []
    for filename in os.listdir(datasetPath):
        path = os.path.join(datasetPath, filename)
        img = load_img(path, target_size=(imgHeight, imgWeight))
        img = img_to_array(img)
        img = (img - 127.5) / 127.5
        images.append(img)
    return np.array(images)

x_train = loadImage()

# Ayırt edici (Discriminator)
def buildDiscriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same', input_shape = (imgHeight, imgWeight, channels)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Üretici (Generator)
def buildGenerator():
    model = Sequential()
    model.add(Dense(8 * 8 * 256, input_dim=latent_dim))
    model.add(Reshape((8, 8, 256)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(128, kernel_size = (4,4), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(64, kernel_size = (4,4), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(channels, kernel_size = (4,4), strides=(2,2), padding='same', activation='tanh'))
    return model

# Modeli oluşturma ve derleme
discriminator = buildDiscriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Üretici ve Ayırt Edici'yi birleştirerek GAN modelini oluşturma
generator = buildGenerator()
# Dikkat generator modelini derlemiyoruz çünkü GAN modelini oluştururken discriminator'ün trainable özelliğini False yapacağız.

# GAN modelini oluşturma
gan = Sequential()
gan.add(generator)
gan.add(discriminator)
discriminator.trainable = False
# GAN modelini derleme
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

def train(epochs, batch_size, save_interval):
    for epoch in range(epochs):

        # Datasetten rastgele gerçek örnekler alıyoruz.
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        realImages = x_train[idx]

        # Rastgele gürültü kullanarak sahte görüntüler üretiyoruz.
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fakeImages = generator.predict(noise)

        # gerçek değerler 1 ve sahte değerler 0 olarak ayarlanıyor.
        realY = np.ones((batch_size, 1))
        fakeY = np.zeros((batch_size, 1))

        # Discriminator'ü gerçek ve sahte görüntülerle eğitiyoruz.
        # train_on_batch metodu ile her iki gruptan da kayıp ve doğruluk hesaplanıyor böylece her model kendi parametrelerini güncellemiş oluyor.
        discriminatorLossReal = discriminator.train_on_batch(realImages, realY)
        discriminatorLossFake = discriminator.train_on_batch(fakeImages, fakeY)
        discriminatorLoss = 0.5 * np.add(discriminatorLossReal, discriminatorLossFake)

        # Generator'ü eğitiyoruz.
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generatorY = np.ones((batch_size, 1))
        generatorLoss = gan.train_on_batch(noise, generatorY)

        # Her save_interval'de kayıp ve doğruluk değerlerini yazdırıyoruz ve görselleri kaydediyoruz.
        if epoch % save_interval == 0:
            print(f"Epoch {epoch} - D loss: {discriminatorLoss[0]}, D acc: {100 * discriminatorLoss[1]:.2f}%, G loss: {generatorLoss}")
            save_images(epoch)


# === Görsel Kaydetme ===
def save_images(epoch, n=5):
    noise = np.random.normal(0, 1, (n * n, latent_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(n, n, figsize=(n, n))
    count = 0
    for i in range(n):
        for j in range(n):
            axs[i, j].imshow(gen_imgs[count])
            axs[i, j].axis('off')
            count += 1
    plt.tight_layout()
    plt.savefig(f"generated_faces_epoch_{epoch}.png")
    plt.close()

# === Modeli Eğit ===
train(epochs, batch_size, save_interval)