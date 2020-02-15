### LSTM-CNN GAN model. Generator: LSTM, Discriminator: CNN, these two combined into a GAN.

#imports

import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LSTM, Conv1D, MaxPool1D, BatchNormalization, Reshape, Flatten


# fix random seed for reproducibility
np.random.seed(7)

def create_generator():
  generator = Sequential()
  generator.add(LSTM(50, input_shape=(12,4)))
  generator.add(Dense(6))
  generator.compile(loss="mean_squared_error", optimizer="RMSprop", metrics=["accuracy"])

  return generator

#generator = create_generator()
#generator.summary()

# CNN discriminator, Learn the distribution of the price.
# The goal of the gan model is to study the "characteristics" of, for example, the "Inflation" rate.
# The generator tries to generate "Inflation" data as real as possible based on the other features, e.g "Unempolyment", "Wage", etc.
FILTER_SIZE = 3
NUM_FILTER = 32
INPUT_SIZE = 3 # num of days we want to predict
MAXPOOL_SIZE = 1 # our data set is small, so we don't even need it 
BATCH_SIZE = 1 # our data set is small, we don't need large batch size
STEP_PER_EPOCH = 612//BATCH_SIZE 
EPOCHS = 10


def create_discriminator():
  discriminator = Sequential()
  discriminator.add(Reshape((6,1), input_shape=(6,)))
  discriminator.add(Conv1D(NUM_FILTER, FILTER_SIZE, input_shape = (6,1)))
  discriminator.add(LeakyReLU(0.2))
  discriminator.add(Dropout(0.1))
  discriminator.add(Conv1D(2*NUM_FILTER, FILTER_SIZE))
  discriminator.add(BatchNormalization())

  discriminator.add(Dense(units=50))
  discriminator.add(Dropout(0.1))
  
  #reduce the dimension of the model to 1
  discriminator.add(Flatten())

  discriminator.add(Dense(units=1, activation="sigmoid"))

  discriminator.compile(loss="mean_squared_error", optimizer="RMSprop")
  return discriminator

#discriminator = create_discriminator()
#discriminator.summary()

#Create a GAN model with LSTM as the generator and CNN as the discriminator
def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(12,4))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='mean_squared_error', optimizer='adam')
    return gan
gan = create_gan(discriminator, generator)
gan.summary()


#@title Training function for the entangled GAN model
def training(x_train, y_train, x_test, y_test, epochs=1, random_size=128):
    
    #Loading the data
    random_count = 4*x_train.shape[0] / random_size
    
    # Creating GAN
    generator= create_generator()

    #y_lstm = np.reshape(y_train, (y_train.shape[0],1))
    #scores = generator.fit(x=x_train,y=y_lstm, batch_size=1, epochs = 100, validation_data = (x_test, y_test))
    #plt.plot(generator.predict(x_train))
    #plt.plot(y_lstm)
    #plt.show()


    discriminator= create_discriminator()
    gan = create_gan(discriminator, generator)
    
    for e in range(1,epochs+1 ):
        print("Epoch %d" %e)
        for index in tqdm(range(random_size)):
        #generate  random noise as an input  to  initialize the  generator
            feature = x_train[np.random.randint(low=0,high=x_train.shape[0],size=random_size),:]
            
            # Generate fake MNIST images from noised input
            fake_money = generator.predict(feature)
            #print(fake_money)
            #print(fake_money.shape)
            # Get a random set of  real images

            #real_money =y_train[np.random.randint(low=0,high=y_train.shape[0],size=random_size),:]
            upper_bound = int(np.random.randint(low=random_size, high=y_train.shape[0], size=1))

            real_money = y_train[upper_bound-random_size: upper_bound]

            real_money = np.reshape(real_money, (real_money.shape[0],6))
            #print(real_money)
            #print(real_money.shape)
            
            #Construct different batches of  real and fake data 
            combination = np.concatenate([real_money, fake_money])
            
            # Labels for generated and real data
            y_dis=np.zeros(2*random_size)
            y_dis[:random_size]=0.9
            
            #Pre train discriminator on  fake and real data  before starting the gan. 
            discriminator.trainable=True
            discriminator.train_on_batch(combination, y_dis)
            
            #Tricking the noised input of the Generator as real data
            trick_feature = x_train[np.random.randint(low=0,high=x_train.shape[0],size=random_size),:]
            y_gen = np.ones(random_size)
            
            # During the training of gan, 
            # the weights of discriminator should be fixed. 
            #We can enforce that by setting the trainable flag
            discriminator.trainable=False
            
            #training  the GAN by alternating the training of the Discriminator 
            #and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(trick_feature, y_gen)
            
        if e == 1 or e % 20 == 0:
          #plot generator_predict(x_test) and y_test on the same graph
          plt.figure(figsize=(17,8))
          plt.plot(generator.predict(x_train))
          y_plot = np.reshape(y_train, (y_train.shape[0],6))
          plt.plot(y_plot)
          plt.show()
