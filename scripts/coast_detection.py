# Imports:
import os
import numpy as np
import matplotlib as plt
from PIL import Image
from IPython.display import display
import random
from itertools import islice
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, ReLU, ELU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.metrics import BinaryIoU, Precision, Recall, MeanIoU
from keras.losses import BinaryCrossentropy
from keras import backend as K
import tensorflow as tf
import tensorflow_datasets as tfds

# KH: Stop tensorflow from grabbing the entire GPU, as we need to share it with someone else
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

print(os.getcwd())

# Define Mean Intersection over Union (iou) metric: 
def mIoU_metric(y_true, y_pred):
  yt0 = K.cast(y_true[:,:,:,0] > 0.5, 'float32')
  yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')

  inter0 = tf.math.count_nonzero(tf.math.logical_and(tf.math.equal(yt0, 0), tf.math.equal(yp0, 0)))
  union0 = tf.math.count_nonzero(tf.math.logical_or(tf.math.equal(yt0, 0), tf.math.equal(yp0, 0)))
  iou0 = tf.where(tf.equal(union0, 0), 1., tf.cast(inter0/union0, 'float32'))

  inter1 = tf.math.count_nonzero(tf.math.logical_and(tf.math.equal(yt0, 1), tf.math.equal(yp0, 1)))
  union1 = tf.math.count_nonzero(tf.math.add(yt0, yp0))
  iou1 = tf.where(tf.equal(union1, 0), 1., tf.cast(inter1/union1, 'float32'))    

  meanIoU = (iou0 + iou1)/2
  return meanIoU


# Define Sobel edge loss function (Seale et al., 2022)
def sobel_edge_loss(y_true, y_pred):

  yt0 = tf.cast(y_true, tf.float32)
  yp0 = tf.cast(y_pred, tf.float32)

  # Calculate the Sobel edges for the ground truth and predicted images
  sobel_true = tf.image.sobel_edges(yt0)
  sobel_pred = tf.image.sobel_edges(yp0)

  # Calculate the mean squared error between the Sobel edges of the ground truth and predicted images
  loss = tf.reduce_mean(tf.square(sobel_true - sobel_pred))
  return loss

def sobel_crossentropy(y_true, y_pred):
  return sobel_edge_loss(y_true, y_pred) + BinaryCrossentropy()(y_true, y_pred)


def sample_loader(dataset, split):
  data = tfds.load(dataset, split=split)
  data = data.map(lambda x: (x['image'], x['mask']))
  return tfds.as_numpy(data)

def batch_loader(dataset, split, batch_size):
  do_shuffle = (split == 'train')
  data = tfds.load(dataset, split=split, shuffle_files=do_shuffle)
  data = data.map(lambda x: (x['image'] / 255, x['mask'] > 0))
  if do_shuffle:
    data = data.shuffle(1024 * 16)
  data = data.batch(batch_size)
  return data


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Load files here, define train, test, batch size

batch_size = 32
EPOCHS = 100

coasts_train = sample_loader('glaciers', 'train')
coasts_test = sample_loader('glaciers', 'test')
coasts_val = sample_loader('glaciers', 'val')

for image, mask in coasts_train:
  print('\nShape of images: ', image.shape)
  print('Shape of ground truth: ', mask.shape)
  break

train_batches = batch_loader('glaciers', 'train', batch_size)
val_batches = batch_loader('glaciers', 'val', batch_size)

print('Training coast samples: ', len(coasts_train))
print('Validation samples: ', len(coasts_val))
print('Test samples: ', len(coasts_test))
print('\n')

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------


def unet_CoastDetection(input_shape = (512, 512, 3)):       
  x = Input(input_shape)
  inputs = x

  #down sampling
  f = 8
  layers = []

  for i in range(0, 6):
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    layers.append(x)
    x = MaxPooling2D() (x)
    f = f*2
  ff2 = 64

  #bottleneck
  j = len(layers) - 1
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
  x = Concatenate(axis=3)([x, layers[j]])
  j = j -1

  #upsampling
  for i in range(0, 5):
    ff2 = ff2//2
    f = f // 2
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j -1


  #classification
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  outputs = Conv2D(1, 1, activation='sigmoid') (x)

  #model creation
  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', Precision(), Recall(), BinaryIoU(), mIoU_metric])

  return model



# New model - taken from paper
# Sobel edge loss

"""def double_conv_block(x, num_filters):
    # Conv2D then ELU activation
    x = Conv2D(num_filters, 3, activation='relu', padding="same")(x)    
    # x = BatchNormalization()(x)
    # x = ReLU()(x)
    # Conv2D then ELU activation
    x = Conv2D(num_filters, 3, activation='relu', padding="same")(x)
    # x = BatchNormalization()(x)
    # x = ReLU()(x)
    return x

def unet_CoastDetection(input_shape = (512, 512, 3)):       
  x = Input(input_shape)
  inputs = x

  #down sampling
  num_filters = 16
  layers = []

  # encoder path:
  for _ in range(0, 4):
    x = double_conv_block(x, num_filters)
    layers.append(x)
    x = MaxPooling2D(2)(x)
    num_filters = num_filters*2
  
  ff2 = num_filters // 2

  # bottleneck:
  j = len(layers) - 1
  x = double_conv_block(x, num_filters)
  x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
  x = Concatenate(axis=3)([x, layers[j]])
  j = j - 1

  #upsampling
  for _ in range(0, 3):
    ff2 = ff2 // 2
    num_filters = num_filters // 2
    x = double_conv_block(x, num_filters)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j - 1

  # classification
  num_filters = num_filters // 2
  x = double_conv_block(x, num_filters)
  outputs = Conv2D(1, 1, activation='sigmoid')(x)

  #model creation
  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', Precision(), Recall(), BinaryIoU(), mIoU_metric])

  return model
"""


# Callbacks
def build_callbacks():
  checkpointer = ModelCheckpoint(filepath='workspace_private/trained_models/unet_glacierDetection2-{epoch:02d}.h5', verbose=0, save_weights_only=True, save_best_only=True) # save_freq=freq) #  
  reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5)
  log_csv = CSVLogger('workspace_private/trained_models/glacier_model_logs2.csv', separator=',')
  early_stop = EarlyStopping(patience=10)       
  callbacks = [checkpointer, log_csv, reduce_lr, PlotLearning()] #  early_stop,
  # callbacks = [PlotLearning()]
  return callbacks

# inheritance for training process plot
class PlotLearning(keras.callbacks.Callback):

  def on_train_begin(self, logs={}):
    self.i = 0
    self.x = []
    self.losses = []
    self.val_losses = []
    self.acc = []
    self.val_acc = []
    #self.fig = plt.figure()
    self.logs = []
  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(logs)
    self.x.append(self.i)
    self.losses.append(logs.get('loss'))
    self.val_losses.append(logs.get('val_loss'))
    self.acc.append(logs.get('mIoU_metric'))
    self.val_acc.append(logs.get('val_mIoU_metric'))
    self.i += 1
    print('\ni=',self.i,'loss=',logs.get('loss'),'val_loss=',logs.get('val_loss'),'mIoU_metric=',logs.get('mIoU_metric'),'val_IoU_metric=',logs.get('val_mIoU_metric'))

    plot_prediction = False
    if plot_prediction:
      #choose a random test image and preprocess:
      img_index = random.randint(1, len(coasts_test))
      image, mask = list(islice(coasts_test, img_index-1, img_index))[0]

      # Predict the mask:
      pred = model.predict(np.asarray(image)[None, ...])

      # mask post-processing
      msk_pred  = pred[0,:,:,0]
      msk_pred = np.stack((msk_pred,)*3, axis=-1)
      msk_pred[msk_pred >= 0.5] = 1
      msk_pred[msk_pred < 0.5] = 0

      # show the mask and the segmented image
      combined = np.concatenate([image, msk_pred, image*msk_pred], axis = 1)
      plt.imshow(combined)
      plt.title(f'Prediction for test sample at epoch {self.i}')
      plt.axis('off')
      plt.show()



model = unet_CoastDetection()
# model.summary()

# # Load pretrained model:
# model = keras.models.load_model('workspace_private/trained_models/CoastDetection_model12_pretrain.keras', custom_objects={"sobel_crossentropy":sobel_crossentropy, "mIoU_metric":mIoU_metric})


# Training
train_steps = len(train_batches)
val_steps = len(val_batches)
# callb_freq = 5*train_steps
model.fit(train_batches,
          epochs = EPOCHS, steps_per_epoch = train_steps,
          validation_data = val_batches, validation_steps = val_steps,
          callbacks = build_callbacks())

model.save('workspace_private/trained_models/glacier_model2.keras')