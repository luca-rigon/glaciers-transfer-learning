# Imports:
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.losses import BinaryCrossentropy
import tensorflow_datasets as tfds
import tensorflow as tf
import time

# KH: Stop tensorflow from grabbing the entire GPU, as we need to share it with someone else
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


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



# test spatial accuracy:
def get_contours(img, y_true, y_pred, show_plot=False):
    
   cs_true = plt.contour(y_true[:,:,0], colors='red')
   cs_pred = plt.contour(y_pred[:,:,0])

   cont_len_true = len(cs_true.collections[0].get_paths()[0])
   vertices_true = cs_true.allsegs[0][:][0]

   pred_segs = cs_pred.allsegs
   pred_x = []
   pred_y = []

   # Extract the coordinates of every single contour point (of prediction):
   for k in range(len(pred_segs)):
      pred_elems = pred_segs[k]
      for j in range(len(pred_elems)):                         
         pred_vertices = pred_segs[k][j]
         xij= pred_vertices[:,0]                        
         yij = pred_vertices[:,1]
         pred_x = np.concatenate((pred_x, xij))
         pred_y = np.concatenate((pred_y, yij))

   if show_plot:
      plt.imshow(img)
      plt.axis('off')
      plt.show()
   else:
      plt.close()

   return vertices_true, pred_x, pred_y

def closest_coordinate(point, contour_coordinates, d_max=512):

    min_distance = d_max
    for coord in contour_coordinates:
        distance = np.linalg.norm(np.array(point) - np.array(coord))
        if distance < min_distance:
            min_distance = distance
    
    return min_distance


def sample_loader(dataset, split):
  data = tfds.load(dataset, split=split)
  data = data.map(lambda x: (x['image'], x['mask']))
  return tfds.as_numpy(data)


# Load trained model:
model = keras.models.load_model('/root/workspace_private/trained_models/glacier_model2.keras', custom_objects={"sobel_crossentropy":sobel_crossentropy, "mIoU_metric": mIoU_metric})

# Data:
coasts_test = sample_loader('glaciers', 'test')
n_test = len(coasts_test)


# Here: compute the overall spatial accuracy (average) over all samples

print('\nStart computing average buffer size...')
buffer_list = np.arange(0, 512, 5)
buffer_sum = np.zeros_like(buffer_list, dtype=float)

t_start = time.time()
for i, sample in enumerate(coasts_test):
    img, y_true = sample

    img = np.asarray(img)[None, ...]
    # msk_true = np.asarray(mask)[None, ...]
    pred = model(img, training=False)

    # mask post-processing
    y_pred  = pred[0,:,:,0]
    y_pred = np.stack((y_pred,)*3, axis=-1)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    msk_vertices, pred_x, pred_y = get_contours(img[0,:,:,:], y_true, y_pred, show_plot=False)

    contour_coordinates = [(x,y) for (x,y) in zip(msk_vertices[:,0],msk_vertices[:,1])]
    distances_to_coastline = [closest_coordinate(point, contour_coordinates) for point in zip(pred_x, pred_y)] ### !!!! (very) long computation time, is there a more efficient way?

    
    tot_points = len(distances_to_coastline)

    part_per_buffer = []
    for buffer in buffer_list:
        points_within_buffer = len([dist for dist in distances_to_coastline if dist <= buffer])
        part_per_buffer.append(points_within_buffer*100/tot_points)

    buffer_sum += np.array(part_per_buffer)

    if i%5 == 0: 
       time_i = time.time()
       dimediff_rel = (time_i - t_start)/(i+1)
       print(f'Processed images: {i}/{len(coasts_test)}')
       print(f'Est. duration of one image: {dimediff_rel:4f}s, remaining estimated:{dimediff_rel*(n_test-i-1):4f}s')
    # if i == 10: break


buffer_mean = buffer_sum/n_test
print(buffer_mean)
np.save('/root/workspace_private/glacierDetection_BCE_buffers.npy', buffer_mean)

