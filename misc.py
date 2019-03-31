
# The evalation of the robustness of a model stored in the file 'simplenet_rbf_kmeans_fourth.h5'
with tf.Session() as sess:

    tf.keras.backend.set_session(sess)
  
    model = create_model(act="relu")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.load_weights('simplenet_rbf_kmeans_fourth.h5')
  
    logits = tf.keras.Model(model.inputs, model.get_layer('logits').output)

    attack = cleverhans.attacks.FastGradientMethod(cleverhans.model.CallableModelWrapper(logits, 'logits'), sess=sess)
    
    
    results = []
    eps_vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    for eps in eps_vals:
      r = []
      for i in range(10):
        start = 1000*i
        end = 1000*(i+1)
        adv_x = attack.generate_np(x_test[start:end], eps=eps)
        r.append(model.evaluate(adv_x, y_test[start:end])[1])
      results.append(np.mean(r))
    
    print(results)

# Visualization of the features (Fig. 1) (for the RBFolution)
with tf.Session() as sess:
  model = create_model()
  model.load_weights('simplenet_rbf_kmeans_fourth.h5')
  m1 = tf.keras.Model(model.input, model.layers[0].output)
  features = m1(x_train[idx].reshape((1,32,32,3))).eval()

import matplotlib.gridspec as gridspec

plt.figure(figsize = (8,4))
gs1 = gridspec.GridSpec(4, 8)
gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 

for i in range(32):
   # i = i + 1 # grid spec indexes from 0
    ax1 = plt.subplot(gs1[i])
    plt.axis('on')
    plt.grid(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    ax1.imshow(features[0, :, :, i])

plt.show()

# Dependence of robustness on the number of epochs on MNIST

import tensorflow as tf
import numpy as np

import cleverhans.attacks
import cleverhans.model

import data_sources
import models_mnist as models

def evaluate_adv(model, sess, test_x, test_y):
    logits = tf.keras.Model(model.inputs, model.get_layer('logits').output)

    attack = cleverhans.attacks.FastGradientMethod(cleverhans.model.CallableModelWrapper(logits, 'logits'), sess=sess)

    results = []
    eps_vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    for eps in eps_vals:
        r = []
        adv_x = attack.generate_np(test_x, eps=eps, clip_min=0, clip_max=1)
        r.append(model.evaluate(adv_x, test_y)[1])

        results.append(np.mean(r))

    return results

with tf.Session() as sess:

    tf.keras.backend.set_session(sess)

    (train_x, train_y), (test_x, test_y) = data_sources.mnist()

    results = []
    for _ in range(10):
        r = []
        #model = models.create_model_rbfolution_pretrained(train_x[:100], input_shape=train_x.shape[1:], nb_classes=train_y.shape[1])
        model = models.create_model_cnn(input_shape=train_x.shape[1:], nb_classes=train_y.shape[1])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        for _ in range(20):

            model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=1, batch_size=256, verbose=None)
            r.append(evaluate_adv(model, sess, test_x, test_y))

        print(r)

        results.append(r)

    print(results)