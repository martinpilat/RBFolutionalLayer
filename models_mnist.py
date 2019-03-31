import tensorflow as tf
import numpy as np

from custom_layers import RBFolution

def create_model_rbfolution_pretrained(x, input_shape, nb_classes):

    ccs_initializer = tf.keras.initializers.RandomUniform(0, 1)
    beta_initilizer = tf.keras.initializers.RandomUniform(0, 1)

    if x is not None:
        init_ccs, init_beta = get_init_ccs_beta(x, 32, (1,5,5,input_shape[-1]))
        ccs_initializer = tf.constant_initializer(init_ccs, verify_shape=True)
        beta_initilizer = tf.constant_initializer(init_beta, verify_shape=True)

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        RBFolution(filters=32,
                   kernel_size=(1,5,5,input_shape[-1]),
                   ccs_initializer=ccs_initializer,
                   beta_initilizer=beta_initilizer),
        tf.keras.layers.MaxPool2D(strides=2, pool_size=(2,2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(strides=2, pool_size=(2, 2)),
        tf.keras.layers.Flatten(name='Flatten'),
        tf.keras.layers.Dense(units=30, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=nb_classes, activation=tf.keras.activations.linear, name='logits'),
        tf.keras.layers.Activation(activation=tf.nn.softmax)
    ])

    return model

def create_model_cnn(input_shape, nb_classes):

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5)),
        tf.keras.layers.MaxPool2D(strides=2, pool_size=(2,2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(strides=2, pool_size=(2, 2)),
        tf.keras.layers.Flatten(name='Flatten'),
        tf.keras.layers.Dense(units=30, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=nb_classes, activation=tf.keras.activations.linear, name='logits'),
        tf.keras.layers.Activation(activation=tf.nn.softmax)
    ])

    return model

def get_init_ccs_beta(X, out_filters, kernel_shape):
    import sklearn.cluster
    from sklearn.feature_extraction.image import extract_patches

    kms = sklearn.cluster.KMeans(
        out_filters,
        max_iter=100,
        n_jobs=-1,
        n_init=3
    )

    patch_dim = (1, kernel_shape[1], kernel_shape[2], 1)
    reshaped_patches = extract_patches(X, patch_dim).reshape(
        [-1, np.prod(kernel_shape[1:])]
    )

    ccs = kms.fit(reshaped_patches).cluster_centers_
    beta = get_init_beta(
        ccs, kms.predict(reshaped_patches), reshaped_patches)
    return ccs.T, beta

def get_init_beta(ccs, closest_cluster_ids, reshaped_X):
    def mean_sq_distance_to_cc(i):
        X = reshaped_X[closest_cluster_ids == i]
        if not len(X):
            return np.inf
        dists = np.sum(np.square(ccs[i] - X), axis=1)
        return np.mean(dists)

    init_beta = [
        1 / (2 * mean_sq_distance_to_cc(i))
        for i in range(len(ccs))
    ]
    return init_beta

def create_simplenet(input_shape, nb_classes, act="relu"):
    model = tf.keras.models.Sequential()

    # Block 1
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.glorot_normal(), input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(act))
    model.add(tf.keras.layers.Dropout(0.2))

    # Block 2
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.glorot_normal()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(act))
    model.add(tf.keras.layers.Dropout(0.2))

    # Block 3
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(act))
    model.add(tf.keras.layers.Dropout(0.2))

    # Block 4
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(act))
    # First Maxpooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(tf.keras.layers.Dropout(0.2))

    # Block 5
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(act))
    model.add(tf.keras.layers.Dropout(0.2))

    # Block 6
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.glorot_normal()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(act))
    model.add(tf.keras.layers.Dropout(0.2))

    # Block 7
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.glorot_normal()))
    # Second Maxpooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(act))
    model.add(tf.keras.layers.Dropout(0.2))

    # Block 8
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.glorot_normal()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(act))
    model.add(tf.keras.layers.Dropout(0.2))

    # Block 9
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.glorot_normal()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(act))
    model.add(tf.keras.layers.Dropout(0.2))
    # Third Maxpooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

    # Block 10
    model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.glorot_normal()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(act))
    model.add(tf.keras.layers.Dropout(0.2))

    # Block 11
    model.add(tf.keras.layers.Conv2D(2048, (1, 1), padding='same', kernel_initializer=tf.keras.initializers.glorot_normal()))
    model.add(tf.keras.layers.Activation(act))
    model.add(tf.keras.layers.Dropout(0.2))

    # Block 12
    model.add(tf.keras.layers.Conv2D(256, (1, 1), padding='same', kernel_initializer=tf.keras.initializers.glorot_normal()))
    model.add(tf.keras.layers.Activation(act))
    # Fourth Maxpooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(tf.keras.layers.Dropout(0.2))

    # Block 13
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.glorot_normal()))
    model.add(tf.keras.layers.Activation(act))
    # Fifth Maxpooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

    # Final Classifier
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(nb_classes, name='logits'))
    model.add(tf.keras.layers.Activation(tf.nn.softmax))

    return model