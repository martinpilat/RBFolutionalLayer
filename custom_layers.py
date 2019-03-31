import tensorflow as tf
import numpy as np

class RBFolution(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=(1, 3, 3, 1), padding="VALID", strides=(1, 1, 1, 1), name="RBFolution",
                 dilation_rate=(1,1),
                 ccs_initializer=tf.keras.initializers.RandomUniform(0,1),
                 beta_initilizer=tf.keras.initializers.RandomUniform(0,1)):
        super(RBFolution, self).__init__(name=name)
        self.padding = padding
        self.strides = strides
        self.filters = filters
        self.kernel_size = kernel_size
        self.ccs_initializer = ccs_initializer
        self.beta_initilizer = beta_initilizer
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        self.input_s = input_shape
        self.output_s = self.compute_output_shape(input_shape)

        patch_dim = np.prod(self.kernel_size[1:])

        self.ccs_tensor = self.add_weight("cluster_centers", shape=(patch_dim, self.filters), dtype="float32", initializer=self.ccs_initializer)
        self.beta = self.add_weight("beta", shape=[self.filters], dtype="float32", initializer=self.beta_initilizer)

    def call(self, input, **kwargs):
        return tf.reshape(self.__rbfolution(input), self.output_s)

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = RBFolution.conv_output_length(
                space[i],
                self.kernel_size[1:-1][i],
                padding=self.padding.lower(),
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim.value)
        return (-1,) + tuple(new_space) + (self.filters,)

    def __rbfolution(self, inputs):
        batch_size = tf.shape(inputs)[0]
        patch_dim = np.prod(self.kernel_size[1:])

        # Patches extracted from the images (convolution-like).
        # shape=[batch_size, new_height, new_width, patch_dim] (i. e. individual
        # patches are flattened)
        # tf.extract_image_patches "Only supports ksizes across space" -> we change
        # kernel_size[3] to 1.
        patches = tf.extract_image_patches(
            inputs,
            ksizes=list(self.kernel_size[:3]) + [1],
            strides=self.strides,
            rates=[1, 1, 1, 1],
            padding=self.padding
        )

        patches_shape = tf.shape(patches)
        new_height = patches_shape[1]
        new_width = patches_shape[2]

        # shape=[batch_size, num_patches, patch_dim]
        reshaped_patches = tf.reshape(patches, [batch_size, -1, patch_dim])

        # all_scores[i,j,k] = sum_{l=0}^{patch_dim-1} (
        #     (ccs_tensor[l,k] - reshaped_patches[i,j,l]) ** 2
        # )
        # shape=[batch_size, num_patches, filters]
        all_scores = (
            tf.reduce_sum(tf.square(reshaped_patches), 2, keepdims=True) -
            2 * tf.einsum("aij,jk->aik", reshaped_patches, self.ccs_tensor) +
            tf.reduce_sum(tf.square(self.ccs_tensor), 0, keepdims=True)
        )

        res = tf.reshape(
            tf.exp(tf.multiply(-self.beta, all_scores)),
            [batch_size, new_height, new_width, self.filters],
            name="rbfolution_activation"
        )
        return res

    @staticmethod
    def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
        """Determines output length of a convolution given input length.
        Arguments:
            input_length: integer.
            filter_size: integer.
            padding: one of "same", "valid", "full", "causal"
            stride: integer.
            dilation: dilation rate, integer.
        Returns:
            The output length (integer).
        """
        if input_length is None:
            return None
        assert padding in {'same', 'valid', 'full', 'causal'}
        dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
        if padding in ['same', 'causal']:
            output_length = input_length
        elif padding == 'valid':
            output_length = input_length - dilated_filter_size + 1
        elif padding == 'full':
            output_length = input_length + dilated_filter_size - 1
        return (output_length + stride - 1) // stride