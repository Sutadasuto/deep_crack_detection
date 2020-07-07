import tensorflow as tf

from tensorflow.keras.utils import plot_model


def gram_matrix(input_tensor):
    # Einsum allows defining Tensors by defining their element-wise computation.
    # This computation is defined by equation, a shorthand form based on Einstein summation.
    # G[b, c, d] = sum_i sum_j F[b, i, j, c] * F[b, i, j, d]
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def gram_matrix_masked(input_tensor, mask):
    input_tensor = tf.multiply(input_tensor, mask)
    return gram_matrix(input_tensor)


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    plot_model(vgg, 'vgg19_diagram.png')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def __call__(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}
        # return {'content': content_outputs, 'style': style_outputs}


class StyleContentClassModel(StyleContentModel):

    def __call__(self, inputs, mask):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        if mask:
            # Expects mask as binary [0,1]
            style_outputs = [gram_matrix_masked(style_output, tf.image.resize(mask, tf.shape(style_output)[1:-1]))
                             for style_output in style_outputs]
            content_outputs = [tf.multiply(content_output, tf.image.resize(mask, tf.shape(content_output)[1:-1]))
                               for content_output in content_outputs]
        else:
            style_outputs = [gram_matrix(style_output)
                             for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}