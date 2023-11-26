import tensorflow as tf
import math

# --------------------------------------------------------------------------------------------

class FADLSelectionLayer(tf.keras.layers.Layer):
    """
    Custom TensorFlow Keras layer for time selection.

    Args:
        num_outputs (int): Number of output units.
        regularization (float, optional): Regularization strength. Defaults to 0.01.
        **kwargs: Additional layer arguments.
    """
    def __init__(self, num_outputs: int, regularization: float = 0.01, **kwargs):
        super(FADLSelectionLayer, self).__init__( **kwargs)
        self.mask = None
        self.num_outputs = num_outputs              # Numero de salidas del modelo completo (para usarlo para regularizacion)
        self.regularization = regularization

    # --------------------------------------------------
    # Initialize layer weights

    def build(self, input_shape: tuple):
        """
        Initialize layer weights to 0.01.

        Args:
            input_shape (tuple): Input shape.
        """
        if len(input_shape)>2:
            shape = [int(input_shape[-2]), int(input_shape[-1])]
        else:
            shape = [int(input_shape[-1])]

        self.mask = self.add_weight("kernel",
                                      shape=shape,
                                      initializer=tf.keras.initializers.Constant(value=0.01),
                                      regularizer=self.custom_regularizer)
        
    # --------------------------------------------------
    # Calculate layer weights

    def get_mask(self):
        return binary_sigmoid_unit(tf.expand_dims(self.mask, 0))[0]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.multiply(inputs, self.get_mask())                     # Multiplicacion elemento a elemento,
                                                                        # así cada atributo de entrada tiene asociada una única neurona
                                                                        # y el peso de la conexión mide la importancia de ese atributo
                                                                        # La transformación de pesos de esta capa es aplicar sigmoide para
                                                                        # que los pesos estén en el rango [0,1] mediante la transformacion hardsigmoide y el redondeo
    
    # --------------------------------------------------
    # Regularization: forze not to select many features 
    # by intentionally penalizing the weights of the selection layer near 0
    # and so importan features should struggle in order to make themselves bigger     
    
    def custom_regularizer(self, weights: tf.Tensor) -> tf.Tensor:
        """
        Custom regularization function for the layer.

        Args:
            weights (tf.Tensor): Layer weights.

        Returns:
            tf.Tensor: Regularization term.
        """
        weight = self.regularization/(10**math.log2(self.num_outputs))
        return tf.reduce_sum(weight * binary_sigmoid_unit(weights))
       

# --------------------------------------------------------------------------------------------
# Auxiliar functions

def binary_sigmoid_unit(x: tf.Tensor) -> tf.Tensor:
    """
    Compute the binary sigmoid unit using rounding with gradient propagation.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Binary sigmoid unit output.
    """
    return round_through(hard_sigmoid(x))


def hard_sigmoid(x: tf.Tensor) -> tf.Tensor:
    """
    Compute the hard sigmoid activation function.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Output tensor with values in the range [0, 1].
    """
    return tf.clip_by_value((x+1.)/2., 0, 1)                     


def round_through(x: tf.Tensor) -> tf.Tensor:
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    a op that behave as f(x) in forward mode,
    but as g(x) in the backward mode.
    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Rounded tensor with gradient propagation.
    '''
    rounded = tf.round(x)

    # diferencia_redondeo = x_redondeado - x
    # x_redondeado = x + diferencia_redondeo = x + (x_redondeado - x)
    # Para que sea diferenciable no permitimos la propagacion de gradientes en (x_redondeado - x)
    return x + tf.stop_gradient(rounded-x) 
  
