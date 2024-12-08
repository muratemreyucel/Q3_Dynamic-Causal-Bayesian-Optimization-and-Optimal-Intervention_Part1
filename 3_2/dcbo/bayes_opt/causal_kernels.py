import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfk = tfp.math.psd_kernels


class CausalRBF(tfk.PositiveSemidefiniteKernel):
    """
    Custom Radial Basis Function (RBF) kernel with causal adjustments.

    This kernel incorporates a variance adjustment mechanism and rescaling for causal inference tasks.
    """

    def __init__(
        self,
        input_dim,
        variance_adjustment,
        variance=1.0,
        lengthscale=1.0,
        rescale_variance=1.0,
        name="CausalRBF",
    ):
        """
        Initialize the Causal RBF kernel.

        Parameters
        ----------
        input_dim : int
            Number of input dimensions.
        variance_adjustment : Callable
            Function to compute variance adjustments for causal effects.
        variance : float, optional
            Variance parameter, by default 1.0.
        lengthscale : float, optional
            Lengthscale parameter, by default 1.0.
        rescale_variance : float, optional
            Scaling factor for variance, by default 1.0.
        name : str, optional
            Kernel name, by default "CausalRBF".
        """
        super().__init__(feature_ndims=1, name=name)
        self.input_dim = input_dim
        self.variance_adjustment = variance_adjustment
        self.variance = tf.Variable(variance, dtype=tf.float64, name="variance")
        self.lengthscale = tf.Variable(lengthscale, dtype=tf.float64, name="lengthscale")
        self.rescale_variance = tf.Variable(rescale_variance, dtype=tf.float64, name="rescale_variance")

    def _apply(self, x1, x2):
        """
        Compute the kernel matrix between two inputs.

        Parameters
        ----------
        x1 : tf.Tensor
            First input tensor.
        x2 : tf.Tensor
            Second input tensor.

        Returns
        -------
        tf.Tensor
            Kernel matrix.
        """
        diff = tf.expand_dims(x1, -2) - tf.expand_dims(x2, -3)
        scaled_square_distances = tf.reduce_sum(tf.square(diff) / tf.square(self.lengthscale), axis=-1)
        base_kernel = self.variance * tf.exp(-0.5 * scaled_square_distances)

        # Variance adjustment terms
        value_diagonal_x1 = self.variance_adjustment(x1)
        value_diagonal_x2 = self.variance_adjustment(x2)
        additional_matrix = tf.sqrt(value_diagonal_x1)[:, None] * tf.sqrt(value_diagonal_x2)[None, :]

        return base_kernel + self.rescale_variance * additional_matrix

    def matrix(self, x1, x2):
        """
        Wrapper to call the kernel computation.

        Parameters
        ----------
        x1 : tf.Tensor
            First input tensor.
        x2 : tf.Tensor
            Second input tensor.

        Returns
        -------
        tf.Tensor
            Computed kernel matrix.
        """
        return self._apply(x1, x2)

    def diag(self, x):
        """
        Compute the diagonal of the kernel matrix.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Diagonal of the kernel matrix.
        """
        base_diag = tf.fill(tf.shape(x)[:-1], self.variance)
        variance_adjustment_diag = self.variance_adjustment(x)
        return base_diag + self.rescale_variance * variance_adjustment_diag

    def __call__(self, x1, x2=None):
        """
        Callable interface for the kernel.

        Parameters
        ----------
        x1 : tf.Tensor
            First input tensor.
        x2 : tf.Tensor, optional
            Second input tensor, by default None (implies x2=x1).

        Returns
        -------
        tf.Tensor
            Kernel matrix or vector.
        """
        if x2 is None:
            return self.diag(x1)
        return self.matrix(x1, x2)
