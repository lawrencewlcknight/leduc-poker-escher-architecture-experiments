"""TensorFlow/Keras networks used by the ESCHER solver."""

from __future__ import annotations

import tensorflow as tf


def _make_activation(activation: str):
    """Create a fresh activation layer from the experiment config value."""
    activation = str(activation).lower()
    if activation == "leakyrelu":
        return tf.keras.layers.LeakyReLU(alpha=0.2)
    if activation == "relu":
        return tf.keras.layers.ReLU()
    if activation == "elu":
        return tf.keras.layers.ELU()
    if activation == "gelu":
        return tf.keras.layers.Activation(tf.keras.activations.gelu)
    if activation == "swish":
        return tf.keras.layers.Activation(tf.keras.activations.swish)
    if activation == "tanh":
        return tf.keras.layers.Activation("tanh")
    raise ValueError(f"Unsupported activation: {activation!r}")


class ResidualDense(tf.keras.layers.Layer):
    """Dense layer with an optional projection residual connection."""

    def __init__(self, units, *, use_projection=False, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.use_projection = bool(use_projection)
        self.hidden = tf.keras.layers.Dense(self.units, kernel_initializer="he_normal")
        self.projection = None

    def build(self, input_shape):
        input_units = int(input_shape[-1])
        if self.use_projection and input_units != self.units:
            self.projection = tf.keras.layers.Dense(
                self.units, kernel_initializer="he_normal"
            )
        super().build(input_shape)

    def call(self, x):
        residual = self.projection(x) if self.projection is not None else x
        return self.hidden(x) + residual


class SkipDense(ResidualDense):
    """Backwards-compatible same-width residual dense layer."""

    def __init__(self, units, **kwargs):
        super().__init__(units, use_projection=False, **kwargs)


class ActionHeadLayer(tf.keras.layers.Layer):
    """Independent shallow MLP head per action after a shared trunk."""

    def __init__(
        self,
        num_actions: int,
        *,
        head_depth: int,
        head_units: int,
        activation: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.heads = []
        for _ in range(int(num_actions)):
            layers = []
            for _ in range(int(head_depth)):
                layers.append(
                    tf.keras.layers.Dense(
                        int(head_units), kernel_initializer="he_normal"
                    )
                )
                layers.append(_make_activation(activation))
            layers.append(tf.keras.layers.Dense(1))
            self.heads.append(tf.keras.Sequential(layers))

    def call(self, x):
        return tf.concat([head(x) for head in self.heads], axis=-1)


class ScalarHeadLayer(tf.keras.layers.Layer):
    """Shallow scalar MLP head after a shared trunk."""

    def __init__(
        self,
        *,
        head_depth: int,
        head_units: int,
        activation: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        layers = []
        for _ in range(int(head_depth)):
            layers.append(
                tf.keras.layers.Dense(
                    int(head_units), kernel_initializer="he_normal"
                )
            )
            layers.append(_make_activation(activation))
        layers.append(tf.keras.layers.Dense(1))
        self.head = tf.keras.Sequential(layers)

    def call(self, x):
        return self.head(x)


def _center_legal_outputs(values, mask):
    """Centre action outputs over legal actions and keep illegal actions zero."""
    mask = tf.cast(mask, values.dtype)
    legal_count = tf.reduce_sum(mask, axis=-1, keepdims=True)
    safe_count = tf.maximum(legal_count, tf.ones_like(legal_count))
    legal_mean = tf.reduce_sum(values * mask, axis=-1, keepdims=True) / safe_count
    return mask * (values - legal_mean)


def _make_hidden_layer(units: int, prev_units: int, residual_mode: str):
    residual_mode = str(residual_mode).lower()
    if residual_mode in {"same_width", "auto"} and prev_units == units:
        return SkipDense(units)
    if residual_mode == "projection" and prev_units > 0:
        return ResidualDense(units, use_projection=True)
    if residual_mode == "none":
        return tf.keras.layers.Dense(units, kernel_initializer="he_normal")
    if residual_mode not in {"same_width", "auto", "projection"}:
        raise ValueError(f"Unsupported residual_mode: {residual_mode!r}")
    return tf.keras.layers.Dense(units, kernel_initializer="he_normal")


class PolicyNetwork(tf.keras.Model):
    """Average-policy MLP with configurable residual and head structure."""

    def __init__(
        self,
        input_size,
        policy_network_layers,
        num_actions,
        activation="leakyrelu",
        use_layer_norm=True,
        residual_mode="same_width",
        head_depth=0,
        head_units=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input_size = input_size
        self._num_actions = num_actions
        self.activation = _make_activation(activation)
        policy_network_layers = tuple(policy_network_layers)
        if not policy_network_layers:
            raise ValueError("policy_network_layers must contain at least one layer")

        self.softmax = tf.keras.layers.Softmax()
        self.hidden = []
        prev_units = 0
        for units in policy_network_layers[:-1]:
            self.hidden.append(_make_hidden_layer(units, prev_units, residual_mode))
            prev_units = units
        self.normalization = (
            tf.keras.layers.LayerNormalization() if use_layer_norm else None
        )
        self.lastlayer = tf.keras.layers.Dense(
            policy_network_layers[-1], kernel_initializer="he_normal"
        )
        if int(head_depth) > 0:
            self.out_layer = ActionHeadLayer(
                num_actions,
                head_depth=int(head_depth),
                head_units=int(head_units or policy_network_layers[-1]),
                activation=activation,
            )
        else:
            self.out_layer = tf.keras.layers.Dense(num_actions)

    @tf.function
    def call(self, inputs):
        x, mask = inputs
        for layer in self.hidden:
            x = layer(x)
            x = self.activation(x)
        if self.normalization is not None:
            x = self.normalization(x)
        x = self.lastlayer(x)
        x = self.activation(x)
        x = self.out_layer(x)
        x = tf.where(mask == 1, x, -10e20)
        return self.softmax(x)


class RegretNetwork(tf.keras.Model):
    """Regret/advantage approximation network."""

    def __init__(
        self,
        input_size,
        regret_network_layers,
        num_actions,
        activation="leakyrelu",
        use_layer_norm=True,
        residual_mode="same_width",
        head_depth=0,
        head_units=None,
        output_mode="direct",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input_size = input_size
        self._num_actions = num_actions
        self.activation = _make_activation(activation)
        regret_network_layers = tuple(regret_network_layers)
        if not regret_network_layers:
            raise ValueError("regret_network_layers must contain at least one layer")
        valid_output_modes = {"direct", "centered", "dueling"}
        self._output_mode = str(output_mode).lower()
        if self._output_mode not in valid_output_modes:
            raise ValueError(
                f"Unsupported regret output_mode: {output_mode!r}. "
                f"Expected one of {sorted(valid_output_modes)}."
            )

        self.hidden = []
        prev_units = 0
        for units in regret_network_layers[:-1]:
            self.hidden.append(_make_hidden_layer(units, prev_units, residual_mode))
            prev_units = units
        self.normalization = (
            tf.keras.layers.LayerNormalization() if use_layer_norm else None
        )
        self.lastlayer = tf.keras.layers.Dense(
            regret_network_layers[-1], kernel_initializer="he_normal"
        )
        if int(head_depth) > 0:
            self.out_layer = ActionHeadLayer(
                num_actions,
                head_depth=int(head_depth),
                head_units=int(head_units or regret_network_layers[-1]),
                activation=activation,
            )
        else:
            self.out_layer = tf.keras.layers.Dense(num_actions)
        if self._output_mode == "dueling":
            if int(head_depth) > 0:
                self.state_value_layer = ScalarHeadLayer(
                    head_depth=int(head_depth),
                    head_units=int(head_units or regret_network_layers[-1]),
                    activation=activation,
                )
            else:
                self.state_value_layer = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs):
        x, mask = inputs
        for layer in self.hidden:
            x = layer(x)
            x = self.activation(x)
        if self.normalization is not None:
            x = self.normalization(x)
        x = self.lastlayer(x)
        x = self.activation(x)
        action_values = self.out_layer(x)
        if self._output_mode == "direct":
            return mask * action_values
        centred_actions = _center_legal_outputs(action_values, mask)
        if self._output_mode == "centered":
            return centred_actions
        state_value = self.state_value_layer(x)
        return mask * state_value + centred_actions


class ValueNetwork(tf.keras.Model):
    """History-value network used by ESCHER to estimate regret targets."""

    def __init__(
        self,
        input_size,
        val_network_layers,
        activation="leakyrelu",
        use_layer_norm=True,
        residual_mode="same_width",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input_size = input_size
        self.activation = _make_activation(activation)
        val_network_layers = tuple(val_network_layers)
        if not val_network_layers:
            raise ValueError("val_network_layers must contain at least one layer")

        self.hidden = []
        prev_units = 0
        for units in val_network_layers[:-1]:
            self.hidden.append(_make_hidden_layer(units, prev_units, residual_mode))
            prev_units = units
        self.normalization = (
            tf.keras.layers.LayerNormalization() if use_layer_norm else None
        )
        self.lastlayer = tf.keras.layers.Dense(
            val_network_layers[-1], kernel_initializer="he_normal"
        )
        self.out_layer = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs):
        x, _mask = inputs
        for layer in self.hidden:
            x = layer(x)
            x = self.activation(x)
        if self.normalization is not None:
            x = self.normalization(x)
        x = self.lastlayer(x)
        x = self.activation(x)
        return self.out_layer(x)
