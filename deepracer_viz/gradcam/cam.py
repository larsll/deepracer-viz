import tensorflow.compat.v1 as tf1
import numpy as np
import cv2

from deepracer_viz.model.model import Model

METHODS = {
    "gradcam": "Grad-CAM",
    "hirescam": "HiResCAM",
    "gradcampp": "Grad-CAM++",
    "xgradcam": "XGrad-CAM",
    "layercam": "LayerCAM",
}

# Index of each continuous action in the model output, matching ContinuousActionSpace.
CONTINUOUS_TARGETS = {"steering": 0, "speed": 1}

LAYER_SELECTIONS = ("last", "all")

_EPS = 1e-7


def layer_cam(method, activations, grads):
    """Compute the (unnormalised) activation map of a single layer.

    Args:
        method (str): One of METHODS.
        activations (np.ndarray): Layer activations of shape (height, width, channels).
        grads (np.ndarray): Gradients of the target score w.r.t. the activations, same shape.
    Returns:
        np.ndarray: Non-negative map of shape (height, width).
    """
    if method == "gradcam":
        # Selvaraju et al. 2017: channel weights are spatially averaged gradients.
        weights = np.mean(grads, axis=(0, 1))
        cam = np.dot(activations, weights)
    elif method == "hirescam":
        # Draelos & Carin 2020: element-wise gradient * activation, no spatial averaging.
        cam = np.sum(grads * activations, axis=-1)
    elif method == "gradcampp":
        # Chattopadhay et al. 2018: higher-order (closed-form) pixel-wise weighting of positive gradients.
        grads_2 = grads**2
        grads_3 = grads**3
        sum_activations = np.sum(activations, axis=(0, 1))
        alpha = grads_2 / (2 * grads_2 + sum_activations * grads_3 + _EPS)
        alpha = np.where(grads != 0, alpha, 0)
        weights = np.sum(np.maximum(grads, 0) * alpha, axis=(0, 1))
        cam = np.dot(activations, weights)
    elif method == "xgradcam":
        # Fu et al. 2020: gradients weighted by normalised activations.
        sum_activations = np.sum(activations, axis=(0, 1))
        weights = np.sum(grads * activations, axis=(0, 1)) / (sum_activations + _EPS)
        cam = np.dot(activations, weights)
    elif method == "layercam":
        # Jiang et al. 2021: element-wise positive gradient * activation.
        cam = np.sum(np.maximum(grads, 0) * activations, axis=-1)
    else:
        raise ValueError("Unsupported CAM method '{}', choose one of: {}".format(method, ", ".join(METHODS)))

    return np.maximum(cam, 0)


def select_target_layers(model: Model, method, layers=None):
    """Select the convolutional layers to explain.

    By default only the last convolutional layer is used, except for LayerCAM which is designed
    to fuse the maps of all convolutional layers.
    """
    if layers is None:
        layers = "all" if method == "layercam" else "last"

    if layers not in LAYER_SELECTIONS:
        raise ValueError(
            "Unsupported layer selection '{}', choose one of: {}".format(layers, ", ".join(LAYER_SELECTIONS))
        )

    conv_outputs = model.get_conv_outputs()
    if not conv_outputs:
        raise ValueError("No convolutional layers found in the model.")

    return conv_outputs if layers == "all" else conv_outputs[-1:]


def _normalize(cam):
    cam = cam - np.min(cam)
    return cam / (np.max(cam) + _EPS)


def _post_activation(tensor):
    # Conv outputs are matched on BiasAdd; CAMs are defined on the (non-negative) activated feature maps.
    consumers = tensor.consumers()
    if len(consumers) == 1 and consumers[0].type in ("Relu", "Relu6", "Elu", "Selu", "LeakyRelu"):
        return consumers[0].outputs[0]
    return tensor


def _pre_softmax(tensor):
    # Explain the logits rather than the softmax probabilities, as recommended by Grad-CAM.
    t = tensor
    while t.op.type == "Identity":
        t = t.op.inputs[0]
    if t.op.type == "Softmax":
        return t.op.inputs[0]
    return tensor


class GradCam:
    def __init__(self, model: Model, target_layers, method="gradcam", action=None, continuous_target="steering"):
        """Class activation mapping for a DeepRacer model.

        Args:
            model (Model): The loaded model.
            target_layers (list): Convolutional layer tensors to compute maps for (see select_target_layers).
            method (str): One of METHODS.
            action (int): Discrete action index to explain. Defaults to the action selected by the model
                for each frame. Not supported for continuous action spaces.
            continuous_target (str): For continuous action spaces, the output to explain ("steering" or "speed").
                Steering is explained in the direction the car is turning, speed towards going faster.
        """
        if method not in METHODS:
            raise ValueError("Unsupported CAM method '{}', choose one of: {}".format(method, ", ".join(METHODS)))

        self.model = model
        self.method = method
        self.action = action
        self.continuous_target = continuous_target
        self.continuous = model.metadata.action_space_type == "continuous"

        self.input_layer = self.model.get_model_input()
        self.output_layer = self.model.get_model_output()
        self.target_layers = [_post_activation(t) for t in target_layers]

        with self.model.session.graph.as_default():
            scores = _pre_softmax(self.output_layer)
            num_outputs = scores.shape[-1]
            depth = num_outputs if num_outputs is not None else tf1.shape(scores)[-1]

            if self.continuous:
                if action is not None:
                    raise ValueError("Explaining a specific action is only supported for discrete action spaces.")
                if continuous_target not in CONTINUOUS_TARGETS:
                    raise ValueError(
                        "Unsupported continuous target '{}', choose one of: {}".format(
                            continuous_target, ", ".join(CONTINUOUS_TARGETS)
                        )
                    )
                target_index = CONTINUOUS_TARGETS[continuous_target]
                if num_outputs is not None and target_index >= num_outputs:
                    raise ValueError("Model output has no '{}' value.".format(continuous_target))

                target_score = tf1.reduce_sum(scores * tf1.one_hot([target_index], depth), axis=-1)
                if continuous_target == "steering":
                    direction = tf1.stop_gradient(
                        tf1.where(target_score >= 0, tf1.ones_like(target_score), -tf1.ones_like(target_score))
                    )
                    target_score = target_score * direction
            else:
                if action is None:
                    target_index = tf1.argmax(scores, axis=-1)
                else:
                    if num_outputs is not None and not 0 <= action < num_outputs:
                        raise ValueError(
                            "Action index {} out of range, the model has {} actions.".format(action, num_outputs)
                        )
                    target_index = [action]

                target_score = tf1.reduce_sum(scores * tf1.one_hot(target_index, depth), axis=-1)

            self.target_grads = tf1.gradients(tf1.reduce_sum(target_score), self.target_layers)

    def _preprocess(self, input):
        input_resized = cv2.resize(input, self.model.input_size())
        input_preprocessed = cv2.cvtColor(input_resized, cv2.COLOR_RGB2GRAY)
        input_frame = np.expand_dims(input_preprocessed, axis=2)

        return input_resized, {self.input_layer: [input_frame]}

    def heatmap(self, input):
        """Compute the model output and the normalised activation map for an RGB frame.

        Returns:
            tuple: (model output, float32 map in [0, 1] with the model input resolution)
        """
        input_resized, feed_dict = self._preprocess(input)
        input_h, input_w = input_resized.shape[:2]

        fetches = [self.output_layer, self.target_layers, self.target_grads]
        result, outs, grads = self.model.session.run(fetches, feed_dict=feed_dict)

        cams = []
        for out, grads_value in zip(outs, grads):
            cam = _normalize(layer_cam(self.method, out[0], grads_value[0]))
            cams.append(cv2.resize(np.float32(cam), (input_w, input_h)))

        cam = _normalize(np.mean(cams, axis=0))

        return result[0, :], np.float32(cam)

    def process(self, input):
        input_resized, _ = self._preprocess(input)
        result, cam = self.heatmap(input)

        cam = blend(input_resized, cam)

        # Scale back to original frame dimensions.
        input_h, input_w = input.shape[:2]
        cam = cv2.resize(cam, (input_w, input_h))

        return result, cam


def blend(input_rgb, cam):
    """Overlay a [0, 1] activation map as a JET heatmap on an RGB image of the same size."""
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = np.float32(heatmap) + np.float32(input_rgb)
    overlay = 255 * overlay / np.max(overlay)

    return np.uint8(overlay)
