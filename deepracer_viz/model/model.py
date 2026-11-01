import re

import tensorflow.compat.v1 as tf1
from deepracer_viz.model.metadata import ModelMetadata


class Model:
    def __init__(self, session, metadata: ModelMetadata):
        self.metadata = metadata
        self.session = session

        # Resolve input/output once, as ops added later (e.g. gradients) would change the first/last op.
        ops = self.session.graph.get_operations()
        self._input = ops[0].outputs[0]
        self._output = ops[-1].outputs[0]

    def input_size(self):
        input = self.get_model_input()

        height = input.shape[1]
        width = input.shape[2]

        return (width, height)

    def get_conv_outputs(self):
        ops = self.session.graph.get_operations()
        # The optional `_\d+` suffix handles SimApp v6+ models where TF appends a
        # deduplication suffix to op names when weight variables already occupy the
        # base name (e.g. Conv2d_0_1/BiasAdd instead of Conv2d_0/BiasAdd).
        conv_ops = list(filter(lambda x: re.match(r".*Conv2d_\d+(?:_\d+)?/BiasAdd$", x.name), ops))

        return [op.outputs[0] for op in conv_ops]

    def get_model_input(self):
        # First operation output tensor of the loaded graph.
        return self._input

    def get_model_output(self):
        # Last operation output tensor of the loaded graph.
        return self._output

    def get_model_convolutional_output(self):
        # Get last convolutional operator.
        print(self.get_conv_outputs())
        return self.get_conv_outputs()[-1]

    def get_action(self, model_output):
        return self.metadata.action_space.select_action(model_output)

    @classmethod
    def from_file(cls, model_pb_path: str, metadata: ModelMetadata, log_device_placement: bool = False):
        """Load the TensorFlow graph for a model.pb model file.
        Args:
            pbpath (str): Path to the model.pb file
        Raises:
            Exception: If the session cannot be loaded from the model file.
        Returns:
            [tf1.Session]: TensorFlow session object.
        """
        try:
            tf1.reset_default_graph()
            sess = tf1.Session(
                config=tf1.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement)
            )

            with tf1.io.gfile.GFile(model_pb_path, "rb") as f:
                graph_def = tf1.GraphDef()
                graph_def.ParseFromString(f.read())

            sess.graph.as_default()
            tf1.import_graph_def(graph_def, name="")

            return cls(sess, metadata)
        except Exception as e:
            raise Exception("Could not get session for model: {}".format(e))

    @classmethod
    def from_bytes(cls, model_bytes: bytes, metadata: ModelMetadata, log_device_placement: bool = False):
        """Load the TensorFlow graph from a bytes array.
        Args:
            model_bytes (bytes): Bytes array containing the model.pb data
        Raises:
            Exception: If the session cannot be loaded from the model bytes.
        Returns:
            [tf1.Session]: TensorFlow session object.
        """
        try:
            tf1.reset_default_graph()
            sess = tf1.Session(
                config=tf1.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement)
            )

            graph_def = tf1.GraphDef()
            graph_def.ParseFromString(model_bytes)

            sess.graph.as_default()
            tf1.import_graph_def(graph_def, name="")

            return cls(sess, metadata)
        except Exception as e:
            raise Exception("Could not get session for model: {}".format(e))
