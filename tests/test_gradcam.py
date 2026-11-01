import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
tf1 = tf.compat.v1

from deepracer_viz.gradcam.cam import METHODS, GradCam, blend, layer_cam, select_target_layers  # noqa: E402
from deepracer_viz.model.metadata import ModelMetadata  # noqa: E402
from deepracer_viz.model.model import Model  # noqa: E402

NUM_ACTIONS = 5
DISCRETE_ACTIONS = [{"steering_angle": a, "speed": 1.0} for a in np.linspace(-30, 30, NUM_ACTIONS)]
CONTINUOUS_ACTIONS = {"steering_angle": {"low": -30, "high": 30}, "speed": {"low": 0.5, "high": 4}}


def _build_model(continuous):
    """Build a frozen graph that mimics the DeepRacer shallow CNN layout and op naming."""
    rng = np.random.RandomState(42)
    graph = tf1.Graph()
    with graph.as_default():
        # The first op in the graph is used as model input, the last op as model output.
        x = tf1.placeholder(tf1.float32, [None, 120, 160, 1], name="main_level/agent/main/online/observation")
        x = x / 255.0

        channels = 1
        for i, (filters, kernel, stride) in enumerate([(8, 8, 4), (16, 4, 2), (16, 3, 1)]):
            with tf1.name_scope("main_level/agent/main/online/Conv2d_{}".format(i)):
                w = tf1.constant(rng.normal(0, 0.3, (kernel, kernel, channels, filters)), dtype=tf1.float32)
                b = tf1.constant(rng.normal(0, 0.1, filters), dtype=tf1.float32)
                x = tf1.nn.relu(tf1.nn.bias_add(tf1.nn.conv2d(x, w, [1, stride, stride, 1], "VALID"), b))
            channels = filters

        x = tf1.reshape(x, [-1, int(np.prod(x.shape[1:]))])
        num_outputs = 2 if continuous else NUM_ACTIONS
        w = tf1.constant(rng.normal(0, 0.002, (int(x.shape[-1]), num_outputs)), dtype=tf1.float32)
        logits = tf1.matmul(x, w)
        if continuous:
            tf1.tanh(logits, name="policy")
        else:
            tf1.nn.softmax(logits, name="policy")

    if continuous:
        metadata = ModelMetadata(["FRONT_FACING_CAMERA"], "SHALLOW", "5", "continuous", CONTINUOUS_ACTIONS)
    else:
        metadata = ModelMetadata(["FRONT_FACING_CAMERA"], "SHALLOW", "5", "discrete", DISCRETE_ACTIONS)

    return Model.from_bytes(graph.as_graph_def().SerializeToString(), metadata)


@pytest.fixture
def frame():
    return np.random.RandomState(0).randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def discrete_model():
    return _build_model(continuous=False)


@pytest.fixture
def continuous_model():
    return _build_model(continuous=True)


def test_layer_cam_formulas():
    rng = np.random.RandomState(1)
    activations = np.abs(rng.normal(size=(6, 8, 4)))
    grads = rng.normal(size=(6, 8, 4))

    np.testing.assert_allclose(
        layer_cam("hirescam", activations, grads), np.maximum(np.sum(grads * activations, axis=-1), 0)
    )
    np.testing.assert_allclose(
        layer_cam("layercam", activations, grads), np.sum(np.maximum(grads, 0) * activations, axis=-1)
    )

    # With spatially constant gradients Grad-CAM and HiResCAM are equivalent.
    constant_grads = np.broadcast_to(grads[:1, :1, :], grads.shape)
    np.testing.assert_allclose(
        layer_cam("gradcam", activations, constant_grads), layer_cam("hirescam", activations, constant_grads)
    )

    for method in METHODS:
        cam = layer_cam(method, activations, grads)
        assert cam.shape == (6, 8)
        assert np.all(cam >= 0)

    with pytest.raises(ValueError):
        layer_cam("unknown", activations, grads)


def test_blend_keeps_rgb_channel_order():
    red = np.zeros((4, 4, 3), dtype=np.uint8)
    red[..., 0] = 200

    # A zero map is dark blue in the JET colormap, the red input must stay in the red channel.
    overlay = blend(red, np.zeros((4, 4), dtype=np.float32))
    assert np.all(overlay[..., 0] > overlay[..., 1])
    assert np.all(overlay[..., 0] > overlay[..., 2])
    assert np.all(overlay[..., 2] > 0)


def test_select_target_layers(discrete_model):
    conv_outputs = discrete_model.get_conv_outputs()
    assert len(conv_outputs) == 3

    assert select_target_layers(discrete_model, "hirescam") == conv_outputs[-1:]
    assert select_target_layers(discrete_model, "layercam") == conv_outputs
    assert select_target_layers(discrete_model, "gradcam", "all") == conv_outputs

    with pytest.raises(ValueError):
        select_target_layers(discrete_model, "gradcam", "first")


def test_cam_uses_activated_feature_maps(discrete_model):
    cam = GradCam(discrete_model, select_target_layers(discrete_model, "gradcam"))
    assert [t.op.type for t in cam.target_layers] == ["Relu"]


@pytest.mark.parametrize("method", list(METHODS))
def test_methods_produce_overlay(discrete_model, frame, method):
    with discrete_model.session:
        cam = GradCam(discrete_model, select_target_layers(discrete_model, method), method=method)

        result, heatmap = cam.heatmap(frame)
        assert result.shape == (NUM_ACTIONS,)
        assert heatmap.shape == (120, 160)
        assert heatmap.min() >= 0 and heatmap.max() <= 1

        result, overlay = cam.process(frame)
        assert overlay.shape == frame.shape
        assert overlay.dtype == np.uint8


def test_discrete_defaults_to_selected_action(discrete_model, frame):
    with discrete_model.session:
        layers = select_target_layers(discrete_model, "hirescam")
        result, selected = GradCam(discrete_model, layers, method="hirescam").heatmap(frame)
        chosen = int(np.argmax(result))

        _, explicit = GradCam(discrete_model, layers, method="hirescam", action=chosen).heatmap(frame)
        np.testing.assert_allclose(selected, explicit, atol=1e-6)

        others = [
            GradCam(discrete_model, layers, method="hirescam", action=a).heatmap(frame)[1]
            for a in range(NUM_ACTIONS)
            if a != chosen
        ]
        assert any(not np.allclose(selected, other, atol=1e-3) for other in others)


def test_discrete_action_out_of_range(discrete_model):
    with pytest.raises(ValueError):
        GradCam(discrete_model, select_target_layers(discrete_model, "gradcam"), action=NUM_ACTIONS)


def test_continuous_targets(continuous_model, frame):
    with continuous_model.session:
        layers = select_target_layers(continuous_model, "hirescam")
        steering = GradCam(continuous_model, layers, method="hirescam", continuous_target="steering")
        speed = GradCam(continuous_model, layers, method="hirescam", continuous_target="speed")

        result, steering_map = steering.heatmap(frame)
        _, speed_map = speed.heatmap(frame)

        assert result.shape == (2,)
        assert steering_map.max() > 0 and speed_map.max() > 0
        assert not np.allclose(steering_map, speed_map, atol=1e-3)

        with pytest.raises(ValueError):
            GradCam(continuous_model, layers, action=0)
        with pytest.raises(ValueError):
            GradCam(continuous_model, layers, continuous_target="brake")
