import os, sys
import cv2, numpy as np
from deepracer_viz.gradcam.cam import METHODS, GradCam, select_target_layers
from deepracer_viz.model.metadata import ModelMetadata
from deepracer_viz.model.model import Model

img_path, pb, meta_path, out_dir = sys.argv[1:5]
metadata = ModelMetadata.from_file(meta_path)
model = Model.from_file(pb, metadata)
frame = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

print("input", model.get_model_input().name, model.get_model_input().shape)
print("output", model.get_model_output().name, model.get_model_output().op.type)
print("conv", [t.name for t in model.get_conv_outputs()])

tiles = []
with model.session:
    for method in METHODS:
        cam = GradCam(model, select_target_layers(model, method), method=method)
        result, overlay = cam.process(frame)
        action = int(np.argmax(result))
        steer, speed = model.get_action(result)
        label = "{} | action {} ({:+.1f} deg, p={:.2f})".format(METHODS[method], action, steer, result[action])
        print(label, "layers:", [t.name for t in cam.target_layers])
        bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (640, 30), (0, 0, 0), -1)
        cv2.putText(bgr, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, "{}.png".format(method)), bgr)
        tiles.append(bgr)

    top3 = np.argsort(result)[::-1][:3]
    print("top3", [(int(i), round(float(result[i]), 3)) for i in top3])

orig = cv2.imread(img_path)
cv2.rectangle(orig, (0, 0), (640, 30), (0, 0, 0), -1)
cv2.putText(orig, "Input frame 000389", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
tiles = [orig] + tiles
grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in (0, 3)])
cv2.imwrite(os.path.join(out_dir, "grid.png"), grid)
