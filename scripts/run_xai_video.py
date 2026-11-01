import glob, json, os, subprocess, sys
import cv2, numpy as np
from deepracer_viz.gradcam.cam import METHODS, GradCam, select_target_layers
from deepracer_viz.model.metadata import ModelMetadata
from deepracer_viz.model.model import Model

img_dir, pb, meta_path, out_path = sys.argv[1:5]
fps = 30

metadata = ModelMetadata.from_file(meta_path)
model = Model.from_file(pb, metadata)
images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

# Reference inference results recorded with the car's TFLite model, keyed by file name.
reference = {}
for results_file in glob.glob(os.path.join(img_dir, "results-*.json")):
    for frame in json.load(open(results_file))["frames"].values():
        res = frame["tflite"]["results"]
        reference[os.path.basename(frame["filename"])] = np.array([res[str(i)] for i in range(len(res))])


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


ffmpeg = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x960", "-r", str(fps),
     "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
     out_path],
    stdin=subprocess.PIPE,
)

max_diff, mismatches = 0.0, 0
with model.session:
    cams = {m: GradCam(model, select_target_layers(model, m), method=m) for m in METHODS}
    for i, path in enumerate(images):
        name = os.path.basename(path)
        bgr = cv2.imread(path)
        if bgr.shape[:2] != (480, 640):
            bgr = cv2.resize(bgr, (640, 480))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        tiles = []
        for method, cam in cams.items():
            result, overlay = cam.process(rgb)
            tile = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            label(tile, METHODS[method])
            tiles.append(tile)

        action = int(np.argmax(result))
        steer, speed = model.get_action(result)
        if name in reference:
            max_diff = max(max_diff, float(np.max(np.abs(reference[name] - result))))
            mismatches += int(np.argmax(reference[name]) != action)

        label(bgr, "{}  action {} ({:+.1f} deg, p={:.2f})".format(name, action, steer, result[action]))
        # Probability bar over the steering actions, highlighting the selected one.
        n = len(result)
        bw = 600 // n
        for a, p in enumerate(result):
            x0 = 20 + (n - 1 - a) * bw  # positive steering (left) drawn on the left
            h = int(80 * p)
            color = (0, 200, 255) if a == action else (230, 230, 230)
            cv2.rectangle(bgr, (x0, 470 - h), (x0 + bw - 3, 470), color, -1)
        cv2.rectangle(bgr, (15, 385), (25 + n * bw, 474), (0, 0, 0), 1)

        tiles = [bgr] + tiles
        grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        ffmpeg.stdin.write(grid.tobytes())
        if i % 200 == 0:
            print("frame", i, "/", len(images), flush=True)

ffmpeg.stdin.close()
ffmpeg.wait()
print("done", len(images), "frames; vs TFLite reference: max |p diff| = {:.4f}, argmax mismatches = {}".format(max_diff, mismatches))
