from collections import OrderedDict
from PIL import Image
import torch
from facenet_pytorch.models.mtcnn import MTCNN
import cv2
from data_utils.utils import *


def get_face_detector_model(name='default'):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # default is mtcnn model from facenet_pytorch
    if name == 'default':
        name = 'mtcnn'

    if name == 'mtcnn':
        detector = MTCNN(margin=0, thresholds=[0.85, 0.95, 0.95], device=device)
    else:
        raise Exception("Unknown face detector model.")

    return detector


def locate_face_in_videofile(input_filepath=None, outfile_filepath=None):
    capture = cv2.VideoCapture(input_filepath)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    org_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    org_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    org_res = (org_width, org_height)
    org_fps = int(capture.get(cv2.CAP_PROP_FPS))

    detector = get_face_detector_model()
    frames = list()
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        face_box = list(detector.detect(frame, landmarks=False))[0]
        if face_box is not None:
            for f in range(len(face_box)):
                fc = list(face_box[f])
                cv2.rectangle(frame, (fc[0], fc[1]), (fc[2], fc[3]), (0, 255, 0), 4)
        frames.append(frame)
    create_video_from_images(frames, outfile_filepath, fps=org_fps, res=org_res)


def extract_faces_from_image(image):
    detector = get_face_detector_model()
    face_boxes = list(detector.detect(image, landmarks=False))[0]
    return face_boxes


def extract_faces_from_video(input_videofile, out_dir, batch_size=32, detector=None):
    capture = cv2.VideoCapture(input_videofile)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    id = os.path.splitext(os.path.basename(input_videofile))[0]

    if detector is None:
        detector = get_face_detector_model()

    frames_dict = OrderedDict()
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames_dict[i] = frame

    result = {}
    batches = list()
    frames = list(frames_dict.values())
    num_frames_detected = len(frames)
    for i in range(0, num_frames_detected, batch_size):
        end = i + batch_size
        if end > num_frames_detected:
            end = num_frames_detected
        batches.append((list(range(i, end)), frames[i:end]))

    for j, frames_list in enumerate(batches):
        frame_indices, frame_items = frames_list
        batch_boxes, _ = detector.detect(frame_items, landmarks=False)
        batch_boxes = [b.tolist() if b is not None else None for b in batch_boxes]

        result.update(
            {i: b for i, b in zip(frame_indices, batch_boxes)}
        )

    out_file = os.path.join(out_dir, "{}.json".format(id))
    with open(out_file, "w") as f:
        json.dump(result, f)
