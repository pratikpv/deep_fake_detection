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
