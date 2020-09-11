import cv2
import torch
from facenet_pytorch.models.mtcnn import MTCNN

random_setting = '$RANDOM$'

def get_frame_count_from_video(video_file):
    cap = cv2.VideoCapture(video_file)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def create_video_from_images(images, output_video_filename, fps=30, res=(1920, 1080)):
    video = cv2.VideoWriter(output_video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, res)
    for image in images:
        video.write(image)
    video.release()

def locate_face_in_videofile(input_filepath=None, outfile_filepath=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    detector = MTCNN(margin=0, thresholds=[0.85, 0.95, 0.95], device=device)

    capture = cv2.VideoCapture(input_filepath)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    org_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    org_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    org_res = (org_width, org_height)
    org_fps = int(capture.get(cv2.CAP_PROP_FPS))

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
