import cv2


def get_frame_count_from_video(video_file):
    cap = cv2.VideoCapture(video_file)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
