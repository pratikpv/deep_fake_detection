import cv2
import torch
from facenet_pytorch.models.mtcnn import MTCNN
import json
import os
from glob import glob
from pathlib import Path
import random
from subprocess import Popen, PIPE
from utils import *
import shutil

random_setting = '$RANDOM$'


def get_frame_count_from_video(video_file):
    cap = cv2.VideoCapture(video_file)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def compress_video(input_videofile, output_videofile, lvl=None):
    if lvl is None:
        lvl = random.choice([27, 28, 29])
    command = ['ffmpeg', '-i', input_videofile, '-c:v', 'libx264', '-crf', str(lvl),
               '-threads', '1', '-loglevel', 'quiet', '-y', output_videofile]
    try:
        # print(command)
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        process.wait()
        p_out, p_err = process.communicate()
    except Exception as e:
        print_line()
        print("Failed to compress video", str(e))


def in_range(val, min_v, max_v):
    if min_v <= val <= max_v:
        return True
    return False


def adaptive_video_compress(input_videofile, min_file_size, max_file_size, max_tries=4):
    output_videofile = os.path.join(os.path.dirname(input_videofile),
                                    os.path.splitext(os.path.basename(input_videofile))[0] + '_tmp.mp4',
                                    )
    file_size_org = os.path.getsize(input_videofile)
    result = {'input_file': input_videofile,
              'cmprsn_lvl': -1,
              'file_size_org': file_size_org,
              'file_size_comprsd': -1}
    if file_size_org <= max_file_size:
        return result

    cmprsn_lvl = 30
    already_tried_lvls = list()
    for i in range(max_tries):
        if cmprsn_lvl in already_tried_lvls:
            break
        compress_video(input_videofile, output_videofile, lvl=cmprsn_lvl)
        already_tried_lvls.append(cmprsn_lvl)
        f_size = os.path.getsize(output_videofile)
        if f_size > max_file_size:
            # increase compression
            cmprsn_lvl += 1
        elif f_size < min_file_size:
            # decrease compression
            cmprsn_lvl -= 1
        else:
            break

    shutil.move(output_videofile, input_videofile)
    file_size_comprsd = os.path.getsize(input_videofile)
    result['cmprsn_lvl'] = cmprsn_lvl
    result['file_size_comprsd'] = file_size_comprsd
    return result


def create_video_from_images(images, output_video_filename, fps=30, res=(1920, 1080)):
    video = cv2.VideoWriter(output_video_filename, cv2.VideoWriter_fourcc('H', '2', '6', '4'), fps, res)
    for image in images:
        video.write(image)
    video.release()


def extract_images_from_video(input_video_filename, output_folder, res=None):
    os.makedirs(output_folder, exist_ok=True)
    capture = cv2.VideoCapture(input_video_filename)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        out_image_name = os.path.join(output_folder, "{}.jpg".format(i))
        if res is not None:
            frame = cv2.resize(frame, res, interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_image_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])


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


"""
sample entries from metadata.json

{"iqqejyggsm.mp4": {"label": "FAKE", "split": "train", "original": "gzesfubacw.mp4"}
{"ooafcxxfrs.mp4": {"label": "REAL", "split": "train"}

"""


def get_original_video_paths(root_dir, basename=False):
    originals = set()
    originals_v = set()
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        dir = Path(json_path).parent
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "REAL":
                original = k
                originals_v.add(original)
                originals.add(os.path.join(dir, original))
    originals = list(originals)
    originals_v = list(originals_v)
    print(len(originals))
    return originals_v if basename else originals


def get_original_with_fakes(root_dir):
    pairs = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "FAKE":
                pairs.append((original[:-4], k[:-4]))

    return pairs


def get_originals_and_fakes(root_dir):
    originals = []
    fakes = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            if v["label"] == "FAKE":
                fakes.append(k[:-4])
            else:
                originals.append(k[:-4])

    return originals, fakes


def get_all_video_filepaths(root_dir):
    video_filepaths = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        dir = Path(json_path).parent
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            full_path = os.path.join(dir, k)
            video_filepaths.append(full_path)
    return video_filepaths
