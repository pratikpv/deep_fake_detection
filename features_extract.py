import argparse
from models.training_testing import *
from data_utils.utils import *
from features.utils import *
from utils import *
from features.encoders import DeepFakeEncoder
from datetime import datetime


def generate_cnn_video_encodings_main(crops_dir, features_dir):
    print(f'crops_dir: {crops_dir}')
    print(f'features_dir: {features_dir}')

    crops_paths = []
    for d in glob(os.path.join(crops_dir, "*")):
        crops_paths.append(os.path.join(crops_dir, d))

    #generate_cnn_video_encodings(crops_paths[0], features_dir)
    crops_paths = crops_paths[0:10]
    crops_paths_len = len(crops_paths)
    with multiprocessing.Pool(2) as pool:
        jobs = []
        results = []
        for idx in tqdm(range(crops_paths_len), desc="Scheduling jobs"):
            jobs.append(pool.apply_async(generate_cnn_video_encodings,
                                         (crops_paths[idx], features_dir,)
                                         )
                        )

        for job in tqdm(jobs, desc="Encoding video features"):
            results.append(job.get())


def main():
    if args.encode_video_frames:
        print('Encoding video frames for training data')
        generate_cnn_video_encodings_main(get_train_crop_faces_data_path(), get_train_faces_cnn_features_data_path())
        print('Encoding video frames for validation data')
        generate_cnn_video_encodings_main(get_valid_crop_faces_data_path(), get_valid_faces_cnn_features_data_path())
        print('Encoding video frames for test data')
        generate_cnn_video_encodings_main(get_test_crop_faces_data_path(), get_test_faces_cnn_features_data_path())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from DFDC')

    parser.add_argument('--encode_video_frames', action='store_true',
                        help='Use an encoder to get CNN features from each video frames of dataset',
                        default=False)

    args = parser.parse_args()
    main()
