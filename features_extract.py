import argparse
from models.training_testing import *
from data_utils.utils import *
from features.utils import *
from utils import *
from features.encoders import DeepFakeEncoder


def generate_cnn_video_encodings_batch(crops_dir, features_dir):
    print(crops_dir)
    crops_paths = []
    for d in glob(os.path.join(crops_dir, "*")):
        crops_paths.append(os.path.join(crops_dir, d))

    print(crops_paths[0])
    generate_cnn_video_encodings(crops_paths[0], features_dir)


def main():
    if args.encode_video_frames:
        print('Encoding video frames for training data')
        generate_cnn_video_encodings_batch(get_train_crop_faces_data_path(), get_train_faces_cnn_features_data_path())
        print('Encoding video frames for validation data')
        generate_cnn_video_encodings_batch(get_valid_crop_faces_data_path(), get_valid_faces_cnn_features_data_path())
        print('Encoding video frames for test data')
        generate_cnn_video_encodings_batch(get_test_crop_faces_data_path(), get_test_faces_cnn_features_data_path())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from DFDC')

    parser.add_argument('--encode_video_frames', action='store_true',
                        help='Use an encoder to get CNN features from each video frames of dataset',
                        default=False)

    args = parser.parse_args()
    main()
