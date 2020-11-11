import argparse
from models.training import *
from data_utils.utils import *
from features.utils import *
from features.face_xray import *
from utils import *
from features.encoders import DeepFakeEncoder
from datetime import datetime


def generate_cnn_video_encodings_main(crops_dir, features_dir):
    print(f'crops_dir: {crops_dir}')
    print(f'features_dir: {features_dir}')

    crops_paths = []
    for d in glob(os.path.join(crops_dir, "*")):
        crops_paths.append(os.path.join(crops_dir, d))

    # generate_cnn_video_encodings(crops_paths[0], features_dir)
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


def generate_optical_flow_data_batch(crop_faces_data_path, optical_flow_data_path, optical_png_data_path):
    print(f'Crops dir {crop_faces_data_path}')
    print(f'Optical flow dir {optical_flow_data_path}')
    print(f'Optical png dir {optical_png_data_path}')

    encoder_name = get_default_cnn_encoder_name()
    imsize = encoder_params[encoder_name]["imsize"]

    video_ids_path = glob(crop_faces_data_path + "/*")
    video_ids_len = len(video_ids_path)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for vidx in tqdm(range(video_ids_len), desc="Scheduling jobs"):
            jobs.append(pool.apply_async(generate_optical_flow_data,
                                         (crop_faces_data_path, optical_flow_data_path, optical_png_data_path,
                                          os.path.basename(video_ids_path[vidx]), imsize)
                                         )
                        )

        for job in tqdm(jobs, desc="Generating optical flow data"):
            results.append(job.get())


def generate_xray_batch_dfdc():
    pairs = get_training_original_with_fakes(get_train_data_path())
    pairs_len = len(pairs)
    xray_basedir = get_xray_path()
    csv_file = get_xray_metadata_csv()
    print(f'Train Crops dir {get_train_crop_faces_data_path()}')
    print(f'Xray data dir {xray_basedir}')
    print(f'Xray metadata csv {csv_file}')

    results = []
    df = pd.DataFrame(columns=['real_image', 'fake_image', 'xray_image'])

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        for pid in tqdm(range(pairs_len), desc="Scheduling jobs"):
            item = pairs[pid]
            jobs.append(pool.apply_async(gen_xray_per_folder, (item[0], item[1], xray_basedir,)))

        for job in tqdm(jobs, desc="Generating xrays"):
            results.append(job.get())

    for r in tqdm(results, desc='Consolidating results'):
        if r is not None:
            df = df.append(r, ignore_index=True)

    df.set_index('real_image', inplace=True)
    if os.path.isfile(csv_file):
        df.to_csv(csv_file, mode='a', header=False)
    else:
        df.to_csv(csv_file)


def main():
    if args.encode_video_frames:
        print('Encoding video frames for training data')
        generate_cnn_video_encodings_main(get_train_crop_faces_data_path(), get_train_faces_cnn_features_data_path())
        print('Encoding video frames for validation data')
        generate_cnn_video_encodings_main(get_valid_crop_faces_data_path(), get_valid_faces_cnn_features_data_path())
        print('Encoding video frames for test data')
        generate_cnn_video_encodings_main(get_test_crop_faces_data_path(), get_test_faces_cnn_features_data_path())

    if args.gen_optic_data:
        print(f'Generating optical flow data for training samples')
        generate_optical_flow_data_batch(get_train_crop_faces_data_path(), get_train_optical_flow_data_path(),
                                         get_train_optical_png_data_path())
        print(f'Generating optical flow data for valid samples')
        generate_optical_flow_data_batch(get_valid_crop_faces_data_path(), get_valid_optical_flow_data_path(),
                                         get_valid_optical_png_data_path())
        print(f'Generating optical flow data for test samples')
        generate_optical_flow_data_batch(get_test_crop_faces_data_path(), get_test_optical_flow_data_path(),
                                         get_test_optical_png_data_path())

    if args.gen_xray:
        print('Generating face x-ray')
        generate_xray_batch_dfdc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from DFDC')

    parser.add_argument('--encode_video_frames', action='store_true',
                        help='Use an encoder to get CNN features from each video frames of dataset',
                        default=False)

    parser.add_argument('--gen_optic_data', action='store_true',
                        help='Generate optical flow data',
                        default=False)

    parser.add_argument('--gen_xray', action='store_true',
                        help='Generate xray',
                        default=False)

    args = parser.parse_args()
    main()
