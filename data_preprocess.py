import multiprocessing
from functools import partial
from tqdm import tqdm
import data_utils.augmentation as augmentation
import data_utils.distractions as distractions
import os
import cv2
from utils import *
from glob import glob
from data_utils.utils import *
import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle


def test_data_augmentation(input_file, output_folder):
    """
    Test all kinds of data augmentation. Applies supported augmentation to input_file
    and save individual output videos in output_folder
    :param input_file:
    :param output_folder:
    :return:
    """
    os.makedirs(output_folder, exist_ok=True)

    augmentation_methods = augmentation.get_supported_augmentation_methods()
    augmentation_methods.remove('noise')
    augmentation_param = [augmentation.get_augmentation_setting_by_type(m) for m in augmentation_methods]
    noise_methods = augmentation.get_supported_noise_types()
    noise_methods_param = [augmentation.get_noise_param_setting(m) for m in noise_methods]
    augmentation_methods.extend(['noise'] * len(noise_methods))
    augmentation_param.extend(noise_methods_param)

    augmentation_plan = list(zip(augmentation_methods, augmentation_param))

    out_id = os.path.splitext(os.path.basename(input_file))[0]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for aug in augmentation_plan:
            aug_func, aug_param = aug
            if aug_func == 'noise':
                suffix = out_id + '_' + aug_func + '_' + aug_param['noise_type']
            else:
                suffix = out_id + '_' + aug_func
            outfile = os.path.join(output_folder, suffix + '.mp4')
            jobs.append(pool.apply_async(augmentation.apply_augmentation_to_videofile,
                                         (input_file, outfile,),
                                         dict(augmentation=aug_func, augmentation_param=aug_param)
                                         )
                        )

        for job in tqdm(jobs, desc="Applying augmentation"):
            results.append(job.get())


def locate_faces(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    input_filepath_list = glob(input_root + '/*.mp4')
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for input_filepath in input_filepath_list:
            output_filepath = os.path.join(output_root, os.path.basename(input_filepath))
            jobs.append(pool.apply_async(augmentation.locate_face_in_videofile,
                                         (input_filepath, output_filepath,),
                                         )
                        )

        for job in tqdm(jobs, desc="Tracking faces"):
            results.append(job.get())


def test_data_distraction(input_file, output_folder):
    """
    Test all kinds of data distraction. Applies supported distraction to input_file
    and save individual output videos in output_folder
    :param input_file:
    :param output_folder:
    :return:
    """
    os.makedirs(output_folder, exist_ok=True)

    distraction_methods = distractions.get_supported_distraction_methods()
    distraction_params = [distractions.get_distractor_setting_by_type(m) for m in distraction_methods]
    distraction_plan = list(zip(distraction_methods, distraction_params))
    out_id = os.path.splitext(os.path.basename(input_file))[0]
    pprint(distraction_plan)

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for distract in distraction_plan:
            distract_func, distract_param = distract
            outfile = os.path.join(output_folder, out_id + '_' + distract_func + '.mp4')
            jobs.append(pool.apply_async(distractions.apply_distraction_to_videofile,
                                         (input_file, outfile,),
                                         dict(distraction=distract_func, distraction_param=distract_param)
                                         ))

        for job in tqdm(jobs, desc="Applying distraction"):
            results.append(job.get())

    return results


def generate_poster(org_file, aug_output_folder, frames_count=10, out_res=None):
    id = os.path.splitext(os.path.basename(org_file))[0]
    org_out_images = os.path.join(aug_output_folder, id + '_original')
    # extract_images_from_video(org_file, org_out_images)
    sub_fol = []
    for f in glob(aug_output_folder + '/*'):
        if os.path.isdir(f):
            sub_fol.append(f)
    sub_fol_count = len(sub_fol)
    print(f'sub_fol_count {sub_fol_count}')
    result_grid_filename = os.path.join(aug_output_folder, 'poster.jpg')
    result_figsize_resolution = 40  # 1 = 100px

    total_frames = 300
    step = int(total_frames / frames_count)
    images_list = ['{}.jpg'.format(i) for i in range(0, total_frames, step)]
    images_count = len(images_list)
    print('Images: ', images_list)
    print('Images count: ', images_count)

    out_res_per_image = 1920, 1080
    result_figsize_resolution = out_res_per_image[0] * images_count, out_res_per_image[1] * images_count
    print(f'result_figsize_resolution = {result_figsize_resolution}')
    fig, axes = plt.subplots(sub_fol_count, frames_count,
                             figsize=(40, 40)
                             )

    fol_n_replace_str = id + '_'
    for i, folder in enumerate(sub_fol):
        fol_n = folder.replace(fol_n_replace_str, '')
        axes[i, 0].set_ylabel(fol_n)
        for j, frame in enumerate(images_list):
            plt_image = plt.imread(folder + '/' + frame)
            axes[i, j].imshow(plt_image)
            axes[i, j].axis('off')
            """
            if j == 0:
                n = os.path.basename(folder)
                # axes[i, j].set_title(n, loc='left',y=True)
                axes[i, j].set_ylabel(n)
            else:
                axes[i, j].set_title('')
            """
            if i == 0:
                fnum = frame.replace('.jpg', '')
                axes[0, j].set_title('frame {}'.format(fnum))
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)
            # print(folder + '/' + frame)
    plt.savefig(result_grid_filename)

    print(f'generated {result_grid_filename}')


def generate_data_augmentation_plan(root_dir=None, plan_pkl_file=None, plan_txt_file=None):
    # Apply various kinds of data augmentation to 30 % of whole training set.
    # Sample without replacement in this case and each below case.
    # Form these randomly selected video files,
    #
    # apply distractions to 35% of files
    # distractions and random noise to 35%
    # distractions, random noise, and augmentation to 15%
    # noise to 5%
    # augmentation and noise to 5%
    # augmentation to 5%
    #

    results = get_all_video_filepaths(root_dir=root_dir)
    polulation_size = len(results)
    random.shuffle(results)

    samples_size = int(polulation_size * 0.30)
    distr_samples_size = int(samples_size * 0.35)
    dist_noise_sample_size = int(samples_size * 0.35)
    dist_noise_aug_size = int(samples_size * 0.15)
    noise_sample_size = int(samples_size * 0.05)
    aug_noise_sample_size = int(samples_size * 0.05)
    aug_sample_size = int(samples_size * 0.05)

    print(f'Total data count {polulation_size}')
    print(f'Total samples count {samples_size}')

    samples = random.sample(results, samples_size)

    distr_samples = random.sample(samples, distr_samples_size)
    samples = list(filter(lambda i: i not in distr_samples, samples))

    dist_noise_samples = random.sample(samples, dist_noise_sample_size)
    samples = list(filter(lambda i: i not in dist_noise_samples, samples))

    dist_noise_aug_samples = random.sample(samples, dist_noise_aug_size)
    samples = list(filter(lambda i: i not in dist_noise_aug_samples, samples))

    noise_samples = random.sample(samples, noise_sample_size)
    samples = list(filter(lambda i: i not in noise_samples, samples))

    aug_noise_samples = random.sample(samples, aug_noise_sample_size)
    samples = list(filter(lambda i: i not in aug_noise_samples, samples))

    aug_samples = random.sample(samples, aug_sample_size)

    assert len(distr_samples) == distr_samples_size
    assert len(dist_noise_samples) == dist_noise_sample_size
    assert len(dist_noise_aug_samples) == dist_noise_aug_size
    assert len(noise_samples) == noise_sample_size
    assert len(aug_noise_samples) == aug_noise_sample_size
    assert len(aug_samples) == aug_sample_size

    distr_samples_exec_plan = []
    for i in distr_samples:
        plan = list()
        plan.append({'distraction': distractions.get_random_distractor()})
        entry = i, plan
        distr_samples_exec_plan.append(entry)
    # pprint(distr_samples_exec_plan)

    dist_noise_samples_exec_plan = []
    for i in dist_noise_samples:
        plan = list()
        plan.append({'distraction': distractions.get_random_distractor()})
        noise_type = augmentation.get_random_noise_type()
        noise_param = augmentation.get_noise_param_setting(noise_type)
        plan.append({'augmentation': ('noise', noise_param)})
        entry = i, plan
        dist_noise_samples_exec_plan.append(entry)
    # pprint(dist_noise_samples_exec_plan)

    dist_noise_aug_exec_plan = []
    for i in dist_noise_aug_samples:
        plan = list()
        plan.append({'distraction': distractions.get_random_distractor()})
        noise_type = augmentation.get_random_noise_type()
        noise_param = augmentation.get_noise_param_setting(noise_type)
        plan.append({'augmentation': ('noise', noise_param)})
        plan.append({'augmentation': augmentation.get_random_augmentation(avoid_noise=True)})
        entry = i, plan
        dist_noise_aug_exec_plan.append(entry)
    # pprint(dist_noise_aug_exec_plan)

    noise_samples_exe_plan = []
    for i in noise_samples:
        plan = list()
        noise_type = augmentation.get_random_noise_type()
        noise_param = augmentation.get_noise_param_setting(noise_type)
        plan.append({'augmentation': ('noise', noise_param)})
        entry = i, plan
        noise_samples_exe_plan.append(entry)
    # pprint(noise_samples_exe_plan)

    aug_noise_samples_exec_plan = []
    for i in aug_noise_samples:
        plan = list()
        plan.append({'augmentation': augmentation.get_random_augmentation(avoid_noise=True)})
        noise_type = augmentation.get_random_noise_type()
        noise_param = augmentation.get_noise_param_setting(noise_type)
        plan.append({'augmentation': ('noise', noise_param)})
        entry = i, plan
        aug_noise_samples_exec_plan.append(entry)
    # pprint(aug_noise_samples_exec_plan)

    aug_samples_exec_plan = []
    for i in aug_samples:
        plan = list()
        plan.append({'augmentation': augmentation.get_random_augmentation(avoid_noise=True)})
        entry = i, plan
        aug_samples_exec_plan.append(entry)
    # pprint(aug_samples_exec_plan)

    data_augmentation_plan = list()
    data_augmentation_plan.extend(distr_samples_exec_plan)
    data_augmentation_plan.extend(dist_noise_samples_exec_plan)
    data_augmentation_plan.extend(dist_noise_aug_exec_plan)
    data_augmentation_plan.extend(noise_samples_exe_plan)
    data_augmentation_plan.extend(aug_noise_samples_exec_plan)
    data_augmentation_plan.extend(aug_samples_exec_plan)

    print(f'Saving plan to {plan_pkl_file}')
    with open(plan_pkl_file, 'wb') as f:
        pickle.dump(data_augmentation_plan, f)

    print(f'Saving plan to {plan_txt_file}')
    with open(plan_txt_file, 'w') as f:
        for listitem in data_augmentation_plan:
            f.write('%s\n' % str(listitem))

    return data_augmentation_plan




def execute_data_augmentation_plan(data_augmentation_plan_filename, metadata_folder):
    with open(data_augmentation_plan_filename, 'rb') as f:
        data_augmentation_plan = pickle.load(f)

    # i wish to apply plan in sequence for each file, but if the file has multiple plans
    # later plans may get started before eariler one finishes.
    # to workaround this, execute plan[0] for each file, then plan[1] for each file
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for plan_id in range(5):
            for filename, plan in data_augmentation_plan:
                if len(plan) > plan_id:
                    p = plan[plan_id]

                    if 'augmentation' in p.keys():
                        distr = p['augmentation']
                        jobs.append(pool.apply_async(augmentation.apply_augmentation_to_videofile,
                                                     (filename, filename,),
                                                     dict(augmentation=distr[0], augmentation_param=distr[1])
                                                     )
                                    )
                    elif 'distraction' in p.keys():
                        distr = p['distraction']
                        jobs.append(pool.apply_async(distractions.apply_distraction_to_videofile,
                                                     (filename, filename,),
                                                     dict(distraction=distr[0], distraction_param=distr[1])
                                                     ))

        for job in tqdm(jobs, desc="Executing data augmentation plan"):
            r = job.get()
            df = pd.DataFrame.from_dict(r, orient='index')
            rfilename = os.path.basename(r['input_file']) + \
                        '_' + str(datetime.now().strftime("%d-%b-%Y_%H_%M_%S")) + '.csv'
            df.to_csv(os.path.join(metadata_folder, rfilename))
            results.append(r)

    return results


def main():
    data_root_dir = '/home/therock/dfdc/train/'
    generate_data_augmentation_plan(data_root_dir, get_data_aug_plan_pkl_filename(), get_data_aug_plan_txt_filename())
    execute_data_augmentation_plan(get_data_aug_plan_pkl_filename(), get_aug_metadata_folder())


if __name__ == '__main__':
    main()
