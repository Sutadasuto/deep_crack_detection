import cv2
import numpy as np
import scipy.io
import os

from math import ceil


### Getting image paths
def create_image_paths(dataset_names, dataset_paths):
    paths = np.array([[], []], dtype=np.str)
    for idx, dataset in enumerate(dataset_names):
        dataset_name = dataset
        dataset_path = dataset_paths[idx]
        if dataset_name == "cfd" or dataset_name == "cfd-pruned":
            or_im_paths, gt_paths = paths_generator_cfd(dataset_path)
        elif dataset_name == "aigle-rn":
            or_im_paths, gt_paths = paths_generator_crack_dataset(dataset_path, "AIGLE_RN")
        elif dataset_name == "esar":
            or_im_paths, gt_paths = paths_generator_crack_dataset(dataset_path, "ESAR")
        paths = np.concatenate([paths, [or_im_paths, gt_paths]], axis=-1)
    return paths


def paths_generator_crack_dataset(dataset_path, subset):
    ground_truth_path = os.path.join(dataset_path, "TITS", "GROUND_TRUTH", subset)
    training_data_path = os.path.join(dataset_path, "TITS", "IMAGES", subset)
    images_path, dataset = os.path.split(training_data_path)
    if dataset == "ESAR":
        file_end = ".jpg"
    elif dataset == "AIGLE_RN":
        file_end = "or.png"

    ground_truth_image_paths = sorted([os.path.join(ground_truth_path, f) for f in os.listdir(ground_truth_path)
                                       if not f.startswith(".") and (f.endswith(".png") or f.endswith(".jpg"))],
                                      key=lambda f: f.lower())

    training_image_paths = [os.path.join(training_data_path, "Im_" + os.path.split(f)[-1].replace(".png", file_end)) for
                            f in ground_truth_image_paths]

    return training_image_paths, ground_truth_image_paths


def paths_generator_cfd(dataset_path):
    ground_truth_path = os.path.join(dataset_path, "groundTruthPng")

    if not os.path.exists(ground_truth_path):
        os.makedirs(ground_truth_path)

        ground_truth_image_paths = sorted(
            [os.path.join(dataset_path, "groundTruth", f) for f in os.listdir(os.path.join(dataset_path, "groundTruth"))
             if not f.startswith(".") and f.endswith(".mat")],
            key=lambda f: f.lower())
        for idx, path in enumerate(ground_truth_image_paths):
            mat = scipy.io.loadmat(path)
            img = (mat["groundTruth"][0][0][0] - 1).astype(np.float32)
            cv2.imwrite(path.replace("groundTruth", "groundTruthPng").replace(".mat", ".png"), 255 * img)

    ground_truth_image_paths = sorted([os.path.join(ground_truth_path, f) for f in os.listdir(ground_truth_path)
                                       if not f.startswith(".") and f.endswith(".png")],
                                      key=lambda f: f.lower())

    training_image_paths = [os.path.join(dataset_path, "image", os.path.split(f)[-1].replace(".png", ".jpg")) for f in
                            ground_truth_image_paths]

    return training_image_paths, ground_truth_image_paths


### Loading images for Keras
def get_image(im_path, input_size, normalize=True):
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    if input_size:
        im = cv2.resize(im, input_size)
    if normalize:
        im = ((im - im.min()) / (im.max() - im.min()) - 0.5) / 0.5
    else:
        im = im / 255.0
    im = im[..., None]  # Channels last
    return im


def get_gt_image(gt_path, input_size):
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if input_size:
        gt = cv2.resize(gt, input_size)
    ret, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    gt = (gt / 255)
    n_white = np.sum(gt)
    n_black = gt.shape[0] * gt.shape[1] - n_white
    if n_black < n_white:
        gt = 1 - gt
    return gt


def save_results_on_paths(model, paths, save_to="results"):
    compound_images = test_images_from_paths(model, paths)
    n_im = len(compound_images[0])

    if not os.path.exists(save_to):
        os.makedirs(save_to)
    for im in range(n_im):
        im_name = os.path.split(paths[0][im])[-1]
        cv2.imwrite(os.path.join(save_to, im_name),
                    255 * np.concatenate([compound_images[0][im], compound_images[1][im], compound_images[2][im]],
                                         axis=1))


def test_images_from_paths(model, paths):
    input_shape = tuple(model.input.shape[1:-1])
    predictions = [model.predict(get_image(im_path, None, normalize=True)[None, ...])[0, :, :, 0] for im_path in
                   paths[0, :]]
    gts = [get_gt_image(gt_path, None) for gt_path in paths[1, :]]
    ims = [get_image(im_path, None, normalize=False)[..., 0] for im_path in paths[0, :]]
    return [ims, gts, predictions]


def test_image_generator(paths, input_size, batch_size=1):
    _, n_images = paths.shape
    i = 0
    while True:
        batch_x = []
        for b in range(batch_size):
            if i == n_images:
                i = 0
            im_path = paths[0][i]
            i += 1

            im = get_image(im_path, input_size)
            batch_x.append(im)
        yield np.array(batch_x)


def train_image_generator(paths, input_size, batch_size=1, resize=False):
    _, n_images = paths.shape
    i = 0
    while True:
        batch_x = []
        batch_y = []
        for b in range(batch_size):
            if i == n_images:
                i = 0
                np.random.shuffle(paths.transpose())
            im_path = paths[0][i]
            gt_path = paths[1][i]
            i += 1

            if resize:
                im = get_image(im_path, input_size)
                gt = get_gt_image(gt_path, input_size)
                batch_x.append(im)
                batch_y.append(gt)
            else:
                im = get_image(im_path, None)
                gt = get_gt_image(gt_path, None)
                if input_size:
                    for corner in get_corners(im, input_size):
                        batch_x.append(
                            im[corner[0]:corner[0] + input_size[0], corner[1]:corner[1] + input_size[1], ...])
                        batch_y.append(
                            gt[corner[0]:corner[0] + input_size[0], corner[1]:corner[1] + input_size[1], ...])
                else:
                    batch_x.append(im)
                    batch_y.append(gt)

        yield np.array(batch_x), np.array(batch_y)


def get_corners(im, input_size):
    h, w, c = im.shape
    rows = h / input_size[0]
    cols = w / input_size[0]

    corners = []
    for i in range(ceil(rows)):
        for j in range(ceil(cols)):
            if i + 1 <= rows:
                y = i * input_size[0]
            else:
                y = h - input_size[0]

            if j + 1 <= cols:
                x = j * input_size[1]
            else:
                x = w - input_size[1]

            corners.append([y, x])
    return corners

# create_image_paths(["cfd-pruned", "esar", "aigle-rn"],
#                ["/media/winbuntu/databases/CrackForestDatasetPruned", "/media/winbuntu/databases/CrackDataset",
#                 "/media/winbuntu/databases/CrackDataset"])
