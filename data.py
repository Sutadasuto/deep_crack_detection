import cv2
import importlib
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

# Utilities
def manual_padding(image, n_pooling_layers):
    # Assuming N pooling layers of size 2x2 with pool size stride (like in U-net and multiscale U-net), we add the
    # necessary number of rows and columns to have an image fully compatible with up sampling layers.
    divisor = 2 ** n_pooling_layers
    try:
        h, w = image.shape
    except ValueError:
        h, w, c = image.shape
    new_h = divisor * ceil(h / divisor)
    new_w = divisor * ceil(w / divisor)
    if new_h == h and new_w == w:
        return image

    if new_h != h:
        new_rows = np.flip(image[h - new_h:, :, ...], axis=0)
        image = np.concatenate([image, new_rows], axis=0)
    if new_w != w:
        new_cols = np.flip(image[:, w - new_w:, ...], axis=1)
        image = np.concatenate([image, new_cols], axis=1)
    return image


def flipped_version(image, flip_typ):
    if flip_typ is None:
        return image
    elif flip_typ == "h":
        return np.fliplr(image)
    elif flip_typ == "v":
        return np.flipud(image)


def noisy_version(image, noise_typ):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1 * image.max()
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = int(np.ceil(amount * image.size * s_vs_p))
        coords = [np.random.randint(0, i - 1, num_salt)
                  for i in image.shape[:-1]]
        for channel in range(image.shape[-1]):
            channel_coords = tuple(coords + [np.array([channel for i in range(num_salt)])])
            out[channel_coords] = 1

        # Pepper mode
        num_pepper = int(np.ceil(amount * image.size * (1. - s_vs_p)))
        coords = [np.random.randint(0, i - 1, num_pepper)
                  for i in image.shape[:-1]]
        for channel in range(image.shape[-1]):
            channel_coords = tuple(coords + [np.array([channel for i in range(num_pepper)])])
            out[channel_coords] = -1
        return out

    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch) / 4.0
        noisy = image + image * gauss
        return noisy

    elif noise_typ is None:
        return image


def rotated_version(image, angle):
    if angle is None:
        return image

    k = int(angle / 90)
    return np.rot90(image, k)


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


def crop_generator(im, gt, input_size):
    corners = get_corners(im, input_size)
    for corner in corners:
        x = im[corner[0]:corner[0] + input_size[0], corner[1]:corner[1] + input_size[1], ...]
        y = gt[corner[0]:corner[0] + input_size[0], corner[1]:corner[1] + input_size[1], ...]
        yield [x, y]


# Image generators
def get_image(im_path, input_size, normalize=True, rgb=False):
    if rgb:
        im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    else:
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    if input_size:
        im = cv2.resize(im, (input_size[1], input_size[0]))

    if not rgb:
        if normalize:
            im = ((im - im.min()) / (im.max() - im.min()) - 0.5) / 0.5
            im *= -1  # Negative so cracks are brighter
        else:
            im = im / 255.0

    im = manual_padding(im, n_pooling_layers=4)
    if len(im.shape) == 2:
        im = im[..., None]  # Channels last
    return im


def get_gt_image(gt_path, input_size):
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    binary = True if len(np.unique(gt)) == 2 else False
    if input_size:
        gt = cv2.resize(gt, (input_size[1], input_size[0]))
    if binary:
        ret, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    gt = (gt / 255)
    if binary:
        n_white = np.sum(gt)
        n_black = gt.shape[0] * gt.shape[1] - n_white
        if n_black < n_white:
            gt = 1 - gt
    else:
        gt = 1 - gt
    gt = manual_padding(gt, n_pooling_layers=4)
    return gt[..., None]  # Channels last


def test_image_generator(paths, input_size, batch_size=1, rgb_preprocessor=None):
    _, n_images = paths.shape
    rgb = True if rgb_preprocessor else False
    i = 0
    while True:
        batch_x = []
        batch_y = []
        b = 0
        while b < batch_size:
            if i == n_images:
                i = 0
            im_path = paths[0][i]
            gt_path = paths[1][i]

            im = get_image(im_path, input_size, rgb=rgb)
            gt = get_gt_image(gt_path, input_size)
            if rgb:
                batch_x.append(rgb_preprocessor(im))
            else:
                batch_x.append(im)
            batch_y.append(gt)
            b += 1
            i += 1

        yield np.array(batch_x), np.array(batch_y)


def train_image_generator_legacy(paths, input_size, batch_size=1, resize=False, count_samples_mode=False,
                                 rgb_preprocessor=None):
    _, n_images = paths.shape
    rgb = True if rgb_preprocessor else False
    i = 0
    n_samples = 0
    prev_im = False
    while True:
        batch_x = []
        batch_y = []
        b = 0
        while b < batch_size:
            if i == n_images:
                if count_samples_mode:
                    yield n_samples
                i = 0
                n_samples = 0
                np.random.shuffle(paths.transpose())
            im_path = paths[0][i]
            gt_path = paths[1][i]

            if resize:
                im = get_image(im_path, input_size, rgb=rgb)
                gt = get_gt_image(gt_path, input_size)
                if rgb:
                    batch_x.append(rgb_preprocessor(im))
                else:
                    batch_x.append(im)
                batch_y.append(gt)
                n_samples += 1
                b += 1
                i += 1
            else:
                if input_size:
                    if not prev_im:
                        im = get_image(im_path, input_size=None, rgb=rgb)
                        gt = get_gt_image(gt_path, input_size=None)
                        win_gen = crop_generator(im, gt, input_size)
                        prev_im = True
                    try:
                        [im, gt] = next(win_gen)
                        if rgb:
                            batch_x.append(rgb_preprocessor(im))
                        else:
                            batch_x.append(im)
                        batch_y.append(gt)
                        n_samples += 1
                        b += 1
                    except StopIteration:
                        prev_im = False
                        i += 1

                else:
                    im = get_image(im_path, input_size=None)
                    gt = get_gt_image(gt_path, input_size=None)
                    if rgb:
                        batch_x.append(rgb_preprocessor(im))
                    else:
                        batch_x.append(im)
                    batch_y.append(gt)
                    n_samples += 1
                    b += 1
                    i += 1

        if not count_samples_mode:
            yield np.array(batch_x), np.array(batch_y)


def train_image_generator(paths, input_size, batch_size=1, resize=False, count_samples_mode=False,
                          rgb_preprocessor=None, data_augmentation=True):
    _, n_images = paths.shape
    rgb = True if rgb_preprocessor else False

    # All available transformations for data augmentation. 'None' implies no change
    if data_augmentation:
        noises = [None, "gauss", "s&p", "speckle"]
        rotations = [None, 90.0, 180.0, 270.0]
        flips = [None, "h", "v"]
    # This means no noise, no rotation and no flip (i.e. only the original image is provided)
    else:
        noises = [None]
        rotations = [None]
        flips = [None]

    n_transformations = len(noises) * len(rotations) * len(flips)

    i = -1
    j = n_transformations
    prev_im = False

    n_samples = 0

    while True:
        batch_x = []
        batch_y = []
        b = 0
        while b < batch_size:
            if j == n_transformations:
                j = 0
                i += 1

                if i == n_images:
                    if count_samples_mode:
                        yield n_samples
                    i = 0
                    n_samples = 0
                    np.random.shuffle(paths.transpose())

                im_path = paths[0][i]
                gt_path = paths[1][i]

                or_im = get_image(im_path, input_size=None, rgb=rgb)
                or_gt = get_gt_image(gt_path, input_size=None)

                ims = np.zeros((n_transformations, or_im.shape[0], or_im.shape[1], or_im.shape[2]), dtype=or_im.dtype)
                gts = np.zeros((n_transformations, or_gt.shape[0], or_gt.shape[1], or_gt.shape[2]), dtype=or_gt.dtype)
                channel = 0

                for noise in noises:
                    noisy = noisy_version(or_im, noise)

                    for rotation in rotations:
                        rotated = rotated_version(noisy, rotation)
                        rotated_gt = rotated_version(or_gt, rotation)

                        for flip in flips:
                            flipped = flipped_version(rotated, flip)
                            flipped_gt = flipped_version(rotated_gt, flip)
                            if ims[channel, :, :, :].shape == flipped.shape:
                                ims[channel, ...] = flipped
                                gts[channel, ...] = flipped_gt
                            else:
                                flipped = cv2.resize(flipped, (or_im.shape[1], or_im.shape[0]))
                                if len(flipped.shape) == 2:
                                    flipped = flipped[..., None]
                                ims[channel, ...] = flipped
                                flipped_gt = cv2.resize(flipped_gt, (or_gt.shape[1], or_gt.shape[0]))
                                if len(flipped_gt.shape) == 2:
                                    flipped_gt = flipped_gt[..., None]
                                gts[channel, ...] = flipped_gt
                            channel += 1

            im = ims[j, ...]
            gt = gts[j, ...]

            if resize:
                im = cv2.resize(im, (input_size[1], input_size[0]))
                gt = cv2.resize(gt, (input_size[1], input_size[0]))
                if rgb:
                    batch_x.append(rgb_preprocessor(im))
                else:
                    batch_x.append(im)
                batch_y.append(gt)
                n_samples += 1
                b += 1
                j += 1
            else:
                if input_size:
                    if not prev_im:
                        win_gen = crop_generator(im, gt, input_size)
                        prev_im = True
                    try:
                        [im, gt] = next(win_gen)
                        if rgb:
                            batch_x.append(rgb_preprocessor(im))
                        else:
                            batch_x.append(im)
                        batch_y.append(gt)
                        n_samples += 1
                        b += 1
                    except StopIteration:
                        prev_im = False
                        j += 1

                else:
                    if rgb:
                        batch_x.append(rgb_preprocessor(im))
                    else:
                        batch_x.append(im)
                    batch_y.append(gt)
                    n_samples += 1
                    b += 1
                    j += 1

        if not count_samples_mode:
            yield np.array(batch_x), np.array(batch_y)


# Test model on images
def get_preprocessor(model):
    """
    :param model: A Tensorflow model
    :return: A preprocessor corresponding to the model name
    Model name should match with the name of a model from
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/
    This assumes you used a model with RGB inputs as the first part of your model,
    therefore your input data should be preprocessed with the corresponding
    'preprocess_input' function.
    If the model model is not part of the keras applications models, None is returned
    """
    try:
        m = importlib.import_module('tensorflow.keras.applications.%s' % model.name)
        return getattr(m, "preprocess_input")
    except ModuleNotFoundError:
        return None


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


def test_image_from_path(model, input_path, gt_path):
    input_shape = tuple(model.input.shape[1:-1])
    rgb_preprocessor = get_preprocessor(model)
    rgb = True if rgb_preprocessor else False
    if input_shape == (None, None):
        input_shape = None
    if rgb:
        prediction = model.predict(
            rgb_preprocessor(get_image(input_path, input_shape, normalize=True, rgb=rgb))[None, ...])[0, :, :, 0]
    else:
        prediction = model.predict(get_image(input_path, input_shape, normalize=True)[None, ...])[0, :, :, 0]

    if gt_path:
        gt = get_gt_image(gt_path, input_shape)[..., 0]
    input_image = get_image(input_path, None, normalize=False)[..., 0]
    if gt_path:
        return [input_image, gt, prediction]
    return [input_image, None, prediction]


def test_images_from_paths(model, paths):
    input_shape = tuple(model.input.shape[1:-1])
    rgb_preprocessor = get_preprocessor(model)
    rgb = True if rgb_preprocessor else False
    if input_shape == (None, None):
        input_shape = None
    if rgb:
        predictions = [
            model.predict(rgb_preprocessor(get_image(im_path, input_shape, normalize=True, rgb=rgb))[None, ...])[0, :,
            :, 0] for im_path in
            paths[0, :]]
    else:
        predictions = [model.predict(get_image(im_path, input_shape, normalize=True)[None, ...])[0, :, :, 0] for im_path
                       in
                       paths[0, :]]
    gts = [get_gt_image(gt_path, input_shape)[..., 0] for gt_path in paths[1, :]]
    ims = [get_image(im_path, input_shape, normalize=False)[..., 0] for im_path in paths[0, :]]
    return [ims, gts, predictions]


# Compare GT and predictions from images obtained by save_results_on_paths()
def highlight_cracks(or_im, mask):
    highlight_mask = np.zeros(mask.shape, dtype=np.float)
    highlight_mask[np.where(mask >= 128)] = 1.0
    highlight_mask[np.where(mask < 128)] = 0.5
    return or_im * highlight_mask


def compare_masks(gt_mask, pred_mask):
    new_image = np.zeros(gt_mask.shape, dtype=np.float32)
    new_image[..., 2][np.where(pred_mask[..., 0] >= 128)] = 255
    new_image[..., 0][np.where(gt_mask[..., 0] >= 128)] = 255
    new_image[..., 1][np.where((new_image[..., 0] == 255) & (new_image[..., 2] == 255))] = 255
    new_image[..., 0][np.where((new_image[..., 0] == 255) & (new_image[..., 2] == 255))] = 0
    new_image[..., 2][np.where((new_image[..., 0] == 0) & (new_image[..., 1] == 255) & (new_image[..., 2] == 255))] = 0
    return new_image


def analyze_gt_pred(im, gt, pred):
    gt_highlight_cracks = highlight_cracks(im, gt)
    pred_highlight_cracks = highlight_cracks(im, pred)
    comparative_mask = compare_masks(gt, pred)
    white_line_v = 255 * np.ones((comparative_mask.shape[0], 1, 3))
    first_row = np.concatenate(
        (im, white_line_v, gt_highlight_cracks, white_line_v, pred_highlight_cracks, white_line_v, comparative_mask), axis=1)
    white_line_h = 255 * np.ones((1, first_row.shape[1], 3))

    gt_highlight_cracks = highlight_cracks(255-im, gt)
    pred_highlight_cracks = highlight_cracks(255-im, pred)
    second_row = np.concatenate(
        (255-im, white_line_v, gt_highlight_cracks, white_line_v, pred_highlight_cracks, white_line_v, comparative_mask), axis=1)
    return np.concatenate((first_row, white_line_h, second_row), axis=0)


def analyse_resulting_image(image_path):
    or_im = cv2.imread(image_path).astype(np.float)
    h, w, c = or_im.shape
    w = int(w / 3)
    im = or_im[:, :w, :]
    gt = or_im[:, w:2 * w, :]
    pred = or_im[:, 2 * w:, :]
    return analyze_gt_pred(im, gt, pred)


def analyse_resulting_image_folder(folder_path, new_folder=None):
    if not new_folder:
        new_folder = folder_path + "_mask_comparison"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    image_names = sorted([f for f in os.listdir(folder_path)
                          if not f.startswith(".") and (f.endswith(".png") or f.endswith(".jpg"))],
                         key=lambda f: f.lower())

    for name in image_names:
        cv2.imwrite(os.path.join(new_folder, name), analyse_resulting_image(os.path.join(folder_path, name)))


# analyse_resulting_image_folder(
#     "/media/winbuntu/google-drive/Descargas/deep_crack_detection-VGG-encoder/deep_crack_detection/results_training")
