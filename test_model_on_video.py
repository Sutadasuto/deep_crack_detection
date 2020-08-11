import argparse
import cv2
import importlib
import numpy as np
import os
import re
import time

from distutils.util import strtobool
from tensorflow.keras.models import model_from_json
from json.decoder import JSONDecodeError
from queue import LifoQueue, Empty
from threading import Thread

import data

from models.available_models import get_models_dict

models_dict = get_models_dict()


class DeviceVideoStream:
    # Idea from: https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
    def __init__(self, device, stack_size=0):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(device)
        self.stopped = False

        # initialize the queue used to store frames read from
        # the video file
        self.stack = LifoQueue(maxsize=stack_size)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:

            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return

            # otherwise, ensure the queue has room in it
            if not self.stack.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return

                # add the frame to the queue
                self.stack.put(frame)

    def read(self):
        # return next frame in the queue
        last_frame = self.stack.get()
        stack_size = self.stack.qsize() + 1
        while not self.stack.empty():
            try:
                self.stack.get(False)
            except Empty:
                continue
            self.stack.task_done()
        return last_frame, stack_size

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def main(args):
    input_size = tuple(args.input_resolution)
    # Load model from JSON file if file path was provided...
    if os.path.exists(args.model):
        try:
            with open(args.model, 'r') as f:
                json = f.read()
            model = model_from_json(json)
        except JSONDecodeError:
            raise ValueError(
                "JSON decode error found. File path %s exists but could not be decoded; verify if JSON encoding was "
                "performed properly." % args.model)
    # ...Otherwise, create model from this project by using a proper key name
    else:
        model = models_dict[args.model]((input_size[0], input_size[1], 1))

    model.load_weights(args.weights_path)

    vdo, source_fps, using_camera = open_stream(args)

    if input_size == (None, None):
        im_width = int(vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        im_width = input_size[0]
        im_height = input_size[1]

    if args.save_to is not None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if args.save_to:
            if args.save_to == "result_vdoName":
                args.save_to = "result_%s" % os.path.split(args.video_path)[-1]
        video_dir = os.path.split(args.save_to)[0]
        if not os.path.exists(video_dir) and len(video_dir) > 0:
            os.makedirs(video_dir)
        output = cv2.VideoWriter(args.save_to, fourcc, source_fps,
                                      (im_width, im_height))

    while True:
        start = time.time()
        if not using_camera:
            grabbed, ori_im = vdo.read()
            if not grabbed:
                break
        else:
            ori_im, buffered_frames = vdo.read()

        if input_size != (None, None):
            ori_im = cv2.resize(ori_im, (im_width, im_height))

        try:
            # Model name should match with the name of a model from
            # https://www.tensorflow.org/api_docs/python/tf/keras/applications/
            # This assumes you used a model with RGB inputs as the first part of your model,
            # therefore your input data should be preprocessed with the corresponding
            # 'preprocess_input' function
            m = importlib.import_module('tensorflow.keras.applications.%s' % model.name)
            rgb_preprocessor = getattr(m, "preprocess_input")
        except ModuleNotFoundError:
            rgb_preprocessor = None

        if rgb_preprocessor is not None:
            input_image = rgb_preprocessor(ori_im)
        else:
            input_image = cv2.cvtColor(ori_im, cv2.COLOR_BGR2GRAY)[..., None]
            input_image = ((input_image - input_image.min()) / (input_image.max() - input_image.min()) - 0.5) / 0.5
            input_image *= -1  # Negative so cracks are brighter
        prediction = model.predict(input_image[None, ...])[0, ...]

        # Use prediction as a mask to colorize pixels in red
        mask = 255 * np.concatenate([np.zeros(prediction.shape) for i in range(2)] + [prediction], axis=-1)
        result = np.maximum(ori_im, mask).astype(np.uint8)

        end = time.time()
        last_period = end - start
        print("Source fps: {}, frame time: {:.3f}s, processing fps: {:.1f}".format(
            round(source_fps, 2), last_period, 1 / last_period), end='\r')
        if args.show_result:
            cv2.imshow("test", result)
            # cv2.waitKey(1)

        if args.save_to is not None:
            output.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if using_camera:
        vdo.stop()
        
        
def open_stream(args):
    if os.path.isfile(args.video_path):
        try:
            vdo_stream = cv2.VideoCapture()
            vdo_stream.open(args.video_path)
        except IOError:
            raise IOError("%s is not a valid video file." % args.video_path)
        source_fps = vdo_stream.get(cv2.CAP_PROP_FPS)
        using_camera = False
    elif os.path.isdir(args.video_path):
        vdo_stream = cv2.VideoCapture()
        source_fps = 29.7
        format_str = sorted([f for f in os.listdir(args.video_path)
                             if os.path.isfile(os.path.join(args.video_path, f))
                             and not f.startswith('.') and not f.endswith('~')], key=lambda f: f.lower())[0]
        numeration = re.findall('[0-9]+', format_str)
        len_num = len(numeration[-1])
        format_str = format_str.replace(numeration[-1], "%0{}d".format(len_num))
        vdo_stream.open(os.path.join(args.video_path, format_str))
        using_camera = False
    else:
        try:
            device_id = int(args.video_path)
            vdo_stream = DeviceVideoStream(device_id).start()
            source_fps = vdo_stream.stream.get(cv2.CAP_PROP_FPS)
            using_camera = True
        except ValueError:
            raise ValueError(
                "{} is neither a valid video file, a folder with valid images nor a proper device id.".format(
                    args.video_path))

    return vdo_stream, source_fps, using_camera


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to video to evaluate.")
    parser.add_argument("model", type=str, help="Network to use.")
    parser.add_argument("weights_path", type=str, help="Path to pre-trained weights to use.")
    parser.add_argument("--input_resolution", type=int, nargs=2, default=[None, None],
                        help="If [None, None], original size frames will be fed to the model. Otherwise, frames will "
                             "be resized to this size ([width, height]).")
    parser.add_argument("--show_result", type=str, default="False",
                        help="'True' or 'False'. If True, the input image, GT (if provided) and the prediction will be "
                             "shown together in a new screen.")
    parser.add_argument("--save_to", type=str, default="result_vdoName", help="Save the comparison image to this location. "
                                                                          "If 'None', no image will be saved")
    args_dict = parser.parse_args(args)
    for attribute in args_dict.__dict__.keys():
        if args_dict.__getattribute__(attribute) == "None":
            args_dict.__setattr__(attribute, None)
        if args_dict.__getattribute__(attribute) == "True" or args_dict.__getattribute__(attribute) == "False":
            args_dict.__setattr__(attribute, bool(strtobool(args_dict.__getattribute__(attribute))))
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    main(args)
