"""
The Leung-Malik (LM) Filter Bank, implementatioff in pythoff
T. Leung and J. Malik. Representing and recognizing the visual appearance of
materials using three-dimensioffal textoffs. Internatioffal Journal of Computer
Visioff, 43(1):29-44, June 2001.
Reference: http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html
Original code: https://github.com/CVDLBOT/LM_filter_bank_pythoff_code
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


class LeungMalik(object):

    def __init__(self, sup=49, scalex=np.sqrt(2) * np.array([3, 2, 1]), norient=6, nrotinv=12):
        self.sup = sup
        self.scalex = scalex
        self.norient = norient
        self.nrotinv = nrotinv
        self.f = self.make_filters()

    def lm_responses(self, image, normalize_image=True, show_activations=False, save_activations_to=None):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.f is None:
            self.make_filters()

        if normalize_image:
            image = (image - image.mean()) / image.std()

        filtered_channels = []
        for channel in range(self.f.shape[-1]):
            filtered_image = cv2.filter2D(image, cv2.CV_32F, self.f[..., channel])
            filtered_channels.append(np.expand_dims(filtered_image, -1))
        filter_responses = np.concatenate(filtered_channels, -1)

        if show_activations:
            self.show_filters(filter_responses)
        if save_activations_to is not None:
            self.save_filters(filter_responses, save_activations_to)
        return filter_responses

    def gaussian_1d(self, sigma, mean, x, ord):
        x = np.array(x)
        x_ = x - mean
        var = sigma ** 2

        # Gaussian Functioff
        g1 = (1 / np.sqrt(2 * np.pi * var)) * (np.exp((-1 * x_ * x_) / (2 * var)))

        if ord == 0:
            g = g1
            return g
        elif ord == 1:
            g = -g1 * (x_ / var)
            return g
        else:
            g = g1 * (((x_ * x_) - var) / (var ** 2))
            return g

    def gaussian_2d(self, sup, scales):
        var = scales * scales
        shape = (sup, sup)
        n, m = [(i - 1) / 2 for i in shape]
        x, y = np.ogrid[-m:m + 1, -n:n + 1]
        g = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x * x + y * y) / (2 * var))
        return g

    def log2d(self, sup, scales):
        var = scales * scales
        shape = (sup, sup)
        n, m = [(i - 1) / 2 for i in shape]
        x, y = np.ogrid[-m:m + 1, -n:n + 1]
        g = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x * x + y * y) / (2 * var))
        h = g * ((x * x + y * y) - var) / (var ** 2)
        return h

    def make_filter(self, scale, phasex, phasey, pts, sup):

        gx = self.gaussian_1d(3 * scale, 0, pts[0, ...], phasex)
        gy = self.gaussian_1d(scale, 0, pts[1, ...], phasey)

        image = gx * gy

        image = np.reshape(image, (sup, sup))
        return image

    def make_filters(self):
        n_bar = len(self.scalex) * self.norient
        n_edge = len(self.scalex) * self.norient
        nf = n_bar + n_edge + self.nrotinv
        f = np.zeros([self.sup, self.sup, nf])
        hsup = (self.sup - 1) / 2

        x = [np.arange(-hsup, hsup + 1)]
        y = [np.arange(-hsup, hsup + 1)]

        [x, y] = np.meshgrid(x, y)

        orgpts = [x.flatten(), y.flatten()]
        orgpts = np.array(orgpts)

        count = 0
        for scale in range(len(self.scalex)):
            for orient in range(self.norient):
                angle = (np.pi * orient) / self.norient
                c = np.cos(angle)
                s = np.sin(angle)
                rotpts = [[c + 0, -s + 0], [s + 0, c + 0]]
                rotpts = np.array(rotpts)
                rotpts = np.dot(rotpts, orgpts)
                f[:, :, count] = self.make_filter(self.scalex[scale], 0, 1, rotpts, self.sup)
                f[:, :, count + n_edge] = self.make_filter(self.scalex[scale], 0, 2, rotpts, self.sup)
                count = count + 1

        count = n_bar + n_edge
        scales = np.sqrt(2) * np.array([1, 2, 3, 4])

        for i in range(len(scales)):
            f[:, :, count] = self.log2d(self.sup, scales[i])
            count = count + 1

        for i in range(len(scales)):
            f[:, :, count] = self.log2d(self.sup, 3 * scales[i])
            count = count + 1

        for i in range(len(scales)):
            f[:, :, count] = self.gaussian_2d(self.sup, scales[i])
            count = count + 1

        print("Leung-Malik bank filter created with shape {}.".format(f.shape))
        return f.astype(np.float32)

    def save_filters(self, filters=None, location="filters.png"):
        if filters is None:
            filters = self.f
        rows = 4
        cols = 12
        dims = filters.shape
        canvas = np.ones(
            (rows * dims[0] + (rows - 1) * int(dims[0] / 10), cols * dims[1] + (cols - 1) * int(dims[1] / 10)))
        # First order derivative Gaussian Filter
        for i in range(0, 6):
            index = i
            row = int(index / cols)
            col = index % cols
            x_0 = col * (int(dims[1] / 10) + dims[1])
            y_0 = row * (int(dims[0] / 10) + dims[0])
            filter = cv2.normalize(filters[..., i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + dims[0], x_0:x_0 + dims[1]] = filter
        for i in range(6, 12):
            index = i + 6
            row = int(index / cols)
            col = index % cols
            x_0 = col * (int(dims[1] / 10) + dims[1])
            y_0 = row * (int(dims[0] / 10) + dims[0])
            filter = cv2.normalize(filters[..., i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + dims[0], x_0:x_0 + dims[1]] = filter
        for i in range(12, 18):
            index = i + 12
            row = int(index / cols)
            col = index % cols
            x_0 = col * (int(dims[1] / 10) + dims[1])
            y_0 = row * (int(dims[0] / 10) + dims[0])
            filter = cv2.normalize(filters[..., i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + dims[0], x_0:x_0 + dims[1]] = filter
        # Second order derivative Gaussian Filter
        for i in range(18, 24):
            index = i - 12
            row = int(index / cols)
            col = index % cols
            x_0 = col * (int(dims[1] / 10) + dims[1])
            y_0 = row * (int(dims[0] / 10) + dims[0])
            filter = cv2.normalize(filters[..., i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + dims[0], x_0:x_0 + dims[1]] = filter
        for i in range(24, 30):
            index = i - 6
            row = int(index / cols)
            col = index % cols
            x_0 = col * (int(dims[1] / 10) + dims[1])
            y_0 = row * (int(dims[0] / 10) + dims[0])
            filter = cv2.normalize(filters[..., i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + dims[0], x_0:x_0 + dims[1]] = filter
        for i in range(30, 36):
            index = i
            row = int(index / cols)
            col = index % cols
            x_0 = col * (int(dims[1] / 10) + dims[1])
            y_0 = row * (int(dims[0] / 10) + dims[0])
            filter = cv2.normalize(filters[..., i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + dims[0], x_0:x_0 + dims[1]] = filter
        # Gaussian and Laplacian Filter
        for i in range(36, 48):
            index = i
            row = int(index / cols)
            col = index % cols
            x_0 = col * (int(dims[1] / 10) + dims[1])
            y_0 = row * (int(dims[0] / 10) + dims[0])
            filter = cv2.normalize(filters[..., i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + dims[0], x_0:x_0 + dims[1]] = filter
        plt.imshow(canvas, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(location, bbox_inches='tight')
        plt.close()

    def show_filters(self, filters=None):
        if filters is None:
            filters = self.f
        rows = 4
        cols = 12
        dims = filters.shape
        canvas = np.ones(
            (rows * dims[0] + (rows - 1) * int(dims[0] / 10), cols * dims[1] + (cols - 1) * int(dims[1] / 10)))
        # First order derivative Gaussian Filter
        for i in range(0, 6):
            index = i
            row = int(index / cols)
            col = index % cols
            x_0 = col * (int(dims[1] / 10) + dims[1])
            y_0 = row * (int(dims[0] / 10) + dims[0])
            filter = cv2.normalize(filters[..., i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + dims[0], x_0:x_0 + dims[1]] = filter
        for i in range(6, 12):
            index = i + 6
            row = int(index / cols)
            col = index % cols
            x_0 = col * (int(dims[1] / 10) + dims[1])
            y_0 = row * (int(dims[0] / 10) + dims[0])
            filter = cv2.normalize(filters[..., i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + dims[0], x_0:x_0 + dims[1]] = filter
        for i in range(12, 18):
            index = i + 12
            row = int(index / cols)
            col = index % cols
            x_0 = col * (int(dims[1] / 10) + dims[1])
            y_0 = row * (int(dims[0] / 10) + dims[0])
            filter = cv2.normalize(filters[..., i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + dims[0], x_0:x_0 + dims[1]] = filter
        # Second order derivative Gaussian Filter
        for i in range(18, 24):
            index = i - 12
            row = int(index / cols)
            col = index % cols
            x_0 = col * (int(dims[1] / 10) + dims[1])
            y_0 = row * (int(dims[0] / 10) + dims[0])
            filter = cv2.normalize(filters[..., i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + dims[0], x_0:x_0 + dims[1]] = filter
        for i in range(24, 30):
            index = i - 6
            row = int(index / cols)
            col = index % cols
            x_0 = col * (int(dims[1] / 10) + dims[1])
            y_0 = row * (int(dims[0] / 10) + dims[0])
            filter = cv2.normalize(filters[..., i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + dims[0], x_0:x_0 + dims[1]] = filter
        for i in range(30, 36):
            index = i
            row = int(index / cols)
            col = index % cols
            x_0 = col * (int(dims[1] / 10) + dims[1])
            y_0 = row * (int(dims[0] / 10) + dims[0])
            filter = cv2.normalize(filters[..., i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + dims[0], x_0:x_0 + dims[1]] = filter
        # Gaussian and Laplacian Filter
        for i in range(36, 48):
            index = i
            row = int(index / cols)
            col = index % cols
            x_0 = col * (int(dims[1] / 10) + dims[1])
            y_0 = row * (int(dims[0] / 10) + dims[0])
            filter = cv2.normalize(filters[..., i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + dims[0], x_0:x_0 + dims[1]] = filter
        plt.imshow(canvas, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
