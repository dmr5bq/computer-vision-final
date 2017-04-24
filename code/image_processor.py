import skimage, numpy, math, sys, os, copy
from skimage import io
from scipy import ndimage
import matplotlib.pyplot as plt

__author__ = "Glenna Manns, gmm6jd"

numpy.seterr(divide='ignore', invalid='ignore') # turn off div by 0/NaN warnings for numpy


class ImageProcessor:
    def __init__(self, k=None, low=None, high=None, corner=None):
        """ImageProcessor class holds essential fields for detecting edges and corners for a specific image"""

        # set kernel dimensions
        if k is not None:
            self.KERNEL_SHAPE = k
        else:
            self.KERNEL_SHAPE = 5

        # Low threshold for edge detection
        if low is not None:
            self.T_LOW = low
        else:
            self.T_LOW = .3

        # High threshold for edge detection
        if high is not None:
            self.T_HIGH = high
        else:
            self.T_HIGH = 2

        # Corner threshold
        if corner is not None:
            self.T_CORNER = corner
        else:
            self.T_CORNER = 150

        # Used throughout to determine size of arrays
        self.rows = 0
        self.cols = 0

    def retrieve_luminance(self, filename):
        """Compress image from 3D to 2D w/ ITU Recommendation for HDTV"""
        if type(filename) is str:
            img_array = io.imread(filename)  # read in pixel values (uint8)
            float_array = skimage.img_as_float(img_array)  # convert to float64
        else:
            float_array = filename

        # Set dimensions of ImageProcessor
        size = float_array.shape
        self.rows = size[0]
        self.cols = size[1]

        averaged = numpy.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                luminance = .21 * float_array[r][c][0] + .72 * float_array[r][c][1] + .07 * float_array[r][c][2]
                averaged[r][c] = luminance
        return averaged

    def smooth(self, averaged):
        """"APPLY GAUSSIAN FILTER"""
        kernel = numpy.ones((self.KERNEL_SHAPE, self.KERNEL_SHAPE))
        convolved = ndimage.filters.convolve(averaged, kernel)
        return convolved

    def retrieve_gradient(self, convolved):
        """"DERIVATIVES (via Sobel Filter)"""
        sobel_x = ndimage.sobel(convolved, axis=0, mode='reflect')
        sobel_y = ndimage.sobel(convolved, axis=1, mode='reflect')
        return sobel_x, sobel_y

    def vector_arctan(self, x, y):
        """ For edge detector: calculate the direction of the gradient; sorted into 1/5 buckets"""
        gradient_direction = numpy.zeros((self.rows, self.cols))
        options = [-numpy.pi / 2, -numpy.pi / 4, 0, numpy.pi / 4, numpy.pi / 2]
        compare = numpy.pi/8
        for r in range(self.rows):
            for c in range(self.cols):
                # normal gradient
                if x[r][c] != 0 and y[r][c] != 0:
                    gradient = numpy.arctan(y[r][c] / x[r][c])
                    result = -1
                    for o in options:
                        if math.fabs(gradient - o) < compare:
                            result = o
                        # continue
                    gradient_direction[r][c] = result
                    # print("[" + str(r) + "][" + str(c) + "] = " + str(result))
                # horizontal gradient
                elif x[r][c] != 0 and y[r][c] == 0:
                    gradient_direction[r][c] = 0
                # vertical gradient
                elif x[r][c] == 0 and y[r][c] != 0:
                    gradient_direction[r][c] = options[4]
                else:
                    gradient_direction[r][c] = 0
        return gradient_direction

    def non_max_suppression(self, magnitude, direction):
        """Compare pixels along the direction of their gradient.
        Set equal to 0 if not a local maximum"""
        pixel_array = []
        maximum = -numpy.inf

        if len(magnitude) == len(direction):
            for r in range(self.rows):
                above = max([r - 1, 0])
                below = min([r + 1, self.rows - 1])
                row = []
                for c in range(self.cols):
                    left = max([c - 1, 0])
                    right = min([c + 1, self.cols - 1])

                    # vertical
                    if magnitude[r][c] > maximum:
                        maximum = magnitude[r][c]
                    if math.fabs(direction[r][c]) == numpy.pi / 2:
                        # compare against E and W pixels
                        east = magnitude[r][right]
                        west = magnitude[r][left]
                        if magnitude[r][c] < east or magnitude[r][c] < west:
                            row.append(Pixel(0))
                        else:
                            row.append(Pixel(magnitude[r][c]))
                    # horizontal
                    if direction[r][c] == 0:
                        # compare against N and S pixels
                        north = magnitude[above][c]
                        south = magnitude[below][c]
                        if magnitude[r][c] < north or magnitude[r][c] < south:
                            row.append(Pixel(0))
                        else:
                            row.append(Pixel(magnitude[r][c]))
                    # Diagonal
                    if direction[r][c] == numpy.pi / 4:
                        # compare against north-WEST and south-EAST pixels
                        nw = magnitude[above][left]
                        se = magnitude[below][right]
                        if magnitude[r][c] < nw or magnitude[r][c] < se:
                            row.append(Pixel(0))
                        else:
                            row.append(Pixel(magnitude[r][c]))
                    # Diagonal
                    if direction[r][c] == -numpy.pi / 4:
                        # compare against north-EAST and south-WEST pixels
                        ne = magnitude[above][right]
                        sw = magnitude[below][left]
                        if magnitude[r][c] < ne or magnitude[r][c] < sw:
                            row.append(Pixel(0))
                        else:
                            row.append(Pixel(magnitude[r][c]))

                pixel_array.append(row)

            return pixel_array

    def upgrade_surrounding(self, s, r, c):
        """If the pixel located at s[r][c] is a strong edge, upgrade its weak neighbors"""
        if not s[r][c].visited:
            if s[r][c].value > self.T_LOW:
                s[r][c].visited = True
                s[r][c].value = 255

                left = max([c - 1, 0])
                right = min([c + 1, self.cols - 1])
                above = max([r - 1, 0])
                below = min([r + 1, self.rows - 1])

                if r % 17 != 0 and c % 23 != 0:
                    # Because of the size of the pictures,
                    # not having a check like this results in a stack overflow error,
                    # even if there's no infinite recursion
                    self.upgrade_surrounding(s, above, left)
                    self.upgrade_surrounding(s, above, c)
                    self.upgrade_surrounding(s, above, right)
                    self.upgrade_surrounding(s, r, left)
                    self.upgrade_surrounding(s, r, right)
                    self.upgrade_surrounding(s, below, left)
                    self.upgrade_surrounding(s, below, c)
                    self.upgrade_surrounding(s, below, right)
            else:
                s[r][c].value = 0
                s[r][c].visited = True

    def hysteresis_thresholding(self, s):
        """Used in edge detection to """
        for r in range(self.rows):
            for c in range(self.cols):
                self.check_pixel(s, r, c)

        for r in range(self.rows):
            for c in range(self.cols):
                if not s[r][c].visited:
                    s[r][c].value = 0

        # Change from array of Pixels back to luminance (array of integers)
        luminance = numpy.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                luminance[r][c] = s[r][c].value
        return luminance

    def check_pixel(self, s, r, c):
        """Check pixel against thresholds. Strong, weak, or no edge."""
        if 0 <= r < self.rows and 0 <= c < self.cols:  # bounds check
            if not s[r][c].visited:  # not previously visited
                # Strong edge
                if s[r][c].value > self.T_HIGH:
                    s[r][c].value = 255
                    self.upgrade_surrounding(s, r, c)  # upgrade surrounding pixels
                # Not an edge
                elif s[r][c].value < self.T_LOW:  # not an edge, stop
                    s[r][c].value = 0
                    s[r][c].visited = True

    def calculate_derivative_squares(self, sobel_x, sobel_y):
        """Calculate the squares of Fx, Fy, and Fxy"""
        Fx2 = numpy.zeros((self.rows, self.cols))
        Fy2 = numpy.zeros((self.rows, self.cols))
        Fxy = numpy.zeros((self.rows, self.cols))

        for r in range(self.rows):
            for c in range(self.cols):
                Fx2[r][c] = sobel_x[r][c] ** 2
                Fy2[r][c] = sobel_y[r][c] ** 2
                Fxy[r][c] = sobel_x[r][c] * sobel_y[r][c]

        return Fx2, Fy2, Fxy

    def calculate_windows(self, x, y, xy):
        """Calculate the sum of Fx, Fy, and Fxy over a 25 pixel window
        Changing m increases the size of the window """
        Fx2_avg = numpy.zeros((self.rows, self.cols))
        Fy2_avg = numpy.zeros((self.rows, self.cols))
        Fxy_avg = numpy.zeros((self.rows, self.cols))
        m = 2

        for r in range(self.rows):
            for c in range(self.cols):
                rr = max(r - m, 0)
                cc = max(c - m, 0)
                ccc = min(c + m + 1, self.cols)
                rrr = min(r + m + 1, self.rows)
                Fx = 0
                Fy = 0
                Fxy = 0
                count = 0
                while cc < ccc:
                    while rr < rrr:
                        Fx += x[rr][cc]
                        Fy += y[rr][cc]
                        Fxy += xy[rr][cc]
                        rr += 1
                        count += 1
                    rr = max(r - m, 0)
                    cc += 1
                Fx2_avg[r][c] = 1/count * Fx
                Fy2_avg[r][c] = 1/count * Fy
                Fxy_avg[r][c] = 1/count * Fxy

        return Fx2_avg, Fy2_avg, Fxy_avg

    def compute_response(self, x, y, xy):
        """Compute the magnitude of the corner 'response' -- Project suggests using eigenvalues,
        but the recommended article suggests another, faster method """
        response = numpy.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                H = [[x[r][c], xy[r][c]], [xy[r][c], y[r][c]]]
                response[r][c] = numpy.linalg.det(H) - 0.04*(numpy.trace(H) ** 2)

        return response

class Pixel:
    """Pixel class holds luminance value of a pixel and flag to det. its visited status """
    def __init__(self, val):
        self.value = val
        self.visited = False


class CornerPixel:
    def __init__(self, x, y, lum, fx, fy):
        """Corner pixel holds values needed for calculating covariance matrices, eigenvalues, etc."""
        self.x = x
        self.y = y
        self.lum = lum
        self.fx2 = fx ** 2
        self.fy2 = fy ** 2
        self.fxfy = fx * fy
        self.e = None

def edge_detector(filename):
    """Detect edges of an image using the Canny Edge Detector algorithm"""
    ip = ImageProcessor()

    if type(filename) is str:
        if os.path.isfile(filename):
            lum = ip.retrieve_luminance(filename)
        else:
            raise ValueError(str(filename) + " was not found.")
    else:
        # lum = ip.retrieve_luminance(filename)
        lum = filename
        ip.rows = lum.shape[0]
        ip.cols = lum.shape[1]

    convolved = ip.smooth(lum)
    gradient = ip.retrieve_gradient(convolved)
    magnitude = numpy.hypot(gradient[0], gradient[1]) 
    direction = ip.vector_arctan(gradient[0], gradient[1])
    suppressed = ip.non_max_suppression(magnitude, direction)

    edges = ip.hysteresis_thresholding(suppressed)

    # plt.imshow(edges, cmap='gray')
    # plt.title(filename + ' Edges')
    # plt.show()

    return edges


def corner_detector(filename, threshold=None):
    """Detect the corners of an image using the Harris Corner Detector algorithm"""
    ip = ImageProcessor(corner=threshold)

    if os.path.isfile(filename):
        if filename == 'shapes.png':  # In order to see the triangle edges, threshold needs to be really low
            ip.T_CORNER = 25
        img_array = io.imread(filename)  # read in pixel values (uint8)
        float_array = skimage.img_as_float(img_array)  # convert to float64

        # A. Retrieve luminance
        # B. Smooth with Gaussian
        # 1. Compute Fx, Fy for each pixel
        # 2. Compute Fx*Fx, Fy*Fy, Fx*Fy for each pixel
        # 3. Calculate sums for each of them (81 pixel window); compute average

        lum = ip.retrieve_luminance(filename)
        smoothed = ip.smooth(lum)
        gradient = ip.retrieve_gradient(smoothed)
        # magnitude = numpy.hypot(gradient[0], gradient[1])

        calculated = ip.calculate_derivative_squares(gradient[0], gradient[1])
        Fx2 = calculated[0]
        Fy2 = calculated[1]
        Fxy = calculated[2]

        averaged = ip.calculate_windows(Fx2, Fy2, Fxy)
        response = ip.compute_response(averaged[0], averaged[1], averaged[2])

        result = copy.copy(float_array)

        # r x c x 3 or r x c x 4
        if float_array.shape[2] == 3:
            blue_px = numpy.array([0, 0, 1])
        elif float_array.shape[2] == 4:
            blue_px = numpy.array([0, 0, 1, 1])

        # Thresholding
        for r in range(ip.rows):
            for c in range(ip.cols):
                if response[r][c] > ip.T_CORNER:
                    result[r][c] = blue_px

        # Non-max suppression
        for r in range(ip.rows):
            for c in range(ip.cols):
                if numpy.array_equal(result[r][c], blue_px):  # corner pixel values (all blue)
                    rr = max(r - 1, 0)  # top row
                    cc = max(c - 1, 0)  # left column
                    ccc = min(c + 2, ip.cols)  # two cols to right
                    rrr = min(r + 2, ip.rows)  # two rows below
                    while cc < ccc:
                        while rr < rrr:
                            if response[r][c] > response[rr][cc]:
                                result[rr][cc] = float_array[rr][cc]
                            rr += 1
                        rr = max(r - 1, 0)
                        cc += 1


        # Create cross-hairs
        for r in range(ip.rows):
            for c in range(ip.cols):
                if numpy.array_equal(result[r][c], blue_px):  # corner pixel values (all blue)
                    float_array[r][c] = blue_px
                    rr = max(r - 2, 0)  # top row
                    cc = max(c - 2, 0)  # left column
                    ccc = min(c + 2, ip.cols - 1)  # two cols to right
                    rrr = min(r + 2, ip.rows - 1)  # two rows below

                    float_array[r][cc] = blue_px
                    float_array[r][cc+1] = blue_px
                    float_array[r][ccc] = blue_px
                    float_array[r][ccc-1] = blue_px

                    float_array[rr][c] = blue_px
                    float_array[rr+1][c] = blue_px
                    float_array[rrr][c] = blue_px
                    float_array[rrr-1][c] = blue_px

        plt.imshow(float_array)
        plt.title(filename + ' Corners')
        plt.show()
    else:
        raise ValueError(str(filename) + " was not found.")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        edge_detector(sys.argv[1])
        corner_detector(sys.argv[1])
    if len(sys.argv) == 3:
        edge_detector(sys.argv[2])
        corner_detector(sys.argv[2])
    else:
        print("Please input the file name of at least one image to analyze.")
    # edge_detector('shapes.png')
    # corner_detector('shapes.png', 25)
    # edge_detector('flower.jpg')
    # corner_detector('flower.jpg', 150)


