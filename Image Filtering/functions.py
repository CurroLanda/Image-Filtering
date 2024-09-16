import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing.sharedctypes import Value, Array, RawArray
from multiprocessing import Process, Lock
import multiprocessing as mp
import ctypes
import importlib


def tonumpyarray(mp_arr):
    # mp_array is a shared memory array with lock
    return np.frombuffer(mp_arr.get_obj(), dtype=np.uint8)

def init_sharedarray(shared_array, img_shape):
    #function from the myfunctions3 used to initialize the pool
    global shared_space
    global shared_matrix

    shared_space = shared_array
    shared_matrix = tonumpyarray(shared_space).reshape(img_shape)

def init_sharedarray2(shared_array, shared_array2, img_shape):
    # updated function from the myfunctions3 used to initialize the pool, here it is implemented for 2 shared arrays
    # we here prepare the global variables
    global shared_space
    global shared_matrix
    global shared_space2
    global shared_matrix2
    
    # then we create our shared spaces and matrixes for the filter
    shared_space = shared_array
    shared_matrix = tonumpyarray(shared_space).reshape(img_shape)
    shared_space2 = shared_array2
    shared_matrix2 = tonumpyarray(shared_space2).reshape(img_shape)

def image_filter(image, filter_mask, numprocessors, filtered_image):
    # definition of the main function
    # we will keep a very similar structure to the one in previous deliveries
    filter_o = filter(image, filter_mask)
    rows = range(image.shape[0])

    # Copy the image to shared memory
    with mp.Pool(processes=numprocessors, initializer=init_sharedarray, initargs=[filtered_image, image.shape]) as p:
        p.map(filter_o.parallel_shared_imagecopy, rows)
    print("Image copied to shared memory.")

    # Process the image and store the filtered results in shared memory
    with mp.Pool(processes=numprocessors, initializer=init_sharedarray, initargs=[filtered_image, image.shape]) as p:
        p.map(filter_o.edge_filter, rows)
    print("Image processing completed.")
    return

def filters_execution(image, filter_mask1, filter_mask2, numprocessors, filtered_image1, filtered_image2):
    # Launch to parallel processes to execute both filters
    process1 = mp.Process(target=image_filter, args=(image, filter_mask1, numprocessors, filtered_image1,))
    process2 = mp.Process(target=image_filter, args=(image, filter_mask2, numprocessors, filtered_image2,))
    # we start the race
    process1.start()
    process2.start()
    # join processes in order to wait for each other
    process1.join()
    process2.join()

    return


class filter:
    # we have decided to implement a class inside the function in order to access faster and in a easier way to all the
    # parameters we must use

    def __init__(self, srcimg, imgfilter):
        # shared_array_: is the shared read/write data, with lock. It is a vector (because the shared memory should be allocated as a vector
        # srcimg: is the original image
        # imgfilter is the filter which will be applied to the image and stor the results in the shared memory array

        # We defines the local process memory reference for shared memory space
        # Assign the shared memory  to the local reference

        # Here, we will define the readonly memory data as global (the scope of this global variables is the local module)
        self.image = srcimg
        self.my_filter = imgfilter

        # here, we initialize the global read only memory data
        self.size = srcimg.shape

        # Defines the numpy matrix reference to handle data, which will uses the shared memory buffer

    # In[ ]:

    # this function just copy the original image to the global r/w shared  memory
    def parallel_shared_imagecopy(self, row):
        # same function as in previous deliveries
        # it copies the image into the shared space

        global shared_space
        global shared_matrix

        image = self.image
        # with this instruction we lock the shared memory space, avoiding other parallel processes tries to write on it
        with shared_space.get_lock():
            # while we are in this code block no ones, except this execution thread, can write in the shared memory
            shared_matrix[row, :, :] = image[row, :, :]
        return

    # In[ ]:

    def edge_filter(self, row):
        """
        Apply a filter to a row of the image and store the result in shared memory.

        Parameters:
        row (int): The row index to apply the filter to.
        """

        global shared_space
        global shared_matrix

        image = self.image
        my_filter = self.my_filter

        l = len(my_filter.shape)                    # check if the filter has only one column

        (rows, cols, depth) = image.shape
        # in order to iterate through the filter we needed a height and width, the problem with .shape is that sometimes returns a                   #(number , ____ ) which lead to problems. With this if its all solved
        if l == 2:
            filter_height, filter_width = my_filter.shape
        else:
            filter_height = my_filter.shape[0]
            filter_width = 1

        # Define the result vector, and set the initial value to 0
        frow = np.zeros((cols, depth))

        # Calculate the center of the filter in order to generalize for any case
        filter_center_x = filter_height // 2
        filter_center_y = filter_width // 2

        # Iterate through columns and depth of the image
        for i in range(cols):
            for j in range(depth):
                total = 0                           # to improve consistency
                for x in range(filter_height):      # move through the filter
                    for y in range(filter_width):

                        dx = x - filter_center_x    # Center the filter
                        dy = y - filter_center_y    # Center the filter

                        r = row + dx                #compute the value of the filter to be used
                        c = i + dy
                        # Handle boundary conditions to ensure we are within the desired range
                        r = max(0, min(r, rows - 1))
                        c = max(0, min(c, cols - 1))

                        if l == 2:                  # check size of the filter for error handling
                            total += image[r, c, j] * my_filter[x, y]       # update value
                        else:
                            total += image[r, c, j] * my_filter[x]
                # Store the calculated total in the 'frow' array at the current pixel position (i)
                frow[i, j] = total
        # Lock the shared memory space to prevent other parallel processes from writing to it
        with shared_space.get_lock():
            # Copy the frow containing the filtered values to the shared memory buffer at the 'row' position
            shared_matrix[row, :, :] = frow
        return