# 3D local maxima on foreground


def local_max(img, FWHMxy, pixel_size=0.16125, gauss_sigma=0.25, verbose=False, debug=False,
              save_raster=False, gauss_filter=True):
    """
    Detect local maximas on a numpy array
    """
    import scipy.ndimage
    import numpy as np
    import pandas as pd
    import sys
    import tifffile as tiff

    kernel_radius = (FWHMxy / 2) / pixel_size

    if debug:
        print('Received pixel radius of ' + str(kernel_radius) + ','),
        sys.stdout.flush()

    kernel_radius = int(round(kernel_radius))

    # Generate kernel
    kernel = np.ones([3, 3, 3], dtype='uint8')

    if debug:
        print('using ' + str(kernel_radius) + ' pixels.')
        sys.stdout.flush()

    # Convert to float
    img = img.astype(float)
    if debug:
        print ('Working on ' + str(img.dtype))
        sys.stdout.flush()

    # Smoothing with gaussian filter
    if gauss_filter:
        img_filtered = scipy.ndimage.filters.gaussian_filter(img, sigma=gauss_sigma, mode='constant', cval=0)
    else:
        img_filtered = img

    # Zero mask only makes sense on the accumulator array !
    zero_mask = img_filtered > 0

    if debug:
        print('Kernel dimensions: ' + str(np.shape(kernel)))
        print('Initialising maximum filter...')
        sys.stdout.flush()

    # Maximum filter
    img_max = scipy.ndimage.filters.maximum_filter(img_filtered, footprint=kernel)

    # Compare max and original image, to get local maximas
    local_maximas = [i for i in img_filtered == img_max]
    local_maximas = np.array(local_maximas)
    local_maximas *= zero_mask  # because zero on a neighbourhood of zeros is a local max

    if save_raster:
        if debug:
            print ('Saving detections as raster to disk:')
            print (str(save_raster)),
        sys.stdout.flush()
        img_to_save = local_maximas.astype('uint8', copy=True)
        img_to_save *= 255
        tiff.imsave(save_raster, img_to_save)
        if debug:
            print ('[Done]')

    # Count number of detections
    num_detections = np.sum(local_maximas)
    num_features = num_detections
    if verbose:
        print (str(num_detections) + ' detections')
        sys.stdout.flush()

    positions = np.nonzero(local_maximas)
    positions = np.transpose(positions)
    detections_df = pd.DataFrame(positions, columns=['z', 'y', 'x'])

    return detections_df, num_features, local_maximas, img_filtered


def local_max_old(img, FWHMxy, pixel_size=0.16125, gauss_sigma=0.25, verbose=False, debug=False,
                  save_raster=False, gauss_filter=True, non_zero_mask=False, show_time=True, mask=True):
    """
    Detect local maximas on a numpy array
    """
    import scipy.ndimage
    import numpy as np
    import pandas as pd
    import skimage.morphology
    import skimage.exposure
    # from skimage.filters import threshold_otsu as thresh_func
    from skimage.filters import threshold_yen as thresh_func
    import math
    import sys
    import memotrack.display
    from matplotlib import pyplot as plt
    import tifffile as tiff
    import time

    # Time counters
    start_time = time.time()

    kernel_radius = (FWHMxy / 2) / pixel_size
    z_ratio = 1

    if debug:
        print('Received pixel radius of ' + str(kernel_radius) + ','),
        sys.stdout.flush()

    # Using ceil
    # kernel_radius = int(math.ceil(kernel_radius))

    kernel_radius = int(round(kernel_radius))

    kernel_diameter = 2 * kernel_radius

    # Generate kernel
    kernel = np.ones([3, 3, 3], dtype='uint8')

    if debug:
        print('using ' + str(kernel_radius) + ' pixels.')
        sys.stdout.flush()

    # Convert to float
    img = img.astype(float)
    if debug:
        print ('Working on ' + str(img.dtype))
        sys.stdout.flush()

    temp_timer_start = time.time()
    # Smoothing wit gaussian filter
    if gauss_filter:
        img_filtered = scipy.ndimage.filters.gaussian_filter(img, sigma=gauss_sigma, mode='constant', cval=0)
    else:
        img_filtered = img

    gaussian_filter_time = (time.time() - temp_timer_start)

    if non_zero_mask:
        zero_mask = img_filtered > 0
    else:
        zero_mask = img_filtered >= 0

    if debug:
        print('Kernel dimensions: ' + str(np.shape(kernel)))
        print('Initialising maximum filter...')
        sys.stdout.flush()
    temp_timer_start = time.time()
    # Maximum filter
    img_max = scipy.ndimage.filters.maximum_filter(img_filtered, footprint=kernel)
    max_filter = time.time() - temp_timer_start

    # Compare max and original image, to get local maximas
    local_maximas = [i for i in img_filtered == img_max]
    local_maximas = np.array(local_maximas)

    # Mask the detected maximas, to ignore maximas on the background
    local_maximas = local_maximas * zero_mask
    local_maximas = local_maximas * mask

    if save_raster:
        if debug:
            print ('Saving detections as raster to disk:')
            print (str(save_raster)),
        sys.stdout.flush()
        img_to_save = local_maximas.astype('uint8', copy=True)
        img_to_save *= 255
        tiff.imsave(save_raster, img_to_save)
        if debug:
            print ('[Done]')

    # Count number of detections
    num_detections = np.sum(local_maximas)
    num_features = num_detections
    if verbose:
        print (str(num_detections) + ' detections')
        sys.stdout.flush()

    positions = np.nonzero(local_maximas)
    positions = np.transpose(positions)
    detections_df = pd.DataFrame(positions, columns=['z', 'y', 'x'])

    end_time = time.time() - start_time

    if show_time:
        print ('Process time:')
        print(' Gaussian filter: ' + str(gaussian_filter_time) + ' seconds')
        print(' Maximum filter: ' + str(max_filter) + ' seconds')
        print(' Total time: ' + str(end_time) + ' seconds')

    return detections_df, num_features, local_maximas, img_filtered


# Check distances from the detection
def check_distances(local_maximas, FWHMxy, pixel_size=0.16125, verbose=False, plot_matrix=False, top_limit=20000,
                    save_distribution=True, debug=False):
    """
    Check the distances between detected maximas, looking for double detections
    :param local_maximas: detected points
    :param FWHMxy: distance to check if points are touching
    :param pixel_size: for convertions pixel to micron
    :param verbose: Print process
    :param plot_matrix: True if you want to plot the distances matrix
    :return: number of points closer than FWHMxy
    """
    import numpy as np
    import scipy.spatial.distance
    from matplotlib import pyplot as plt
    import sys
    from scipy.spatial import Delaunay
    from scipy.spatial.distance import euclidean as dist
    import memotrack.analyse
    import math
    from collections import defaultdict
    import math

    # If the number of detections is too big, it uses too much RAM
    # (for sure 74851 detections crash on local)
    ndetections = len(local_maximas.index)
    if ndetections > top_limit:
        print ('\nWARNING: ' + str(ndetections) + ' detections found, something might be wrong.')
        print ('(maximum number of detections set to ' + str(top_limit) + ')')
        print ('Setting all points as double detections, increasing sigma.')
        double_detections = ndetections
        # double_detections = float('inf')
        neighbour_average = 0
        return double_detections, neighbour_average

    # sigma_factor = 2 * math.sqrt(2 * math.log(2))
    # thresh = 2 * ((FWHMxy / pixel_size) / sigma_factor)  # here using two stdev as nucleus size

    thresh = FWHMxy / pixel_size  # To use normal FWHM
    # thresh = (FWHMxy / pixel_size) / 2  # Old version
    if verbose:
        print ('Checking distance of ' + str(thresh) + ' voxels')
        sys.stdout.flush()

    positions = local_maximas.as_matrix(columns=['z', 'y', 'x'])

    tri = Delaunay(positions)

    # Get neighbors list. For each point, the connected ones via Delaunay triangulation
    if verbose:
        print ('Creating neighbors list'),
        sys.stdout.flush()
    neighbors = defaultdict(set)
    for simplex in tri.simplices:
        for idx in simplex:
            other = set(simplex)
            other.remove(idx)
            neighbors[idx] = neighbors[idx].union(other)
    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    if verbose:
        print ('Generating distances matrix'),
        sys.stdout.flush()
    dist_matrix = scipy.spatial.distance.cdist(positions, positions)

    if verbose:
        print (np.shape(dist_matrix)),
        print ('[Done]')
        sys.stdout.flush()

    # Calculate distances
    if verbose:
        print ('Getting distances'),
        sys.stdout.flush()
    dist_list = []
    point_counter = 0
    for point in range(len(positions)):
        if verbose:
            if point_counter >= 100:
                print ('.'),
                sys.stdout.flush()
                point_counter = 0
        temp_dist_list = []
        for neighbor in neighbors[point]:
            if point != neighbor:
                temp_dist_list.append(dist_matrix[point, neighbor])
            else:
                print('Point equals neighbor!')

        # Get median value
        dist_list.append(np.median(temp_dist_list))

        point_counter += 1

    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    if save_distribution:
        bins = 50
        memotrack.display.histogram(dist_list, nbins=bins, ylog=True, rotate_labels=True,
                                    custom_bins=[0, (float(thresh) / 4) * bins], vline=thresh,
                                    ymax=3000, normed=False, title=str(len(dist_list)) + ' detections ',
                                    save='/projects/memotrack/temp/hists/hist' + str(save_distribution) + '.png')

    double_detections = sum([i < thresh for i in dist_list])
    neighbour_average = np.mean(dist_list) * pixel_size
    neighbour_min = np.min(dist_list) * pixel_size
    diagonal = tri.npoints

    if verbose:
        print ('Number of detections: ' + str(diagonal) + ' (safe check)')
        print ('Number of double detections: ' + str(double_detections))
        print ('Neighbour average distance: ' + str(neighbour_average / pixel_size) + ' pixels, ' + str(
            neighbour_average) + ' microns')
        print ('Neighbour minimum distance: ' + str(neighbour_min / pixel_size) + ' pixels, ' + str(
            neighbour_min) + ' microns')

    return double_detections, neighbour_average


# Create pandas dataframe from Local maximas
def create_dataframe(local_maximas, verbose=False):
    """
    Generates a pandas dataframe from the local maxima detection
    :param local_maximas: Detected points, in a TZYX numpy array
    :param verbose: print process
    :return: pandas dataframe with the coordinates
    """
    import numpy as np
    import pandas as pd

    # Start dataframe
    detections_df = pd.DataFrame(columns=['x', 'y', 'z', 't'])

    if verbose:
        print('Reading frames:'),

    frame_num = 0
    for frame in local_maximas:
        if verbose:
            print('[' + str(frame_num) + ']'),

        # Get detection indices
        ind = np.transpose(np.nonzero(frame))
        ind_df = pd.DataFrame(ind, columns=['z', 'y', 'x'])
        ind_df['t'] = frame_num
        detections_df = pd.concat([detections_df, ind_df])
        frame_num += 1

    ndetections = len(detections_df.index)

    detections_df.index = range(ndetections)

    if verbose:
        print('[Done]'),

    return detections_df


def density_from_img(local_maximas):
    """
    Returns a dataframe of density values from the 4D image.
    This function is from before DBSCAN, should be deleted in next versions
    :param local_maximas: Numpy in TZYX
    :return:
    """
    import numpy as np
    import pandas as pd

    detection_density = np.sum(local_maximas, axis=0)

    density_indices = np.transpose(np.nonzero(detection_density))
    density_df = pd.DataFrame(density_indices, columns=['z', 'y', 'x'])

    density_list = np.zeros(len(density_indices))

    # create density values
    for detection in range(len(density_indices)):
        coords = density_indices[detection]
        value = detection_density[(coords[0], coords[1], coords[2])]
        density_list[detection] = value

    density_df['density'] = density_list
    return density_df


def density_from_df(detections_df):
    """
    Generates the density dataframe from the detections dataframe
    This function is from before DBSCAN, should be deleted in next versions
    :param detections_df: Pandas dataframe with txyz coordinates
    :return: density dataframe, with xyz and density (count of detections on the coordinate)
    """
    import pandas as pd

    density_df = detections_df.copy(deep=True)
    del density_df['t']
    density_df['density'] = 1

    density_df = density_df.groupby(['x', 'y', 'z'])['density'].sum().reset_index()
    # density_df.drop_duplicates(inplace=True)

    return density_df


def df_closest(detections_df, loc, verbose=False):
    """
    Gets the closest point on the 4D space, using the dataframe.
    :param detections_df: dataframe of detections
    :param loc: position to check
    :param verbose: prints results
    :return: index of closest point and distance value
    """
    from scipy.spatial import distance
    import numpy as np

    if verbose:
        print ('Analysing:')
    a = detections_df.loc[loc]

    if verbose:
        print (a)

    ndetections = len(detections_df.index)
    distances = np.zeros(ndetections)

    for n in range(ndetections):
        dist = distance.euclidean(a, detections_df.loc[n])
        distances[n] = dist

    # First get itself and delete it from list
    closest_index = np.argmin(distances)
    distances = np.delete(distances, closest_index)

    # Getting the second closest
    closest_index = np.argmin(distances)

    if verbose:
        print ('\nClosest point:')
        print(detections_df.loc[closest_index])
        print ('\nDistance: ' + str(distances[closest_index]))

    return closest_index, distances


def dbscan(detections_df, eps=1.337, min_samples=10, verbose=True, FWHM=True, pixel_size=0.16125, orig_coords=False):
    """
    Runs DBSCAN, to find the nuclei clusters
    :param detections_df: pandas dataframe with the detections
    :param eps: reach distance for points
    :param min_samples: minimum number of points for core
    :param verbose: prints process
    :return: labeled pandas dataframe
    """
    from sklearn.cluster import DBSCAN

    # Converting microns to pixels. Here we want the distance to be half of the diameter
    if FWHM:
        eps = (float(eps) / 4) / pixel_size

    # Getting numpy array from dataframe
    if orig_coords:
        X = detections_df.as_matrix(columns=['x', 'y', 'z'])
    else:
        X = detections_df.as_matrix(columns=['xreg', 'yreg', 'zreg'])

    # clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, leaf_size=3).fit(X)

    # Putting labels back on dataframe
    detections_df['label'] = db.labels_

    # Getting number of labels
    nlabels = detections_df['label'].max()

    if verbose:
        print ('Number of clusters: ' + str(nlabels))

    return detections_df, nlabels


def minimize_dbscan_error(detections_df, eps=1.337, min_samples=10, verbose=True, stop_iter=10, minimize='eps',
                          max_iter=50):
    import numpy as np
    import memotrack

    # Initialize variables
    nframes = detections_df['t'].max()
    stop = 0
    double_stop = 0
    min_error = float('inf')
    min_error_local = float('inf')
    error_list = []

    if minimize == 'min_samples':
        best_eps_value = eps
        for i in range(max_iter):
            if stop > stop_iter:
                break
            if i == 0:
                min_samples = 0

            min_samples += 1

            detections_df, nlabels = memotrack.process.dbscan(detections_df, eps=eps, min_samples=min_samples,
                                                              FWHM=True,
                                                              verbose=False)

            error_list.append(memotrack.analyse.detections_error(detections_df, verbose=False))

            # Check if current error is better
            if sum(error_list[i]) < min_error:
                min_error = sum(error_list[i])
                best_min_value = min_samples
                lowest_error_pos = i
                stop = 0
            else:
                # If current error is worst, add to stop
                stop += 1

            if verbose:
                print ('[' + str(i) + ']\tmin_samples: ' + str(min_samples) + '\tError sum: ' + str(sum(error_list[i])))

    if minimize == 'eps':
        best_min_value = min_samples
        for i in range(max_iter):
            if stop > stop_iter:
                break
            if i == 0:
                eps = 0.80

            eps += 0.05

            detections_df, nlabels = memotrack.process.dbscan(detections_df, eps=eps, min_samples=min_samples,
                                                              FWHM=True,
                                                              verbose=False)

            error_list.append(memotrack.analyse.detections_error(detections_df, verbose=False))

            # Check if current error is better
            if sum(error_list[i]) < min_error:
                min_error = sum(error_list[i])
                best_eps_value = eps
                lowest_error_pos = i
                stop = 0
            else:
                # If current error is worst, add to stop
                stop += 1

            if verbose:
                print ('[' + str(i) + ']\teps: ' + str(eps) + '\tError sum: ' + str(sum(error_list[i])))

    if minimize == 'both':
        initial_eps = 1
        for i in range(max_iter):
            if i == 0:
                min_samples = 1
                eps = initial_eps

            if double_stop > stop_iter:
                break

            eps += 0.05

            if stop > stop_iter:
                min_samples += 1
                double_stop += 1
                eps = initial_eps
                stop = 0
                min_error_local = float('inf')

            detections_df, nlabels = memotrack.process.dbscan(detections_df, eps=eps, min_samples=min_samples,
                                                              FWHM=True,
                                                              verbose=False)

            error_list.append(memotrack.analyse.detections_error(detections_df, verbose=False))

            # Check if current error is better
            if sum(error_list[i]) < min_error_local:
                min_error_local = sum(error_list[i])
                stop = 0
                # Check if its better than global error
                if min_error_local < min_error:
                    min_error = min_error_local
                    best_eps_value = eps
                    best_min_value = min_samples
                    lowest_error_pos = i
                    double_stop = 0

            else:
                # If current error is worst, add to stop
                stop += 1

            if verbose:
                print (
                    '[' + str(i) + ']\teps: ' + str(eps) + '\tmin_samples: ' + str(min_samples) + '\tError sum: ' + str(
                        sum(error_list[i])))

    if verbose:
        print ('\nAll done !')
        print ('best min_value: ' + str(best_min_value))
        print ('best eps_value: ' + str(best_eps_value))
        print ('position of lowest error: ' + str(lowest_error_pos))
        print ('error sum: ' + str(min_error))
        # Error sequence:  double_ratio, missing_ratio, noise_ratio
        print ('   double ratio: ' + str(error_list[best_min_value][0]))
        print ('   missing ratio: ' + str(error_list[best_min_value][1]))
        print ('   noise ratio: ' + str(error_list[best_min_value][2]))

    return error_list, lowest_error_pos, best_eps_value, best_min_value


def fill_gaps_old_pandas(detections_df, verbose=True):
    import pandas as pd
    import sys

    # Getting number of labels
    nlabels = int(detections_df['label'].max())
    nframes = int(detections_df['t'].max())

    # create new df
    new_df = pd.DataFrame(columns=['x', 'y', 'z', 't', 'label'])

    # Run all labels filling time gaps
    if verbose:
        print ('Filling data gaps'),
        sys.stdout.flush()
    counter = 0
    for label in range(nlabels):
        # Fancy dots for progress
        if verbose and counter > 100:
            print ('.'),
            counter = 0

        df = detections_df[detections_df['label'] == label].copy(deep=True)
        df.drop_duplicates(subset='t', inplace=True)
        # Use time as index
        df.set_index('t', inplace=True)

        # Expanding the index, thus creating missing values

        # The inplace argument don't work for reindex on newer versions of Pandas.
        # It was working fine on 0.15.1 but fails on 0.17.0
        # df = df.reindex(index=range(nframes), inplace=True)
        # Replacing for the following instead:
        df2 = df.reindex(index=range(nframes))

        # interpolate values
        df2.interpolate(method='linear', inplace=True)

        # Drop the NaNs
        # Same thing here, dropna don't have inplace argument on new version of pandas
        # df.dropna(inplace=True)
        df2.dropna()

        # expand again, to use bfill
        # Same thing here, reindex don't have inplace argument on new version of pandas
        df3 = df2.reindex(index=range(nframes), method='bfill')

        # Drop the NaNs again
        # Same thing here, dropna don't have inplace argument on new version of pandas
        df3.dropna()

        # get time column back
        df3.reset_index(level=0, inplace=True)

        # concat with new_df
        new_df = pd.concat([new_df, df3])

        # Counter for the dots
        counter += 1

    new_df.reset_index(level=0, inplace=True)
    del new_df['index']

    if verbose:
        print('[Done]')
        sys.stdout.flush()

    return new_df


def fill_gaps(detections_df, verbose=True, debug=False, gap_thresh=0.5):
    import pandas as pd
    import sys

    # Getting number of labels
    nlabels = int(detections_df['label'].nunique())
    print ('Before interpolation, total of ' + str(nlabels) + ' labels')

    if debug:
        nlabels = 2

    nframes = int(detections_df['t'].nunique())

    # create new df
    new_df = pd.DataFrame(columns=['x', 'y', 'z', 't', 'label'])

    # Run all labels filling time gaps
    if verbose:
        print ('Filling data gaps'),
        sys.stdout.flush()
    counter = 0
    for label in range(nlabels):
        # Fancy dots for progress
        if verbose and counter > 100:
            print ('.'),
            counter = 0

            # if debug:
            # label = 2400

        # Copy the frame to work on
        df = detections_df[detections_df['label'] == label].copy(deep=True)
        if debug:
            print ('\n----------------------')
            print ('\nLabel ' + str(label))
        # This is for safety, just in case there are still duplicates
        df.drop_duplicates(subset='t', inplace=True)

        # Use time as index
        df.set_index('t', inplace=True)
        if debug:
            print ('\nAfter putting time as index:')
            print(df)

        # Expanding the index, thus creating missing values
        df = df.reindex(index=range(nframes))
        if debug:
            print ('\nAfter reindex:')
            print(df)

        # count the number of missing data points for this label
        ngaps = df['x'].isnull().sum()

        # Here we check if we have enough data to continue
        if ngaps > (gap_thresh * nframes):
            # Not enough data for interpolation !
            print ('Label ' + str(label) + ' gaps: ' + str(ngaps))
        else:
            # Continue with interpolation

            # interpolate values
            df.interpolate(method='linear', inplace=True)
            if debug:
                print ('\nAfter interpolation:')
                print(df)

            # Drop the NaNs
            df = df.dropna()
            if debug:
                print ('\nAfter dropna:')
                print(df)

            # expand again, to use bfill
            df = df.reindex(index=range(nframes), method='bfill')
            if debug:
                print ('\nAfter reindex with bfill:')
                print(df)

            # Drop the NaNs again
            df = df.dropna()
            if debug:
                print ('\nAfter second dropna:')
                print(df)

            # get time column back
            df.reset_index(level=0, inplace=True)

            # concat with new_df
            new_df = pd.concat([new_df, df])

        # Counter for the dots
        counter += 1

    # Before we had a problem that needed index correction.
    del (new_df['level_0'])
    new_df.reset_index(level=0, inplace=True)
    del new_df['index']

    if verbose:
        print('[Done]')
        sys.stdout.flush()

    # Getting number of labels
    new_nlabels = int(new_df['label'].nunique())
    print ('After interpolation, total of ' + str(new_nlabels) + ' labels')

    # Dataframe has "empty labels" now, that will cause problems later.
    # We need to drop them, and re-index the labels accordingly

    print ('#' * 25)
    label_list = new_df['label'].unique()
    print ('\nTotal of ' + str(len(label_list)) + ' labels')
    print (label_list)
    new_label_list = range(len(label_list))

    # Replacing
    new_df['label'].replace(to_replace=label_list, value=new_label_list, inplace=True)

    final_label_list = new_df['label'].unique()
    print ('\nReplace done.')
    print ('Total of ' + str(len(final_label_list)) + ' labels')
    print (final_label_list)

    return new_df


def get_signal(detections_df, img_path, verbose=True, FWHMxy=1.21, debug=False, spline_order=3,
               save_voronoi=True):
    import memotrack.io
    import sys
    import numpy as np
    import scipy.ndimage
    from scipy.spatial import KDTree
    import tifffile

    # Reading signal metadata
    meta = memotrack.io.meta(img_path, verbose=False)

    # Setting variables for img size
    SizeT = meta['SizeT']
    SizeZ = meta['SizeZ']
    SizeY = meta['SizeY']
    SizeX = meta['SizeX']
    SizeC = meta['SizeC']
    PhysicalSizeX = float(meta['PhysicalSizeX'])
    PhysicalSizeY = float(meta['PhysicalSizeY'])
    PhysicalSizeZ = float(meta['PhysicalSizeZ'])

    # Get info
    nlabels = int(detections_df['label'].max())
    nframes = int(detections_df['t'].nunique())

    if debug:
        ndetections = debug
        print ('* Debug mode, using only  ' + str(debug) + ' detections')
    else:
        ndetections = len(detections_df[detections_df['t'] == 0])

    if debug:
        # For tests, using subset of frames:
        frames_to_run = 1
        print ('* Debug mode, using only first frame')
    else:
        frames_to_run = nframes

    # Create empty intensity
    detections_df['intensity'] = 0

    # Check consistency of dataframe. The number of detections must be the same for all frames
    if verbose:
        print ('Checking dataframe consistency...'),
        sys.stdout.flush()
    first_frame_shape = detections_df[detections_df['t'] == 0].shape
    for frame in range(frames_to_run):
        if first_frame_shape == detections_df[detections_df['t'] == frame].shape:
            pass
        else:
            print ('Found problem on frame' + frame)
            print ('Desired shape:' + str(first_frame_shape))
            print ('Found: ' + str(detections_df[detections_df['t'] == frame].shape))
            return
    if verbose:
        print ('[OK]')
        sys.stdout.flush()

    if save_voronoi:
        # Generate blank image
        nslices = int((float(meta['SizeZ']) * float(meta['PhysicalSizeZ'])) / float(meta['PhysicalSizeX']))
        voronoi_img = np.zeros([frames_to_run, nslices, meta['SizeY'], meta['SizeX']], dtype='uint16')

    # Start to read intensity values
    if verbose:
        print('Reading frames:'),
        sys.stdout.flush()

    for frame in range(frames_to_run):
        if verbose:
            print('[' + str(frame) + ']'),
            sys.stdout.flush()

        # Get detected points for this time frame
        temp_df = detections_df[detections_df['t'] == frame]
        label_coords = np.transpose([temp_df['z'], temp_df['y'], temp_df['x']])

        # Generate KDTree
        if debug:
            print ('Generating KDtree...'),
        kdtree = KDTree(label_coords)
        if debug:
            print('[Done]')

        # Load image for current frame
        signal_img = memotrack.io.read(img_path, meta, frame=frame, verbose=False, channel=1)

        # Interpolation, because coordinates must match !
        if PhysicalSizeX != PhysicalSizeZ:
            nslices = int((SizeZ * PhysicalSizeZ) / PhysicalSizeX)
            zoom_ratio = float(nslices) / float(SizeZ)
            signal_img = scipy.ndimage.interpolation.zoom(signal_img, [zoom_ratio, 1, 1], order=spline_order)
        # Get new size after interpolation
        meta['SizeZinterpolated'] = np.shape(signal_img)[0]

        for detection in range(ndetections):
            # Get position on dataframe. Tricky but works
            position = (detection * nframes) + frame

            # print ('\n*** df so far:')
            # print (detections_df)
            # Get coordinates
            z = detections_df['zsmooth'].loc[position]
            y = detections_df['ysmooth'].loc[position]
            x = detections_df['xsmooth'].loc[position]
            label = detections_df['label'].loc[position]
            # Here the coordinates for this label, at this time frame
            coords = [z, y, x]

            coords_list = memotrack.process.coords_in_range(coords, meta, FWHMxy, label=label, kdtree=kdtree,
                                                            pixel_size=PhysicalSizeX, verbose=False)

            if not coords_list:
                print ('\nWARNING: Empty list of coordinates to read intensities !')
                print ('problem with detection ' + str(detection)),
                print ('at position ' + str(position))
                print ('x:' + str(x) + ' y:' + str(y) + ' z:' + str(z))

            # setting values on the voronoi img array
            if save_voronoi:
                for ZYXcoords in coords_list:
                    voronoi_img[int(frame), int(ZYXcoords[0]), int(ZYXcoords[1]), int(ZYXcoords[2])] = int(detection)

            # Reset intensity
            intensity_list = []

            # Grab intensity from all coordinates in range
            for ZYXcoords in coords_list:
                intensity_list.append(signal_img[ZYXcoords[0], ZYXcoords[1], ZYXcoords[2]])

            average_intensity = np.average(intensity_list)
            detections_df.loc[position, 'intensity'] = average_intensity

    # Saving voronoi image to disk
    if save_voronoi:
        print ('voronoi max: ' + str(np.max(voronoi_img)))
        save_voronoi_path = img_path[:-4] + '_voronoi.tif'
        if verbose:
            print ('Saving voronoi...'),
        # memotrack.io.write(voronoi_img, save_voronoi_path)
        tifffile.imsave(save_voronoi_path, voronoi_img[0])
        if verbose:
            print ('[Done]')
            print (save_voronoi_path)
    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    return detections_df


def get_signal_voronoi(detections_df, img_path, verbose=True, FWHMxy=1.21, signalFWHM=2.0, debug=False, spline_order=3,
                       truncate=False):
    import memotrack.io
    import sys
    import numpy as np
    import scipy.ndimage
    import tifffile as tiff
    import math

    # Reading signal metadata
    meta = memotrack.io.meta(img_path, verbose=False)

    # Setting variables for img size
    SizeT = meta['SizeT']
    SizeZ = meta['SizeZ']
    SizeY = meta['SizeY']
    SizeX = meta['SizeX']
    SizeC = meta['SizeC']
    PhysicalSizeX = float(meta['PhysicalSizeX'])
    PhysicalSizeY = float(meta['PhysicalSizeY'])
    PhysicalSizeZ = float(meta['PhysicalSizeZ'])

    # Get info
    nlabels = int(detections_df['label'].max())
    nframes = int(detections_df['t'].nunique())
    ndetections = len(detections_df[detections_df['t'] == 0])

    if debug:
        # For tests, using subset of frames:
        frames_to_run = debug
        print ('* Debug mode, using only ' + str(debug) + ' first frames')

    elif truncate:
        frames_to_run = truncate
        print ('* Truncate mode, using only ' + str(truncate) + ' first frames')
    else:
        frames_to_run = nframes

    # Create empty intensity
    detections_df['intensity'] = 0

    # Check consistency of dataframe. The number of detections must be the same for all frames
    if verbose:
        print ('Checking dataframe consistency...'),
        sys.stdout.flush()
    first_frame_shape = detections_df[detections_df['t'] == 0].shape
    for frame in range(frames_to_run):
        if first_frame_shape == detections_df[detections_df['t'] == frame].shape:
            pass
        else:
            print ('Found problem on frame' + frame)
            print ('Desired shape:' + str(first_frame_shape))
            print ('Found: ' + str(detections_df[detections_df['t'] == frame].shape))
            return
    if verbose:
        print ('[OK]')
        sys.stdout.flush()

    # Load the full image
    full_img = memotrack.io.load_full_img(img_path, verbose=True)

    if truncate:
        trunc_frame = int(truncate)
        full_img = full_img[:trunc_frame]

        print ('Truncating image to frame {}'.format(trunc_frame))
        print ('New shape: {}'.format(np.shape(full_img)))

    # Start to read intensity values
    if verbose:
        print('Reading frames:'),
        sys.stdout.flush()

    for frame in range(frames_to_run):
        if verbose:
            print('[' + str(frame) + ']'),
            sys.stdout.flush()

        # Load image for current frame
        # signal_img = memotrack.io.read(img_path, meta, frame=frame, verbose=False, channel=1)
        signal_img = full_img[frame, :, 1, :, :]  # signal channel is 1

        # Interpolation, because coordinates must match !
        if PhysicalSizeX != PhysicalSizeZ:
            nslices = int((SizeZ * PhysicalSizeZ) / PhysicalSizeX)
            zoom_ratio = float(nslices) / float(SizeZ)
            signal_img = scipy.ndimage.interpolation.zoom(signal_img, [zoom_ratio, 1, 1], order=spline_order)
        # Get new size after interpolation
        meta['SizeZinterpolated'] = np.shape(signal_img)[0]

        if debug:
            print ('\nStarting voronoi tesselation'),
            sys.stdout.flush()

        df_temp = detections_df[detections_df['t'] == frame]

        # Create the labeled array of the constrained voronoi regions
        voronoi_max_size = ((FWHMxy / PhysicalSizeX) * signalFWHM)
        labeled_array = memotrack.process.generate_voronoi(df_temp, meta, max_distance=voronoi_max_size,
                                                           verbose=False, debug=False)

        if debug:
            voronoi_save_path = img_path[:-4] + '_voronoi_' + str(frame) + '.tif'
            tiff.imsave(voronoi_save_path, labeled_array)
            print ('[Done]')
            print ('')
            print ('labeled array: ' + str(np.shape(labeled_array)))
            print ('labeled array max: ' + str(np.max(labeled_array)))
            print ('signal img: ' + str(np.shape(signal_img)))
            print ('signal img max: ' + str(np.max(signal_img)))
            sys.stdout.flush()

        # Apply filter to signal, before getting the intensity
        signal_img = scipy.ndimage.median_filter(signal_img, size=5)

        # WARNING: Here i'm changing "mean" to "maximum", but keeping the "avg_intensity_list" variable name
        avg_intensity_list = scipy.ndimage.maximum(signal_img, labels=labeled_array, index=range(np.max(labeled_array)))
        avg_value = np.nanmean(avg_intensity_list)

        # handle NaNs
        avg_intensity_list = [avg_value if math.isnan(x) else x for x in avg_intensity_list]
        if debug:
            print ('\nintensity list shape: ' + str(np.shape(avg_intensity_list)))
            print ('values: ' + str(np.sort(avg_intensity_list)))
            print ('avg value: ' + str(avg_value))
            print ('')

        # Add values to dataframe
        for detection in range(np.max(labeled_array)):
            # Get position on dataframe. Tricky but works
            position = (detection * nframes) + frame
            # print (str(detection)+':\t'+str(avg_intensity_list[detection]))
            detections_df.loc[position, 'intensity'] = avg_intensity_list[detection]

    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    del full_img

    return detections_df


def get_signal_old(detections_df, img_path, verbose=True, FWHMxy=1.21, debug=False, spline_order=3):
    import memotrack.io
    import sys
    import numpy as np
    import scipy.ndimage

    # Reading signal metadata
    meta = memotrack.io.meta(img_path, verbose=False)

    # Setting variables for img size
    SizeT = meta['SizeT']
    SizeZ = meta['SizeZ']
    SizeY = meta['SizeY']
    SizeX = meta['SizeX']
    SizeC = meta['SizeC']
    PhysicalSizeX = float(meta['PhysicalSizeX'])
    PhysicalSizeY = float(meta['PhysicalSizeY'])
    PhysicalSizeZ = float(meta['PhysicalSizeZ'])

    # Get info
    nlabels = int(detections_df['label'].max())
    nframes = int(detections_df['t'].nunique())
    ndetections = len(detections_df[detections_df['t'] == 0])

    if debug:
        # For tests, using subset of frames:
        frames_to_run = debug
        print ('* Debug mode, using only ' + str(debug) + ' first frames')
    else:
        frames_to_run = nframes

    # Create empty intensity
    detections_df['intensity'] = 0

    # Check consistency of dataframe. The number of detections must be the same for all frames
    if verbose:
        print ('Checking dataframe consistency...'),
        sys.stdout.flush()
    first_frame_shape = detections_df[detections_df['t'] == 0].shape
    for frame in range(frames_to_run):
        if first_frame_shape == detections_df[detections_df['t'] == frame].shape:
            pass
        else:
            print ('Found problem on frame' + frame)
            print ('Desired shape:' + str(first_frame_shape))
            print ('Found: ' + str(detections_df[detections_df['t'] == frame].shape))
            return
    if verbose:
        print ('[OK]')
        sys.stdout.flush()

    # Start to read intensity values
    if verbose:
        print('Reading frames:'),
        sys.stdout.flush()

    for frame in range(frames_to_run):
        if verbose:
            print('[' + str(frame) + ']'),
            sys.stdout.flush()

        # Load image for current frame
        signal_img = memotrack.io.read(img_path, meta, frame=frame, verbose=False, channel=1)

        # Interpolation, because coordinates must match !
        if PhysicalSizeX != PhysicalSizeZ:
            nslices = int((SizeZ * PhysicalSizeZ) / PhysicalSizeX)
            zoom_ratio = float(nslices) / float(SizeZ)
            signal_img = scipy.ndimage.interpolation.zoom(signal_img, [zoom_ratio, 1, 1], order=spline_order)
        # Get new size after interpolation
        meta['SizeZinterpolated'] = np.shape(signal_img)[0]

        for detection in range(ndetections):
            # Get position on dataframe. Tricky but works
            position = (detection * nframes) + frame

            # print ('\n*** df so far:')
            # print (detections_df)
            # Get coordinates
            z = detections_df['zsmooth'].loc[position]
            y = detections_df['ysmooth'].loc[position]
            x = detections_df['xsmooth'].loc[position]

            # Here the coordinates for this label, at this time frame
            coords = [z, y, x]

            coords_list = memotrack.process.coords_in_range(coords, meta, FWHMxy, pixel_size=0.16125)

            if not coords_list:
                print ('\nWARNING: Empty list of coordinates to read intensities !')
                print ('problem with detection ' + str(detection)),
                print ('at position ' + str(position))
                print ('x:' + str(x) + ' y:' + str(y) + ' z:' + str(z))

            # Reset intensity
            intensity_list = []

            # Grab intensity from all coordinates in range
            for ZYXcoords in coords_list:
                intensity_list.append(signal_img[ZYXcoords[0], ZYXcoords[1], ZYXcoords[2]])

            average_intensity = np.average(intensity_list)
            detections_df.loc[position, 'intensity'] = average_intensity

    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    return detections_df


def coords_in_range(coords, meta, FWHMxy, label=False, kdtree=False, pixel_size=0.16125, verbose=False):
    import scipy.spatial.distance as distance_func
    distance = int((FWHMxy / 2) / pixel_size)
    SizeZ = int(meta['SizeZinterpolated'])
    SizeY = int(meta['SizeY'])
    SizeX = int(meta['SizeX'])

    if verbose:
        print('\nLabel: ' + str(int(label)) + '\tMax distance: ' + str(distance))

    coords_list = []

    for z in range(-distance, distance + 1, 1):
        # Check if Z value is within image
        if 0 < (coords[0] + z) < SizeZ:
            for y in range(-distance, distance + 1, 1):
                # Check if Y value is within image
                if 0 < (coords[1] + y) < SizeY:
                    for x in range(-distance, distance + 1, 1):
                        # Check if X value is within image
                        if 0 < (coords[2] + x) < SizeX:
                            # Calculates distance between coordinates
                            # This assures it's a sphere, not an cube.
                            dist_coords = float(distance_func.pdist(
                                ([coords[0], coords[1], coords[2]], [z + coords[0], y + coords[1], x + coords[2]])))
                            if dist_coords <= distance:
                                current_cords = [int(coords[0] + z), int(coords[1] + y), int(coords[2] + x)]
                                # Check closest label on KDtree. This is equivalent of Voronoi region
                                if kdtree:
                                    closest_dist, closest_label = kdtree.query(current_cords)
                                    if closest_label == label:
                                        coords_list.append(current_cords)
                                else:
                                    coords_list.append(current_cords)

    return coords_list


def estimate_DBSCAN_eps(detections_df, verbose=True, debug=False):
    import scipy.stats
    import numpy as np
    import scipy.spatial.distance
    import scipy.stats as stats

    # Get number of frames on dataframe
    nframes = int(detections_df['t'].nunique())

    # Mean distances list
    mean_distances = []

    # Threshold for pvalue
    pvalue_threshod = 0.05

    # Initiate pvalue
    pvalue = 1

    # Initiate counter
    counter = 0

    if verbose:
        print('Calculating distances between pairs:'),

    while pvalue > pvalue_threshod:
        frame1 = 0
        frame2 = 0
        while frame1 == frame2:
            # Random frames
            frame1 = np.random.randint(0, nframes)
            frame2 = np.random.randint(0, nframes)

        if verbose:
            print ('[' + str(frame1) + ' & ' + str(frame2) + ']'),

        # Get reference dataframe
        df_frame1 = detections_df[detections_df['t'] == frame1].copy(deep=True)
        xyz_frame1 = df_frame1[['xreg', 'yreg', 'zreg']].values

        # Open dataframe to compare
        df_frame2 = detections_df[detections_df['t'] == frame2].copy(deep=True)
        xyz_frame2 = df_frame2[['xreg', 'yreg', 'zreg']].values

        # Distances
        dist = scipy.spatial.distance.cdist(xyz_frame1, xyz_frame2)
        # Sorting matrix to get only the closest points
        dist = np.sort(dist, axis=0)
        dist = dist[0]
        # Take the average
        dist = np.average(dist)
        # Append value to list
        mean_distances.append(dist)

        # Check normality of distribution
        if len(mean_distances) > 100:
            statistic, pvalue = stats.mstats.normaltest(mean_distances)

            if debug:
                print ('\tpvalue: ' + str(pvalue))

        counter += 1
        if counter > 1000:
            if verbose:
                print ('\nOne thousand pairs analysed, stopping with pvalue of ' + str(pvalue))
            pvalue = 0

    global_mean = np.average(mean_distances)

    estimated_eps = global_mean / 2

    if verbose:
        print ('\nDone after ' + str(counter) + ' iterations, normal distribution with pvalue of ' + str(pvalue))
        print ('\nGlobal mean distance:\t' + str(global_mean))
        print ('Estimated eps value:\t' + str(estimated_eps))

    return estimated_eps


def estimate_DBSCAN_min_samples(detections_df, estimated_eps, avg_detections, verbose=True, display=True):
    import memotrack.process
    import memotrack.display

    if verbose:
        print('Target: ' + str(avg_detections) + ' clusters\n')

    # min_samples to start
    initial_nsamples = 1

    # Start counter
    counter = 0

    # Maximum number of consecutive trials before giving up
    max_trials = 10

    # Difference from goal, to minimize
    clusters_dif = float('inf')

    # List of differences between dbscan and target
    diff_list = []

    nsamples = initial_nsamples
    while counter < max_trials:
        # Increase min_samples

        # Run DBSCAN
        detections_df, nlabels = memotrack.process.dbscan(detections_df, eps=estimated_eps, min_samples=nsamples,
                                                          FWHM=False, verbose=False)

        current_clusters_dif = abs(nlabels - avg_detections)

        diff_list.append(current_clusters_dif)

        if verbose:
            print (str(nlabels) + ' clusters for min_samples of ' + str(nsamples) + '\tDifference from target: ' + str(
                current_clusters_dif))

        if current_clusters_dif < clusters_dif:
            estimated_min_samples = nsamples
            clusters_dif = current_clusters_dif
            # Reset counter
            counter = 0
        else:
            #  Case fails to find a better case, increase the counter
            counter += 1

        # Increase min_samples for the next run
        nsamples += 1

    if verbose:
        print ('\nBest value for min_samples: ' + str(estimated_min_samples))

    if display:
        memotrack.display.plot1Dline(diff_list, x_label='min_samples', y_label='Difference from target',
                                     xticks=[initial_nsamples, nsamples], linewidth=4, img_size=7,
                                     title='MinSamples estimation for DBSCAN')

    return estimated_min_samples


def handle_duplicates(detections_df, verbose=True, debug=True):
    import numpy as np
    from scipy import stats
    import scipy.spatial.distance
    import sys
    from sklearn.cluster import KMeans

    # Get number of frames
    nframes = int(detections_df['t'].nunique())
    nlabels = int(detections_df['label'].nunique())

    # Remove points labeled as noise
    detections_df_copy = detections_df[detections_df.label != -1].copy(deep=True)

    # Get only double detections
    double_detections = detections_df_copy[detections_df_copy.duplicated(subset=['t', 'label'])]

    # List of labels for duplicates
    duplicates_labels = double_detections['label'].copy(deep=True)
    duplicates_labels.drop_duplicates(inplace=True)  # We want only one of each, for a list of duplicated labels
    duplicates_labels_list = duplicates_labels.values  # Getting the values as numpy
    duplicates_labels_list = np.sort(duplicates_labels_list)
    # print ('Duplicates labels: ' + str(duplicates_labels_list))
    # Get total number of labels that have duplicates somewhere
    nduplicates = np.shape(duplicates_labels_list)[0]

    if nduplicates < 1:
        print ('There are no duplicates on the dataframe !')
        return detections_df, nduplicates

    # for tests, work only on the first n duplicates
    # if debug:
    #    nduplicates = 100

    if verbose:
        print('Number of labels with duplicates: ' + str(nduplicates) + ' (' +
              str(int((float(nduplicates) / nlabels) * 100)) + '%)')

    # Initiate list of modal values for every label
    mode_list = []

    if verbose:
        print('Resolving duplicates'),
        sys.stdout.flush()
        fancy_dots = 0

    # Beginning to work. Running through all the duplicates on the list
    for duplicate_pos in range(nduplicates):
        if verbose:
            if fancy_dots > 10:
                print('.'),
                sys.stdout.flush()
                fancy_dots = 0

        # Get the label to work on
        duplicate_label = duplicates_labels_list[duplicate_pos]
        # Crop the dataframe to the label we need
        one_label = detections_df[detections_df.label == duplicate_label].copy(deep=True)
        # Start an array to get the number of duplicates on every time frame
        temp_duplicates_list = np.zeros(nframes)

        # Run through frames to get the number of duplicates for this label
        for frame in range(nframes):
            temp_duplicates_list[frame] = len(one_label[one_label['t'] == frame])

        # Mode of the number of duplicates.
        # If the mode is 0 or 1, we can handle the duplicates by getting the closest to the centroid
        # If the mode is 2 or bigger, we need to re-cluster the label
        temp_mode = int(stats.mode(temp_duplicates_list)[0])
        # Append, to create an list with the modes for all the cases
        mode_list.append(temp_mode)

        # Check if the mode for the current duplicate is 0 or 1 (cases solved by average value)
        if temp_mode <= 1:
            # Get centroid
            centroid_x = one_label['x'].mean()
            centroid_y = one_label['y'].mean()
            centroid_z = one_label['z'].mean()
            centroid_xyz = [centroid_x, centroid_y, centroid_z]
            if debug:
                print ('Cluster centroid: ' + str([centroid_x, centroid_y, centroid_z]))
            # Run through all frames
            for frame in range(nframes):
                # Check if it have 2 or more duplicates
                if temp_duplicates_list[frame] >= 2:
                    # Get only the duplicates
                    temp_df = one_label[one_label['t'] == frame]

                    # The indices from the duplicates
                    indices = temp_df.index.values
                    # print ('Indices of duplicates: ' + str(indices))
                    # Run through the duplicates
                    distances = []
                    for i in range(len(indices)):
                        detection = detections_df.loc[indices[i]]

                        # Get coords from df
                        detection_x = detection['x']
                        detection_y = detection['y']
                        detection_z = detection['z']
                        detection_xyz = [detection_x, detection_y, detection_z]

                        # Calculate distance to the centroid
                        distances.append(scipy.spatial.distance.euclidean(centroid_xyz, detection_xyz))

                    # Get closest detection
                    best_index = indices[(distances.index(min(distances)))]

                    # Put bad duplicates as noise
                    for index in indices:
                        # Remove only if it isn't the best !
                        if index != best_index:
                            detections_df.loc[index, 'label'] = -1

                    if debug:
                        print ('Duplicate on frame ' + str(frame))
                        print(temp_df)
                        print ('Index closer to the centroid: ' + str(best_index))

        # Check if mode for this label is bigger than 1 (case solved by re-clustering)
        if temp_mode > 1:
            if debug:
                print('Starting K-Means...')
            # Get coordinates for this label
            coords = one_label.as_matrix(columns=['x', 'y', 'z'])

            # Create kmeans estimator
            estimator = KMeans(n_clusters=temp_mode)

            # Run K-means, getting new labels for the coordinates
            new_labels = estimator.fit_predict(coords)

            # These new labels start from 0 and go to the number of clusters
            # So, for mode=2 (2 clusters) the new labels are 0 and 1
            # We need to fix this before sending they back to detections_df, to avoid conflicts
            original_label = one_label['label'].max()  # as all values are the same, max does the job
            last_label = detections_df['label'].max()  # The last label on the original dataframe
            if debug:
                print('Original label: ' + str(original_label))
                print('Last label on df: ' + str(last_label))

            # Running through the new labels
            # remembering that the number of clusters from kmeans is equal to the mode
            for i in range(len(new_labels)):
                # Here we keep the first k-means cluster with the original dbscan label
                if new_labels[i] == 0:
                    new_labels[i] = original_label
                # For the other cases, we create a new label after the last on the original df
                else:
                    new_labels[i] = last_label + new_labels[i]

            if debug:
                print ('New labels: ' + str(new_labels))

            # The indices from the duplicates
            indices = one_label.index.values

            # Run through the indices changing the labels
            for i in range(len(new_labels)):
                index = indices[i]
                # Replace the labels on the original df
                detections_df.loc[index, 'label'] = new_labels[i]

        # Just a summary for debug
        if debug:
            print ('Label n.' + str(duplicate_label) + '\tMode: ' + str(temp_mode))
            print (temp_duplicates_list)
            print ('\n' * 3)

        fancy_dots += 1

    # Overview
    if verbose:
        count = np.bincount(mode_list)
        print('')
        for n in range(len(count)):
            print (str(count[n]) + ' cases with mode ' + str(n) + ' (' + str(
                int((float(count[n]) / nduplicates) * 100)) + '%)')

    # Look again for remaining duplicates
    # Remove points labeled as noise
    detections_df_copy = detections_df[detections_df.label != -1].copy(deep=True)

    # Get only double detections
    double_detections = detections_df_copy[detections_df_copy.duplicated(subset=['t', 'label'])]

    # List of labels for duplicates
    duplicates_labels = double_detections['label'].copy(deep=True)
    duplicates_labels.drop_duplicates(inplace=True)  # We want only one of each, for a list of duplicated labels
    duplicates_labels_list = duplicates_labels.values  # Getting the values as numpy
    duplicates_labels_list = np.sort(duplicates_labels_list)

    # Get total number of labels that have duplicates somewhere
    remaining_duplicates = np.shape(duplicates_labels_list)[0]

    if verbose:
        print('\nRemaining duplicates: ' + str(remaining_duplicates))

    return detections_df, remaining_duplicates


def convert16to8(img, vmax=65536):
    """
    Converts numpy array from uint16 to uint8
    :param img: numpy array, n-dimensional
    :param vmax: Normalization value. Use 65535 for uint16 max
    :return: 8 bit array
    """

    import numpy as np
    # Convert to 8bit
    img_8bit = img[:] / float(vmax)
    img_8bit = img_8bit[:] * 255
    img_8bit = np.asarray(img_8bit, dtype='uint8')

    return img_8bit


def intensity_normalization_deltaf(detections_df, verbose=True, debug=False):
    import numpy as np
    import pandas as pd
    import sys

    # create column for normalized intensity
    detections_df['norm_intensity'] = float('nan')

    if verbose:
        print ('\nInitializing intensity normalization using deltaF/F0')
        sys.stdout.flush()

    nlabels = detections_df['label'].nunique()
    max_list = []
    for label in range(nlabels):
        if verbose:
            print ('.'),
            sys.stdout.flush()
        label_df = detections_df[detections_df['label'] == label].copy(deep=True)
        original = label_df['filtered'].get_values()
        baseline = pd.rolling_median(original, 20, min_periods=0, center=True)
        normalized = np.divide((original - baseline), baseline)
        # Do we have NaNs ?
        normalized[np.isnan(normalized)] = 0
        #
        detections_df.loc[detections_df.label == label, 'norm_intensity'] = normalized

        # Distribution using normalized signal
        max_list.append(np.max(normalized))


        # Distribution usig df normalized signal
        # print ('list:' + str(temp))
        # max_list.append(np.max(temp))
    # Calculate median and std of peaks distribution
    max_list_median = np.median(max_list)
    max_list_std = np.std(max_list)
    responsive_threshold = max_list_median + (max_list_std * 1)  # getting 3 standard deviations

    print ('Responsive threshold: {}'.format(responsive_threshold))
    detections_df.fillna(value=0, inplace=True)

    # Now pass again labeling the neurons
    detections_df['responsive'] = False
    for label in range(nlabels):
        normalized_signal = detections_df[detections_df['label'] == label].norm_intensity.values

        neuron_peak = np.max(normalized_signal)

        if neuron_peak > responsive_threshold:
            detections_df.loc[(detections_df.label == label), 'responsive'] = True
            # print ('\nNeuron ' + str(label) + ' is responsive')

    return detections_df


def intensity_normalization_distance(detections_df, dist=3, FWHM=1.33, pixel_size=0.16125, verbose=True, debug=False):
    import scipy.spatial
    import numpy as np
    import sys

    size = FWHM / pixel_size

    # create column for normalized intensity
    detections_df['norm_intensity'] = float('nan')

    if verbose:
        print ('\nInitializing intensity normalization using neighbours closer than ' + str(dist) + ' diameters')
        sys.stdout.flush()

    nframes = detections_df['t'].nunique()
    nlabels = detections_df['label'].nunique()
    if debug:
        nframes = debug
        print ('WARNING: Debug mode enabled !')
        print ('frames: ' + str(nframes))
        print ('labels: ' + str(nlabels))
        sys.stdout.flush()

    for frame in range(nframes):
        if verbose:
            print ('[' + str(frame) + ']'),
            sys.stdout.flush()
        frame_df = detections_df[detections_df['t'] == frame].copy(deep=True)
        # Create distance tree
        tree = scipy.spatial.KDTree(frame_df[['x', 'y', 'z']])

        # Frame median intensity:
        frame_median_intensity = frame_df.median().intensity

        if debug:
            print ('Frame median intensity: ' + str(frame_median_intensity))
            sys.stdout.flush()
        for label in range(nlabels):
            label_df = frame_df[frame_df['label'] == label].copy(deep=True)
            # For k neighbours
            # d, close_labels = tree.query(label_df[['x', 'y', 'z']], k=k, distance_upper_bound=size * 10)
            close_labels = tree.query_ball_point(label_df[['x', 'y', 'z']], r=size * dist)
            if debug:
                print ('\nlabel: ' + str(label))
                # print (np.sort(close_labels[0])),
                sys.stdout.flush()

            # Get intensity for close neighbours
            label_values = []
            for match_label in close_labels[0]:
                label_values.append(frame_df[frame_df['label'] == match_label]['intensity'].values)

            # needed only in case of k neighbours
            # label_values = [s for s in label_values if s]

            # number of neighbours in range
            nvalues = np.count_nonzero(label_values)
            if debug:
                print ('Values in range: ' + str(nvalues))
                sys.stdout.flush()
            # median of the intensity values of the k neighbours
            neighbour_median = np.median(label_values)
            # the value of the label itself
            label_intensity = frame_df[frame_df['label'] == label]['intensity'].values

            if debug:
                print ('Neighbour median: ' + str(neighbour_median))
                print ('Label intensity: ' + str(label_intensity))
                sys.stdout.flush()

            # Normalize intensity
            if nvalues > 3:
                normalized = label_intensity / neighbour_median
            else:
                normalized = label_intensity / frame_median_intensity
            if debug:
                print('Normalized intensity: ' + str(normalized))
                sys.stdout.flush()
            detections_df.loc[
                (detections_df.t == frame) & (detections_df.label == label), 'norm_intensity'] = normalized

    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    return detections_df


def intensity_normalization_amount(detections_df, k=12, FWHM=1.33, pixel_size=0.16125, verbose=True, debug=False):
    import scipy.spatial
    import numpy as np
    import sys

    size = float(FWHM) / float(pixel_size)

    # create column for normalized intensity
    detections_df['norm_intensity'] = float('nan')

    if verbose:
        print ('\nInitializing intensity normalization using ' + str(k) + ' neighbours')
        sys.stdout.flush()

    nframes = detections_df['t'].nunique()
    nlabels = detections_df['label'].nunique()

    if debug:
        nframes = debug
        nlabels = debug
        print ('WARNING: Debug mode enabled !')
        print ('frames: ' + str(nframes))
        print ('labels: ' + str(nlabels))
        sys.stdout.flush()

    for frame in range(nframes):
        if verbose:
            print ('[' + str(frame) + ']'),
            sys.stdout.flush()
        frame_df = detections_df[detections_df['t'] == frame].copy(deep=True)
        # Create distance tree
        tree = scipy.spatial.KDTree(frame_df[['x', 'y', 'z']])

        # Frame median intensity:
        frame_median_intensity = frame_df.median().intensity

        if debug:
            print ('Frame median intensity: ' + str(frame_median_intensity))
            sys.stdout.flush()
        for label in range(nlabels):
            label_df = frame_df[frame_df['label'] == label].copy(deep=True)
            # For k neighbours
            d, close_labels = tree.query(label_df[['x', 'y', 'z']], k=k, distance_upper_bound=size * 10)

            if debug:
                print ('\nlabel: ' + str(label))
                # print (np.sort(close_labels[0])),
                sys.stdout.flush()

            # Get intensity for close neighbours
            label_values = []
            for match_label in close_labels[0]:
                label_values.append(frame_df[frame_df['label'] == match_label]['intensity'].values)

            # needed only in case of k neighbours
            label_values = [s for s in label_values if s]

            # number of neighbours in range
            nvalues = np.count_nonzero(label_values)
            if debug:
                print ('Values in range: ' + str(nvalues))
                sys.stdout.flush()
            # median of the intensity values of the k neighbours
            neighbour_median = np.median(label_values)
            # the value of the label itself
            label_intensity = frame_df[frame_df['label'] == label]['intensity'].values

            if debug:
                print ('Neighbour median: ' + str(neighbour_median))
                print ('Label intensity: ' + str(label_intensity))
                sys.stdout.flush()

            # Normalize value
            normalized = label_intensity / neighbour_median

            if debug:
                print('Normalized intensity: ' + str(normalized))
                sys.stdout.flush()
            detections_df.loc[
                (detections_df.t == frame) & (detections_df.label == label), 'norm_intensity'] = normalized

    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    return detections_df


def band_pass(data, lowcut=0.0, highcut=1.0, fs=100, order=2, display=False):
    import scipy.signal
    import memotrack.display
    import numpy as np
    import sys

    # nyq = 0.5 * fs
    low = lowcut  # / nyq
    high = highcut  # / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='bandpass', analog=False)

    # remove NaNs
    data[np.isnan(data)] = 0
    filtered_signal = scipy.signal.filtfilt(b, a, data, method='gust', irlen=10)

    if display:
        memotrack.display.plot1Dline(filtered_signal, ylim=[-2, 2])
        return

    return filtered_signal


def apply_filter(detections_df, lowcut=0.02, highcut=0.3333, verbose=False, display=False, thresh=3,
                 window_start=25, window_end=50):
    import memotrack.process
    import memotrack.analyse
    import memotrack.display
    import numpy as np

    # Create new column
    detections_df['filtered'] = float('nan')

    nlabels = detections_df.label.nunique()

    '''
    max_list = []
    detections_df['responsive'] = False
    detections_df['stim_responsive'] = False
    '''

    # First get the distribution
    for label in range(nlabels):
        temp = detections_df[detections_df['label'] == label].intensity.values

        filtered_signal = memotrack.process.band_pass(temp, lowcut=lowcut, highcut=highcut, order=2)

        # Save filtered signal to dataframe
        detections_df.loc[(detections_df.label == label), 'filtered'] = filtered_signal

        # Distribution using filtered signal
        # max_list.append(np.max(filtered_signal))

        # Distribution usig df normalized signal
        # print ('list:' + str(temp))
        # max_list.append(np.max(temp))

    '''
    # Calculate median and std of peaks distribution
    max_list_median = np.median(max_list)
    max_list_std = np.std(max_list)
    responsive_threshold = max_list_median + (max_list_std * thresh)

    if verbose:
        print ('Using ' + str(len(max_list)) + ' values for calculation.')
        print ('Median: ' + str(max_list_median))
        print ('Std: ' + str(max_list_std))
        print ('Estimated threshold: ' + (str(responsive_threshold) + ' (for ' + str(thresh) + ' std)'))

    # Now pass again labeling the neurons
    for label in range(nlabels):
        filtered_signal = detections_df[detections_df['label'] == label].filtered.values

        neuron_peak = np.max(filtered_signal)

        if neuron_peak > responsive_threshold:
            detections_df.loc[(detections_df.label == label), 'responsive'] = True
            # print ('\nNeuron ' + str(label) + ' is responsive')

        filtered_signal_window = filtered_signal[window_start:window_end]
        neuron_peak_window = np.max(filtered_signal_window)

        if neuron_peak_window > responsive_threshold:
            detections_df.loc[(detections_df.label == label), 'stim_responsive'] = True
            # print ('* Responsive inside window')


    if verbose:
        nresponsive = detections_df[detections_df['responsive'] == True].label.nunique()
        nstimresponsive = detections_df[detections_df['stim_responsive'] == True].label.nunique()
        responsive_labels = detections_df[detections_df['responsive'] == True].label.unique()
        stim_responsive_labels = detections_df[detections_df['stim_responsive'] == True].label.unique()
        print (
            str(nresponsive) + ' responsive neurons on whole sequence (' + str(
                float(nresponsive) * 100 / nlabels) + '%)')
        print ('Labels: '),
        for label in responsive_labels:
            print (int(label)),
        print ('')
        print ('')
        print (str(nstimresponsive) + ' responsive neurons on stimulation window (' + str(
            float(nstimresponsive) * 100 / nlabels) + '%)')
        print ('Labels: '),
        for label in stim_responsive_labels:
            print (int(label)),
        print ('')
    '''
    return detections_df


def inter_responsive(files_path, verbose=False, display=False, nbins=40, auto_bins=True, window_start=25):
    import numpy as np
    import pandas as pd
    import memotrack
    from scipy import stats

    # Get list of paths
    text_file = open(files_path, 'r')
    path_list = text_file.read().split('\n')
    text_file.close()

    fly_list = []
    for path in path_list:
        if len(path) > 0:
            fly_name = path[-13:-7]
            fly_list.append(fly_name)

    fly_list_unique = set(fly_list)

    if verbose:
        print ('Analysing flies:')
        for fly in fly_list_unique:
            print(fly),
        print ('')
    # Run through each fly and create the super data frame
    total_air_neurons = 0
    total_oct_neurons = 0
    total_mch_neurons = 0
    oct_ratio_list = []
    mch_ratio_list = []
    for fly in fly_list_unique:
        temp_air_neurons = 0
        temp_oct_neurons = 0
        temp_oct_ratio = 0
        temp_mch_neurons = 0
        temp_mch_ratio = 0

        # Get paths for this fly
        fly_paths = []
        for path in path_list:
            if path[-13:-7] == fly:
                fly_paths.append(path)
        if verbose:
            print ('\nFly ' + str(fly))

        # Run each path to create the super dataframe
        detections_df_list = []
        for path in fly_paths:
            if verbose:
                print (path)
            df_path = (str(path[:-4]) + '_detections_labels_fixed_signal_normalized_filtered.csv')
            detections_df = memotrack.io.read_df(df_path)
            detections_df_list.append(detections_df)

        super_df = pd.concat(detections_df_list)

        # Go get those bins automatically !
        if auto_bins:
            nbins = super_df.label.nunique()
            if verbose:
                print ('Using ' + str(nbins) + ' bins')

        super_filtered = super_df['filtered'].values
        super_filtered_positive = [y for y in super_filtered if y > 0.0]
        # super_norm_intensity[super_norm_intensity < 0] = 0

        hist, bin_edges = np.histogram(super_filtered_positive, bins=nbins)

        # Linear regression to find the "zero value" of the histogram
        hist_log = np.log10(hist + 0.0001)
        slope, intercept, r_value, p_value, std_err = stats.linregress(hist_log, bin_edges[:-1])

        super_threshold = intercept
        if verbose:
            print ('r2: ' + str(r_value ** 2))
            print ('Fly global threshold: ' + str(super_threshold))

        if display:
            memotrack.display.histogram(super_filtered_positive, nbins=nbins, title=fly, ylog=True,
                                        rotate_labels=True, accent_after=0)

        # Apply threshold back to the dataframes
        if verbose:
            print ('Applying theshold...')
        for path in fly_paths:
            df_path = (str(path[:-4]) + '_detections_labels_fixed_signal_normalized_filtered.csv')
            detections_df = memotrack.io.read_df(df_path)
            nlabels = detections_df.label.nunique()
            detections_df['super_responsive'] = False

            # Get stimulus
            stim_kind = path[-5:-4]
            if stim_kind == 'A':
                stim_string = 'air'
            if stim_kind == 'O':
                stim_string = 'oct'
            if stim_kind == 'M':
                stim_string = 'mch'

            # Now pass again labeling the neurons
            for label in range(nlabels):
                filtered_signal = detections_df[detections_df['label'] == label].filtered.values

                filtered_signal_window = filtered_signal[window_start:]

                neuron_peak = np.max(filtered_signal_window)

                if neuron_peak >= super_threshold:
                    detections_df.loc[(detections_df.label == label), 'super_responsive'] = True
                    if stim_kind == 'A':
                        temp_air_neurons += 1
                    if stim_kind == 'O':
                        temp_oct_neurons += 1
                    if stim_kind == 'M':
                        temp_mch_neurons += 1

                    if verbose == 'full':
                        print ('- Neuron ' + str(label) + '\tis super responsive for ' + str(stim_string))

            # save detections_df back to disk with the super_responsive column
            new_path = df_path[:-4]
            new_path = new_path + '_responsive.csv'
            memotrack.io.write_df(detections_df, new_path)

        # Add to global counter
        total_air_neurons = total_air_neurons + temp_air_neurons
        total_oct_neurons = total_oct_neurons + temp_oct_neurons
        total_mch_neurons = total_mch_neurons + temp_mch_neurons
        temp_oct_ratio = float(temp_oct_neurons) / temp_air_neurons
        oct_ratio_list.append(temp_oct_ratio)
        temp_mch_ratio = float(temp_mch_neurons) / temp_air_neurons
        mch_ratio_list.append(temp_mch_ratio)

        if verbose:
            print ('- air: ' + str(temp_air_neurons))
            print ('- oct: ' + str(temp_oct_neurons) + ' | ratio: ' + str(temp_oct_ratio))
            print ('- mch: ' + str(temp_mch_neurons) + ' | ratio: ' + str(temp_mch_ratio))

    nflies = len(fly_list_unique)
    if verbose:
        print ('\nAll done !')
        print (str(nflies) + ' flies analysed')
        print ('')
        print (str(total_air_neurons) + ' neurons responsive to air, average of ' +
               str(float(total_air_neurons) / nflies))
        print (str(total_oct_neurons) + ' neurons responsive to oct, average of ' +
               str(float(total_oct_neurons) / nflies))
        print (str(total_mch_neurons) + ' neurons responsive to mch, average of ' +
               str(float(total_mch_neurons) / nflies))
        print ('')
        print ('Average oct ratio: ' + str(np.mean(oct_ratio_list)))
        print ('Average mch ratio: ' + str(np.mean(mch_ratio_list)))

    return


def smooth_track(detections_df, sigma=1, verbose=False, debug=False):
    import scipy.ndimage
    import numpy as np

    if debug:
        print ('\ndf before smoothing')
        print (detections_df)
    nlabels = detections_df['label'].nunique()

    if debug:
        print ('\nThe label filter:')
        print (detections_df['label'])

    if debug:
        nlabels = debug
    npoints = len(detections_df)

    if verbose:
        print ('Smoothing ' + str(nlabels) + ' labels on ' + str(npoints) + ' data points'),

    counter = 0
    for label in range(nlabels):

        # Dots counter
        if verbose:
            if counter > 50:
                print('[' + str(int((float(label) * 100 / nlabels))) + '%]'),
                counter = 0
        temp_df = detections_df[detections_df['label'] == label]

        x_values = temp_df.x.get_values().astype(float)
        y_values = temp_df.y.get_values().astype(float)
        z_values = temp_df.z.get_values().astype(float)

        # print('\nx_values:')
        # print (x_values)

        xs_values = scipy.ndimage.filters.gaussian_filter1d(x_values, sigma=sigma)
        ys_values = scipy.ndimage.filters.gaussian_filter1d(y_values, sigma=sigma)
        zs_values = scipy.ndimage.filters.gaussian_filter1d(z_values, sigma=sigma)

        index_list = temp_df.index

        # Writing back to detections_df
        position = 0
        for index in index_list:
            detections_df.set_value(index, 'xsmooth', xs_values[position])
            detections_df.set_value(index, 'ysmooth', ys_values[position])
            detections_df.set_value(index, 'zsmooth', zs_values[position])
            position += 1

        # Counter for dots
        counter += 1

    return detections_df


def cluster_weights(summary_df, display=False, verbose=False, sigma_value=False, title=False):
    import matplotlib.pyplot as plt
    import numpy as np

    w1 = summary_df['w1'].as_matrix()
    w2 = summary_df['w2'].as_matrix()

    if verbose:
        print ('\nw1')
        print (w1)
        print ('w2')
        print (w2)

    if display:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        plt.scatter(w1, w2, color='#3366FF', lw=0, alpha=0.5)

        # Plot fixes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
        plt.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')
        plt.xlabel('w1 (distance)')
        plt.ylabel('w2 (intensity)')
        if sigma_value:
            plt.title('sigma ' + ('%.2f' % sigma_value) + ' | ' + str(len(w1)) + ' detections')
        if title:
            plt.title(title + ' | ' + str(len(w1)) + ' detections')


def GMM_thresh(data, verbose=False, debug=False):
    from sklearn import mixture
    import numpy as np
    import sys

    # Foreground extraction using GMM

    img_flat = np.ravel(data)
    img_flat = img_flat[img_flat > 0]

    if debug:
        print ('Data info:')
        print (' len: ' + str(len(img_flat)))
        print (' min: ' + str(np.min(img_flat)))
        print (' avg: ' + str(np.mean(img_flat)))
        print (' max: ' + str(np.max(img_flat)))

    img_flat = np.expand_dims(img_flat, 1)

    means_init = [0, max(img_flat)]
    means_init = np.expand_dims(means_init, 1)
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='tied', means_init=means_init).fit(img_flat)

    if debug:
        print ('\nGMM results')
        print ('weights:'),
        print (gmm.weights_[0]),
        print (gmm.weights_[1])
        print ('means:'),
        print (str(gmm.means_[0])),
        print (str(gmm.means_[1]))
        print ('covariances:'),
        print (gmm.covariances_[0])
        print ('converged:'),
        print (gmm.converged_)

    if verbose:
        print ('\nSetting threshold...')
        sys.stdout.flush()

    min_value = min(img_flat)
    max_value = max(img_flat)

    if verbose:
        print ('min value: ' + str(min_value) + '\tmax value: ' + str(max_value))

    predict_class = 0
    precision = 1
    check_value = min_value
    check_value = np.expand_dims(check_value, 1)

    while predict_class == 0:
        predict_class = gmm.predict(check_value)
        check_value += precision

    # Getting threshold back to normal value
    thresh = check_value

    # thresh = gmm.means_[1]  # this ignores the intersection, and uses the mean from the second gaussian.

    if debug:
        print ('normalized thresh: ' + str(check_value))
        print ('final thresh: ' + str(thresh))

    return thresh


def GMM_thresh_old(data, verbose=False, debug=False):
    from sklearn import mixture
    import numpy as np
    import sys

    # Foreground extraction using GMM
    if verbose:
        print ('Starting GMM...')
        sys.stdout.flush()

    img_flat = data[data > 0]
    normalization_max = max(img_flat)
    img_flat = img_flat / normalization_max
    img_flat = np.sort(img_flat)

    if debug:
        print ('\nAfter bigger than zero and sorted:')
        print (np.shape(img_flat))
        print (img_flat)
    img_flat = np.expand_dims(img_flat, 1)
    means_init = [0, max(img_flat)]
    means_init = np.expand_dims(means_init, 1)
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='tied', means_init=means_init).fit(img_flat)

    if debug:
        print ('\nGMM results')
        print ('\nweights:')
        print (gmm.weights_)
        print ('\nmeans:')
        print (gmm.means_)
        print ('\ncovariances:')
        print (gmm.covariances_)
        print ('\nconverged:')
        print (gmm.converged_)

    if verbose:
        print ('\nSetting threshold...')
        sys.stdout.flush()
    min_value = min(img_flat)
    max_value = max(img_flat)
    if verbose:
        print ('min value: ' + str(min_value) + '\tmax value: ' + str(max_value))
    predict_class = 0
    precision = 0.001
    check_value = min_value
    check_value = np.expand_dims(check_value, 1)
    while predict_class == 0:
        predict_class = gmm.predict(check_value)
        check_value += precision
    if debug:
        print ('found threshold of ' + str(check_value))

    # Getting threshold back to normal value
    thresh = check_value * normalization_max

    return thresh


def generate_voronoi(detections_df, meta, max_distance=5.0, verbose=True, debug=False):
    """
    Tesselation of the arrain into the coordinates given on detections_df
    :param detections_df:
    :param meta:
    :param verbose:
    :param debug:
    :return:
    """

    import numpy as np
    import sys
    from scipy import ndimage
    from skimage.morphology import watershed
    import memotrack
    import tifffile as tiff

    if debug:
        print ('Starting voronoi...')
        print ('Total of ' + str(len(detections_df)) + ' points for tessellation')
        print ('region max size: ' + str(max_distance))
        sys.stdout.flush()

    # Number of slices we need for an isometric image
    nslices = int((float(meta['SizeZ']) * float(meta['PhysicalSizeZ'])) / float(meta['PhysicalSizeX']))

    # Generate blank image
    dist_coords_img = np.ones([nslices, meta['SizeY'], meta['SizeX']], dtype='uint16')
    labels_coords_img = np.zeros([nslices, meta['SizeY'], meta['SizeX']], dtype='uint16')

    # reference coordinates for distance calculation
    # ATTENTION: Here we choose "xsmooth" before the backprojection, or "xreprojected" after the backprojection
    ref_coords = np.transpose([detections_df['zsmooth'], detections_df['ysmooth'], detections_df['xsmooth'],
                               detections_df['label']])

    sys.stdout.flush()
    if debug:
        print ('Generating base images...'),
        sys.stdout.flush()
    # generate base image from coords
    current_label = 1
    for coords in ref_coords:
        dist_coords_img[int(coords[0]), int(coords[1]), int(coords[2])] = 0
        labels_coords_img[int(coords[0]), int(coords[1]), int(coords[2])] = int(coords[3])
        current_label += 1

    if debug:
        print ('[Done]')
        print ('Starting distance transform...'),
        sys.stdout.flush()
    distance_transform = ndimage.distance_transform_edt(dist_coords_img)

    if debug:
        print ('[Done]')
        print ('Starting objects mask...'),
        sys.stdout.flush()

    objects_mask = distance_transform < max_distance

    if debug:
        print('[Done]')
        print ('Initializing watershed...'),
        sys.stdout.flush()

    watershed_img = watershed(distance_transform, labels_coords_img, mask=objects_mask)

    if debug:
        print('[Done]')
        sys.stdout.flush()
        memotrack.analyse.info_np(watershed_img)

    return watershed_img


def channel_correlation(c1, c2, meta, fix_lim=False, save_path=False, display=True, verbose=True):
    import numpy as np
    import memotrack.display
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib

    if display is False:
        plt.ioff()

    img = [c1, c2]

    if verbose:
        print('\nStarting Z intensity correction')

    img_avg = [[], []]
    for c in range(2):
        if verbose:
            print ('\nChannel ' + str(c))

        for z in range(meta['SizeZ']):
            avg = np.mean(img[c][z])
            img_avg[c].append(avg)

        if verbose:
            print (img_avg[c])

    memotrack.display.plot1Dline(img_avg, data_label=['c1', 'c2'], color=['#22AAFF', '#FFAA22'])

    cmap = matplotlib.cm.get_cmap('rainbow')
    color_list = []
    # Generate color list
    for z in range(meta['SizeZ']):
        rgba = cmap(float(z) / meta['SizeZ'])
        rgb = (rgba[0], rgba[1], rgba[2])
        color_list.append(rgb)

    # Plot
    fig, ax = plt.subplots()
    c1_ravel = np.ravel(c1[z])
    c2_ravel = np.ravel(c2[z])
    ax.scatter(c1_ravel, c2_ravel, lw=0, color=color_list, alpha=1, s=2)

    if fix_lim:
        xmax = fix_lim[0]
        ymax = fix_lim[1]
    else:
        xmax = np.max(c1_ravel)
        ymax = np.max(c2_ravel)

    # Plot fixes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='on')
    plt.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')
    plt.xlim([0, xmax])
    plt.ylim([0, ymax])

    if save_path:
        plt.savefig(save_path, dpi=72)

    # 2d histogram
    fig2, ax2 = plt.subplots()
    plt.hist2d(c1_ravel, c2_ravel, bins=100, norm=LogNorm(), cmap='viridis')
    # Plot fixes
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='on')
    plt.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')
    plt.xlim([0, xmax])
    plt.ylim([0, ymax])


def check_quality(detections_df, threshold=0.8, verbose=True, visualization=False):
    import numpy as np
    import math

    if verbose:
        print ('\nStarting detection quality check...')
    nframes = detections_df['t'].nunique()
    nlabels_list = []
    avg_z_list = []
    dist_list = []

    detections_df = detections_df[detections_df['label'] >= 0].copy(deep=True)

    for frame in range(nframes):
        temp_df = detections_df[detections_df['t'] == frame]
        nlabels_list.append(temp_df['label'].nunique())
        avg_z_list.append(temp_df['z'].mean())

    # Calculate gradient
    nlabels_gradient = abs(np.gradient(nlabels_list))
    nlabels_list = nlabels_gradient

    avg_z_list.append(avg_z_list[-1])
    labels_std = np.std(nlabels_list)
    labels_med = np.median(nlabels_list)
    zdiff = abs(np.gradient(avg_z_list))
    zdiff_med = np.median(zdiff)
    zdiff_std = np.std(zdiff)

    print ('labels_std: ' + str(labels_std))

    # Detections check
    if verbose:
        print ('\nChecking detections...')
        print (' mid: ' + str(labels_med))
        print (' std: ' + str(labels_std))

        for frame in range(nframes):
            if nlabels_list[frame] < (labels_med - (3 * labels_std)):
                print ('Frame ' + str(frame) + ' with only ' + str(nlabels_list[frame]) + str(' detections'))

        # Checking center of mass
        print ('\nChecking center of mass...')
        print (' mid: ' + str(zdiff_med))
        print (' std: ' + str(zdiff_std))
        for frame in range(len(zdiff)):

            if zdiff[frame] > (zdiff_med + (3 * zdiff_std)):
                print ('Frame ' + str(frame) + ' with zdiff ' + str(zdiff[frame]))

    # Z Score calculations
    z_detections = abs((nlabels_list - labels_med) / labels_std) ** 2
    z_center = abs((zdiff - zdiff_med) / zdiff_std) ** 2
    z_center = z_center[:nframes]
    # Quality level
    quality = 100 / (np.mean([z_detections, z_center], axis=0) + 100)

    # Setting quality to time frames
    detections_df['Q'] = 1.0
    for frame in range(nframes):
        detections_df.loc[detections_df['t'] == frame, 'Q'] = quality[frame]

    if verbose:
        print ('\nDetection quality for individual frames:')
        print (quality)
        print ('\n[Done]')

    if visualization:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 10))
        plt.plot(nlabels_gradient)
        plt.title('Labels gradient')

        plt.figure(figsize=(20, 10))
        plt.plot(zdiff)
        plt.title('zdiff')

        plt.figure(figsize=(20, 10))
        plt.plot(z_center, label='center')
        plt.plot(z_detections, label='detections')
        plt.title('center & detections')
        plt.ylim((0, 200))
        plt.legend()
        plt.show()

        plt.figure(figsize=(20, 10))
        plt.plot(quality)
        plt.axhline(y=0.95)
        plt.ylim((0, 1.0))

        plt.title('detection quality')

    return detections_df


def barycentric_coords(vertices, point):
    import numpy as np
    import numpy.linalg as la

    T = (np.array(vertices[:-1]) - vertices[-1]).T
    v = np.dot(la.inv(T), np.array(point) - vertices[-1])
    v.resize(len(vertices))
    v[-1] = 1 - v.sum()

    return v


def tetrahedron_volume(vertices=None, sides=None):
    """
    Return the volume of the tetrahedron with given vertices or sides. If
    vertices are given they must be in a NumPy array with shape (4,3): the
    position vectors of the 4 vertices in 3 dimensions; if the six sides are
    given, they must be an array of length 6. If both are given, the sides
    will be used in the calculation.

    Raises a ValueError if the vertices do not form a tetrahedron (for example,
    because they are coplanar, colinear or coincident). This method implements
    Tartaglia's formula using the Cayley-Menger determinant:
              |0   1    1    1    1  |
              |1   0   s1^2 s2^2 s3^2|
    288 V^2 = |1  s1^2  0   s4^2 s5^2|
              |1  s2^2 s4^2  0   s6^2|
              |1  s3^2 s5^2 s6^2  0  |
    where s1, s2, ..., s6 are the tetrahedron side lengths.

    Warning: this algorithm has not been tested for numerical stability.

    """
    import numpy as np

    # The indexes of rows in the vertices array corresponding to all
    # possible pairs of vertices
    vertex_pair_indexes = np.array(((0, 1), (0, 2), (0, 3),
                                    (1, 2), (1, 3), (2, 3)))
    if sides is None:
        # If no sides were provided, work them out from the vertices
        if type(vertices) != np.ndarray or vertices.shape != (4, 3):
            raise TypeError('Invalid vertex array in tetrahedron_volume():'
                            ' vertices must be a numpy array with shape (4,3)')
        # Get all the squares of all side lengths from the differences between
        # the 6 different pairs of vertex positions
        vertex1, vertex2 = vertex_pair_indexes[:, 0], vertex_pair_indexes[:, 1]
        sides_squared = np.sum((vertices[vertex1] - vertices[vertex2]) ** 2,
                               axis=-1)
    else:
        # Check that sides has been provided as a valid array and square it
        if type(sides) != np.ndarray or sides.shape != (6,):
            raise TypeError('Invalid argument to tetrahedron_volume():'
                            ' sides must be a numpy array with shape (6,)')
        sides_squared = sides ** 2

    # Set up the Cayley-Menger determinant
    M = np.zeros((5, 5))
    # Fill in the upper triangle of the matrix
    M[0, 1:] = 1
    # The squared-side length elements can be indexed using the vertex
    # pair indices (compare with the determinant illustrated above)
    M[tuple(zip(*(vertex_pair_indexes + 1)))] = sides_squared

    # The matrix is symmetric, so we can fill in the lower triangle by
    # adding the transpose
    M = M + M.T

    # Calculate the determinant and check it is positive (negative or zero
    # values indicate the vertices to not form a tetrahedron).
    det = np.linalg.det(M)
    if det <= 0:
        raise ValueError('Provided vertices do not form a tetrahedron')
    return np.sqrt(det / 288)


def crosscheck(df, RT=0.1, verbose=True, diameter=1.21, pixel_size=0.161, debug=False):
    import numpy as np
    import scipy.spatial.distance as distance

    if verbose:
        print ('\nStarting crosstalk check for RT={}, diameter={} and pixel size={} '.format(RT, diameter, pixel_size))

    diameterpx = diameter/pixel_size

    nlabels = df.label.nunique()
    nframes = df.t.nunique()
    # Set responsive neuron based on RT
    df['responsive'] = False

    responsive_labels = []
    if verbose:
        print ('\nResponsive neurons: '),
    for label in range(nlabels):
        normalized_signal = df[df['label'] == label].norm_intensity.values

        neuron_peak = np.max(normalized_signal)

        if neuron_peak > RT:
            df.loc[(df.label == label), 'responsive'] = True
            responsive_labels.append(label)
            if verbose:
                print ('{}'.format(label)),

    if verbose:
        print ('\nTotal of {} responsive neurons (without crosstaltk check)'.format(len(responsive_labels)))
        print ('\n\nChecking crosstalk for diameter of {} pixels'.format(diameterpx))

    max_df = df.groupby(by='label').max().reset_index()
    peak_list = max_df['norm_intensity'].values

    responsive_labels_check = []
    for label in responsive_labels:
        print ('\n\033[1mLabel: {}\033[0m'.format(label))
        label_peak = max_df[max_df['label'] == label]['norm_intensity'].values[0]
        if verbose:
            print ('Peak: {}'.format(label_peak))

        # Make distance matrix
        xs = max_df.x.values
        ys = max_df.y.values
        zs = max_df.z.values

        coords = np.transpose([xs, ys, zs])

        dist_matrix = distance.cdist(coords, coords, 'euclidean')

        label_distances = dist_matrix[:, label]

        neighbours_labels = np.where(label_distances < 2*diameterpx)[0].tolist()
        del neighbours_labels[neighbours_labels == label]

        print ('Neighbours labels: {}'.format(neighbours_labels))
        print ('Number of neighbours: {}'.format(len(neighbours_labels)))
        neighbours_peaks = peak_list[neighbours_labels]
        print ('Neighbours peaks: {}'.format(neighbours_peaks))

        if len(neighbours_peaks>0):
            max_peak = np.max(neighbours_peaks)
        else:
            max_peak = 0

        if label_peak < max_peak:
            print ('\033[91mLabel has a stronger neighbour !\033[0m ')
        else:
            print ('\033[92mStrongest of the neighbourhood.\033[0m')
            responsive_labels_check.append(label)

    if verbose:
        print ('\n\n\nTotal of \033[1m{} neurons\033[0m passed the crosstalk check:'.format(len(responsive_labels_check)))
        print (responsive_labels_check)

    # Marking crosstalkcheck labels on df
    df['responsive_crosscheck'] = False
    for label in responsive_labels_check:
        df.loc[df['label'] == label, 'responsive_crosscheck'] = True

    if verbose:
        print ('\n\n\nFinal check:')
        print ('Responsive neurons: {}'.format(df['responsive'].sum()/nframes))
        print ('Crosstalk checked:  {}'.format(df['responsive_crosscheck'].sum()/nframes))

    return df
