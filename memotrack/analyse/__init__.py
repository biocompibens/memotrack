# Analyse detections
def detections_error(detections_df, verbose=True):
    import numpy as np

    # Get info
    nlabels = detections_df['label'].max()
    nframes = detections_df['t'].max()
    npoints = len(detections_df.index)
    max_points = nlabels * nframes

    # Count noise
    nnoise = len(detections_df[detections_df.label == -1].index)

    # Remove points labeled as noise
    detections_df = detections_df[detections_df.label != -1]
    ndetections = len(detections_df.index)

    # Count double detections
    double_detections = detections_df.duplicated(subset=['t', 'label']).sum()

    # Get number of missing values on the detections
    missing_values = max_points - (ndetections - double_detections)

    # Errors
    # double_ratio = (float(double_detections) / float(ndetections))
    # missing_ratio = (float(missing_values) / float(max_points))
    # noise_ratio = (float(nnoise) / float(max_points))
    double_ratio = (float(double_detections) / npoints)
    missing_ratio = (float(missing_values) / npoints)
    noise_ratio = (float(nnoise) / npoints)

    # Error list
    errors = [double_ratio, missing_ratio, noise_ratio]

    if verbose:
        print ('Double detections: ' + str(double_detections) + ' (' + '%.2f' % (double_ratio * 100) + '%)')
        print ('Missing points: ' + str(missing_values) + ' (' + '%.2f' % (missing_ratio * 100) + '%)')
        print ('Noise count : ' + str(nnoise) + ' (' + '%.2f' % (noise_ratio * 100) + '%)')
        print ('Error: ' + str(np.sum(errors)))
        print ('')

    return errors


def distances_and_frames(detections_df, all_frames=False, start_frame=5, end_frame=6, dist_close=2, dist_far=80,
                         verbose=True):
    """

    :param detections_df:
    :param verbose:
    :return:
    """
    import scipy.stats
    import numpy as np
    import scipy.spatial.distance
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    if verbose:
        print ('Frame distance definitions:')
        print ('   Close is less than\t' + str(dist_close))
        print ('   Far is more than\t' + str(dist_far))

    # Get number of frames on dataframe
    nframes = int(detections_df['t'].max())

    # Counters
    close_count = 0
    far_count = 0

    if all_frames:
        start_frame = 0
        end_frame = nframes

    # Distributions list
    close_distribution = []
    far_distribution = []

    if verbose:
        print ('Starting comparisons'),
    for ref in range(start_frame, end_frame):
        if verbose:
            print ('[' + str(ref) + ']'),

        # Get reference dataframe
        df_frame1 = detections_df[detections_df['t'] == ref].copy(deep=True)
        xyz_frame1 = df_frame1[['x', 'y', 'z']].values

        for frame in range(0, nframes):

            # In case its close
            if abs(frame - ref) <= dist_close and (frame - ref) != 0:
                # Open dataframe to compare
                df_frame2 = detections_df[detections_df['t'] == frame].copy(deep=True)

                # Convert to numpy
                xyz_frame2 = df_frame2[['x', 'y', 'z']].values

                # Distances
                dist = scipy.spatial.distance.cdist(xyz_frame1, xyz_frame2)
                # Sorting matrix to get only the closest points
                dist = np.sort(dist, axis=0)
                dist = dist[0]
                # Take the average
                dist = np.average(dist)
                # Append value to list
                close_distribution.append(dist)
                close_count += 1

            # In case its far
            if abs(frame - ref) >= dist_far:
                # Open dataframe to compare
                df_frame2 = detections_df[detections_df['t'] == frame].copy(deep=True)

                # Convert to numpy
                xyz_frame2 = df_frame2[['x', 'y', 'z']].values

                # Distances
                dist = scipy.spatial.distance.cdist(xyz_frame1, xyz_frame2)
                # Sorting matrix to get only the closest points
                dist = np.sort(dist, axis=0)
                dist = dist[0]
                # Take the average
                dist = np.average(dist)
                # Append value to list
                far_distribution.append(dist)
                far_count += 1

    # KS statistcs for the two distributions
    KSstats, pvalue = scipy.stats.ks_2samp(close_distribution, far_distribution)

    # Averages
    close_mean = np.average(close_distribution)
    far_mean = np.average(far_distribution)

    # Prints and plots
    if verbose:
        print ('\nKS stats: ' + str(KSstats))
        print ('pvalue: ' + str(pvalue))
        print ('\nClose cases: ' + str(close_count))
        print ('Close mean: ' + str(close_mean))
        print ('\nFar cases: ' + str(far_count))
        print ('Far mean: ' + str(far_mean))
        print ('\nMeans difference: ' + str(abs(far_mean - close_mean)))

        # Define custom colors
        gray = '#555555'
        pink = '#FF72CE'
        blue = '#00AEEE'

        # Plots #
        img_size = 12
        plt.rcParams['figure.figsize'] = (img_size, img_size)
        fig, (ax1, ax2) = plt.subplots(2)

        # Box plot #
        bp = ax1.boxplot((close_distribution, far_distribution),
                         labels=['Time difference < ' + str(dist_close), 'Time difference > ' + str(dist_far)],
                         patch_artist=True)

        # Style fixes
        bp['boxes'][0].set(color=gray, facecolor=pink, linewidth=2)
        bp['medians'][0].set(color=gray, linewidth=2)

        bp['boxes'][1].set(color=gray, facecolor=blue, linewidth=2)
        bp['medians'][1].set(color=gray, linewidth=2)

        for whisker in bp['whiskers']:
            whisker.set(color=gray, linewidth=2)

        for cap in bp['caps']:
            cap.set(color=gray, linewidth=2)

        for flier in bp['fliers']:
            flier.set(marker='.', color=gray, alpha=0.5)

        # Plot fixes
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='on')
        ax1.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')
        ax1.set_title('KS statistic: ' + str(KSstats) + '        p value: ' + str(pvalue))

        # Histogram #
        last_bin = 11
        bins = np.linspace(0, last_bin, 50)
        htype = 'step'

        show_fit = False

        # Close distribution
        close_distribution = np.sort(close_distribution)
        ax2.hist(close_distribution, normed=True, color=pink, bins=bins, alpha=0.8, histtype=htype, linewidth=2)

        if show_fit:
            fit_close = stats.norm.pdf(close_distribution, np.mean(close_distribution), np.std(close_distribution))
            ax2.plot(close_distribution, fit_close, '-', color=pink, linewidth=5, alpha=0.5)

        # Far distribution
        far_distribution = np.sort(far_distribution)
        ax2.hist(far_distribution, normed=True, color=blue, bins=bins, alpha=0.8, histtype=htype, linewidth=2)

        if show_fit:
            fit_far = stats.norm.pdf(far_distribution, np.mean(far_distribution), np.std(far_distribution))
            ax2.plot(far_distribution, fit_far, '-', color=blue, linewidth=5, alpha=0.5)

        # Plot fixes
        ax2.legend(['Closer than ' + str(dist_close) + ' frames',
                    'Farther than ' + str(dist_far) + ' frames'],
                   frameon=False, fontsize=11)
        ax2.set_ylim([-0.01, 1.1])
        ax2.set_xlim([0, last_bin])
        ax2.set_xlabel('Mean distance between two frames')
        ax2.set_ylabel('Normalized distribution')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='on')
        ax2.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')

    return KSstats, pvalue, close_mean, far_mean


def detection_precision(ground_df, detections_df, FWHMxy=1.337, pixel_size=0.16125, time_frame=0,
                        verbose=True, display=False, debug=False):
    """
    Compares the detection with the synthetic data real position
    :param ground_df:
    :param detections_df:
    :param FWHMxy:
    :param pixel_size:
    :param verbose:
    :return:
    """
    import numpy as np
    import scipy.spatial.distance as dist
    from matplotlib import pyplot as plt
    import pandas as pd

    check_distance = (FWHMxy / 2) / pixel_size

    if debug:
        print ('*** Check distance in voxels: ' + str(check_distance))
        print ('*** Check distance in microns: ' + str(check_distance * pixel_size))
        print ('')

    # Gets only the desired time frame from the ground truth
    ground_df = ground_df[ground_df['t'] == time_frame]

    nground = len(ground_df[ground_df['t'] == time_frame])
    ndetections = len(detections_df[detections_df['t'] == time_frame])

    if verbose:
        print ('Coordinates on ground truth: ' + str(nground))
        print ('Number of detections : ' + str(ndetections))

    # Gets matrix of coordinates
    x_ground = ground_df['x'].values
    y_ground = ground_df['y'].values
    z_ground = ground_df['z'].values
    ground_coords = [x_ground, y_ground, z_ground]
    ground_coords = np.transpose(ground_coords)

    x_detections = detections_df['x'].values
    y_detections = detections_df['y'].values
    z_detections = detections_df['z'].values
    detections_coords = [x_detections, y_detections, z_detections]
    detections_coords = np.transpose(detections_coords)

    if debug:
        print ('\n*** Ground truth matrix:'),
        print np.shape(ground_coords),
        print (ground_coords.dtype)
        print ('*** Detections matrix:'),
        print np.shape(detections_coords),
        print (ground_coords.dtype)

    dist_matrix = dist.cdist(detections_coords, ground_coords)
    sorted_matrix_truth = np.sort(dist_matrix, axis=0)
    # print (sorted_matrix_truth)
    sorted_matrix_detection = np.sort(dist_matrix, axis=1)

    # Distance Matrix
    if display:
        plt.figure(figsize=(15, 10))
        plt.imshow(sorted_matrix_truth, cmap='RdYlBu_r')
        plt.title('Sorted distances based on truth')
        plt.colorbar()

        plt.figure(figsize=(15, 10))
        plt.imshow(sorted_matrix_detection, cmap='RdYlBu_r')
        plt.title('Sorted distances based on detections')
        plt.colorbar()

    # Get a vector with the closest points
    closest_truth = np.sort(sorted_matrix_truth[0, :])
    closest_detections = np.sort(sorted_matrix_detection[:, 0])

    # Who's smaller, detections or ground truth ?
    max_possible_hits = min(len(closest_detections), len(closest_truth))

    if display:
        plt.figure(figsize=(15, 5))
        plt.plot(closest_detections, color='#30AAFF', linewidth=2, label='detections')
        plt.plot(closest_truth, color='#30FFAA', linewidth=2, label='ground truth')
        plt.legend(frameon=False, loc='upper left')
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('left')

    # Distribution of closest
    if display:
        nbins = 50
        plt.figure(figsize=(15, 8))
        # plt.hist(minimum_distances_micron, 100, histtype='stepfilled', color='blue', label='blue')
        plt.hist(closest_truth, nbins, color='#00AAFF', alpha=0.5)
        plt.axvline(x=check_distance, linewidth=3, color='#FF3322')
        plt.title('Distribution of minimum distances (' + str(nbins) + ' bins)')
        plt.xlabel('Distance in voxels')

    hits_truth = (closest_truth < check_distance).sum()
    hits_detections = (closest_detections < check_distance).sum()
    hits = min(hits_truth, hits_detections)

    # Sanity check
    if hits > max_possible_hits:
        print ('WARNING: More hits than max possible hits. Something is wrong !')

    if debug:
        print ('*** Truth hits: ' + str(hits_truth))
        print ('*** Detection hits: ' + str(hits_detections))

    total_matches = (dist_matrix < check_distance).sum()
    nduplicates = total_matches - hits

    if verbose:
        print ('\nNumber of hits: ' + str(hits) + ' (' + str((float(hits) / ndetections) * 100) + '%)')
        print ('Found ' + str(nduplicates) + ' duplicates,'),
        print ('total of ' + str(total_matches) + ' points closer than ' + str(FWHMxy / 2) + ' microns')

    # Calculation of Jaccard index
    # Defined as intersection over the union
    jaccard = float(hits) / ((ndetections - hits) + nground)

    # Average distance of detections
    avg_dist = np.average(closest_truth)

    if verbose:
        print ('\nJaccard index: ' + str(jaccard))
        print ('Average distance: ' + str(avg_dist) + ' voxels')
    if verbose:
        print ('\nAll done !')

    return ndetections, nground, hits, jaccard, avg_dist


def info_np(img):
    """
    Basic info about the image
    :param img: Numpy array to analyse
    """
    import numpy as np

    print ('Dimensions: ' + str(np.shape(img)))
    print ('Min value: ' + str(np.min(img)))
    print ('Avg value: ' + str(np.average(img)))
    print ('Med value: ' + str(np.median(img)))
    print ('Max value: ' + str(np.max(img)))
    print ('Std dev: ' + str(np.std(img)))
    print ('Sum: ' + str(np.sum(img)))


def info_df(detections_df):
    import pandas as pd
    import numpy as np

    frames = detections_df['t'].unique()
    nframes = detections_df['t'].nunique()
    detections_list = []
    for frame in frames:
        detections_list.append(len(detections_df[detections_df['t'] == frame]))

    # Get values from list
    min_detections = np.min(detections_list)
    max_detections = np.max(detections_list)
    avg_detections = np.average(detections_list)
    std_detections = np.std(detections_list)

    df_analysis = {'min': min_detections,
                   'max': max_detections,
                   'avg': avg_detections,
                   'std': std_detections,
                   'nframes': nframes}
    return df_analysis


def check_results(files_path, save_plots=False, limit_date=False, save_results_df=False, reg_ratio=False,
                  error_log=False):
    import os
    import numpy as np
    import memotrack
    import time
    import datetime
    import matplotlib.pyplot as plt
    import pandas as pd
    import sys
    import csv

    with open('/projects/memotrack/data/automated_ignore_list.csv', 'rb') as f:
        reader = csv.reader(f)
        ignore_list = list(reader)
    ignore_list = ignore_list[0]
    ignore_list.append('/projects/memotrack/data/00316L/SAOAOS/00316L_SAOAOS.tif') # incomplete file


    # Start dataframe
    results_df = pd.DataFrame()

    if limit_date is False:
        limit_date = '02/09/1986'

    # Fix alternative date type
    if limit_date[2] == '_':
        limit_date = limit_date.replace('_', '/')

    # Get fist of paths
    text_file = open(files_path, 'r')
    path_list = text_file.read().split('\n')
    text_file.close()

    print ('Analysing files:')
    sys.stdout.flush()
    nlabels_list = []
    nresponsive_list = []
    missing_files = 0
    for path in path_list:

        df_path = (str(path[:-4]) + '_normalized.csv')
        print ('\n' + str(path))
        sys.stdout.flush()
        if os.path.isfile(df_path):
            # start results dictionary
            data_for_df = {}

            modify_time = os.path.getmtime(df_path)
            file_day = time.strftime('%d/%m/%Y', time.gmtime(modify_time))
            limit_date_ = datetime.datetime.strptime(limit_date, '%d/%m/%Y')
            modification_date = datetime.datetime.strptime(file_day, '%d/%m/%Y')
            data_for_df['date'] = file_day

            # Check if its newer than limit date
            if modification_date >= limit_date_:
                # Get file name
                name = path[-17:-4]
                data_for_df['ID'] = name[:-7]
                # Time of last file modification
                print (file_day)
                sys.stdout.flush()

                '''
                # Memotrack version
                out_file = open(path + '.out', 'r')
                version_full_name = out_file.readline().rstrip()
                out_file.close()
                print (version_full_name)
                sys.stdout.flush()
                version = version_full_name[11:]
                data_for_df['version'] = version
                '''

                # Load detections dataframe with normalized data
                detections_df = memotrack.io.read_df(df_path)

                # Get number of unique labels
                nlabels = int(detections_df['label'].nunique())
                nlabels_list.append(nlabels)
                data_for_df['detections'] = nlabels
                print (str(nlabels) + ' detections')
                sys.stdout.flush()

                '''
                # Get number of responsive neurons
                if 'responsive' in detections_df:
                    nresponsive = detections_df[detections_df['responsive'] == True].label.nunique()
                    nresponsive_list.append(nresponsive)
                    data_for_df['responsive'] = nresponsive
                    print (str(nresponsive) + ' responsive neurons')
                    sys.stdout.flush()
                '''
                if reg_ratio:
                    # Get registration ratio
                    reg_ratio = memotrack.analyse.cluster_density(detections_df, verbose=False)
                    data_for_df['reg_ratio'] = reg_ratio
                    print ('Registration ratio: ' + str(reg_ratio))
                    sys.stdout.flush()

                if name[:2] == '00':
                    data_for_df['group'] = 'naive'
                    data_for_df['color'] = '#C3C4BE'

                if name[:2] == '10':
                    data_for_df['group'] = 'paired1'
                    data_for_df['color'] = '#4F6F84'

                if name[:2] == '01':
                    data_for_df['group'] = 'unpaired1'
                    data_for_df['color'] = '#769DA0'

                if name[:2] == '20':
                    data_for_df['group'] = 'paired2'
                    data_for_df['color'] = '#9B5567'

                if name[:2] == '02':
                    data_for_df['group'] = 'unpaired2'
                    data_for_df['color'] = '#BF707B'

                if path in ignore_list:
                    data_for_df['ignore'] = True
                    print ('Fly in ignore list !')
                else:
                    data_for_df['ignore'] = False


                # Get fly
                data_for_df['fly'] = name[:5]
                # Get brain hemisphere
                data_for_df['lobe'] = name[5]
                # Get stimulus
                data_for_df['stimulus'] = name[7:]

                if error_log:

                    # Read error log
                    error_log = open(path + '.err', 'r').readlines()
                    if len(error_log) > 0:
                        print ('Error log: ')
                        for line in error_log:
                            print (line),
                        data_for_df['error'] = True
                    else:
                        data_for_df['error'] = False

                # Concatenate new data
                temp_df = pd.DataFrame(data_for_df, index=[0])
                results_df = pd.concat([results_df, temp_df], ignore_index=True)

                # Signal plot
                if save_plots:
                    # Generate colormap
                    cmap = memotrack.display.rand_cmap(nlabels, type='super_bright', first_color_black=False,
                                                       verbose=False)

                    # Detection clusters
                    memotrack.display.plot_from_df(detections_df, new_cmap=cmap, size=2, elev=15, azim=30,
                                                   crop_data=False, auto_fit=True, one_frame=False,
                                                   frame=0, time_color=False, intensity=False, borders=False,
                                                   registered_coords=False, title=name,
                                                   save=path + '_clusters.png')

                    # Super responsive neurons
                    memotrack.display.plot_1D_signals(detections_df, normalize='filtered', smooth=0,
                                                      accent=False, only_responsive='false', only_positive=True,
                                                      stim_frame=[25, 125, 225], stim_duration=25,
                                                      cmap=cmap, title=name,
                                                      save=path + '_responsive.png', HighDPI=False)

            else:
                print('File is older than desired limit (' + limit_date + ')')
                sys.stdout.flush()
                missing_files += 1

        elif len(path) > 0:
            print ('Analysis not found for this file')
            sys.stdout.flush()
            missing_files += 1

    print ('Total of ' + str(len(nlabels_list)) + ' files analysed, missing ' + str(missing_files) + ' results.')
    print ('Average of ' + str(np.average(nlabels_list)) + ' detections, std of ' + str(np.std(nlabels_list)))
    print (
        'Average of ' + str(np.average(nresponsive_list)) + ' responsive neurons, std of ' + str(
            np.std(nresponsive_list)))
    sys.stdout.flush()

    if save_results_df:
        print ('Saving results to ' + str(save_results_df))
        sys.stdout.flush()
        results_df.to_csv(save_results_df)

    return results_df


def cluster_density(detections_df, verbose=False, debug=False):
    import scipy.spatial.distance as dist
    import numpy as np

    nframes = detections_df['t'].nunique()
    nlabels = detections_df['label'].nunique()

    if debug:
        nlabels = int(debug)

    if verbose:
        print ('Checking distance of ' + str(nlabels) + ' labels over ' + str(nframes) + ' frames')

    total_avg_raw_dist = 0
    total_avg_reg_dist = 0
    for label in range(nlabels):
        if debug:
            print ('Label [' + str(label) + ']')
        temp_df = detections_df[detections_df['label'] == label]

        # Get mean values
        mean_x = temp_df['x'].mean()
        mean_xreg = temp_df['xreg'].mean()
        mean_y = temp_df['y'].mean()
        mean_yreg = temp_df['yreg'].mean()
        mean_z = temp_df['z'].mean()
        mean_zreg = temp_df['zreg'].mean()

        x_values = temp_df['x'].values
        xreg_values = temp_df['xreg'].values
        y_values = temp_df['y'].values
        yreg_values = temp_df['yreg'].values
        z_values = temp_df['z'].values
        zreg_values = temp_df['zreg'].values

        reg_list = np.array([xreg_values, yreg_values, zreg_values])
        raw_list = np.array([x_values, y_values, z_values])

        total_reg_dist = 0
        for reg_coord in reg_list.T:
            distance = dist.euclidean((mean_xreg, mean_yreg, mean_zreg), reg_coord)
            total_reg_dist += distance

        total_raw_dist = 0
        for raw_coord in raw_list.T:
            distance = dist.euclidean((mean_x, mean_y, mean_z), raw_coord)
            total_raw_dist += distance

        if debug:
            print ('avg reg dist: ' + str(total_reg_dist / nframes))
            print ('avg raw dist: ' + str(total_raw_dist / nframes))

        total_avg_raw_dist += total_raw_dist / nframes
        total_avg_reg_dist += total_reg_dist / nframes

    registered_avg = total_avg_reg_dist / nlabels
    original_avg = total_avg_raw_dist / nlabels
    registration_ratio = registered_avg / original_avg
    if verbose:
        print ('Registered avg:\t' + str(registered_avg))
        print ('Original avg:\t' + str(original_avg))
        print ('Ratio: ' + str(registration_ratio))

        if registered_avg < original_avg:
            print ('Clusters more dense with registration')
        else:
            print ('Clusters less dense with registration')

    return registration_ratio


def global_signal(path, verbose=False, debug=False):
    import numpy as np
    import sys
    from scipy import stats
    import memotrack.io
    import pandas as pd
    import javabridge
    from skimage.filters import threshold_otsu as thresh_func
    import scipy.ndimage

    if verbose:
        print ('Analysing file ' + str(path))

    meta = memotrack.io.meta(path, verbose=verbose)

    nframes = meta['SizeT']
    nslices = meta['SizeZ']

    if debug:
        nframes = debug

    all_intensity_list = []
    foreground_intensity_list = []
    background_intensity_list = []

    print ('Loading and calculating average of frames...')
    for frame in range(nframes):
        img = memotrack.io.read(path, meta, frame=frame, channel=1)

        # Getting foreground and background
        thresh = thresh_func(img)  # Gets threshold value
        foreground = img[img > thresh]
        background = img[img < thresh]

        # Use lines bellow for mean of the projection
        # img_projected = np.max(img, axis=0)
        # intensity_list.append(np.mean(img_projected))

        # Line bellow for the mean of 3D volume
        all_intensity_list.append(np.mean(img))
        foreground_intensity_list.append(np.mean(foreground))
        background_intensity_list.append(np.mean(background))

    print ('\nStarting analysis...'),
    intensity_lists = [all_intensity_list, foreground_intensity_list, background_intensity_list]
    names = iter(['all', 'foreground', 'background'])
    global_df = pd.DataFrame()
    for intensity_list in intensity_lists:
        name = next(names)
        # Photobleacing correction, ignoring 10 first frames
        slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(intensity_list[9:])),
                                                                       intensity_list[9:])
        if verbose:
            print('\nLinear photobleaching correction:')
            print ('slope: ' + str(slope) + '\tintercept: ' + str(intercept) + '\tr2: ' + str(r_value ** 2))
            sys.stdout.flush()

        correction_list = [abs(i * slope) for i in range(len(intensity_list))]
        unbleached = [x + y for x, y in zip(correction_list, intensity_list)]

        # Delta F over F0 calculation
        sigma = 2
        smoothed = scipy.ndimage.filters.gaussian_filter1d(unbleached, sigma)

        # Get F0, the minimum before stim window
        F0 = np.min(smoothed[0:25])

        # Calculate delta F
        deltaF = unbleached - F0

        # Calcultate deltaF over F0
        deltaF_F0 = deltaF / F0

        # Smooth again
        deltaF_F0 = scipy.ndimage.filters.gaussian_filter1d(deltaF_F0, sigma)

        # Add lists to dataframe
        global_df[name + '_intensity'] = intensity_list
        global_df[name + '_unbleached'] = unbleached
        global_df[name + '_deltaF'] = deltaF_F0

    print ('global_df')
    print (global_df)

    memotrack.io.write_df(global_df, str(path[:-4]) + '_global_signal.csv')
    print ('[Done]')
    # Kill the JVM
    if verbose:
        print ('Killing JVM'),
        sys.stdout.flush()
    javabridge.kill_vm()

    return intensity_list, unbleached, deltaF_F0


def similarity_matrix(detections_df, stim_sequence, verbose=False,
                      block_size=40, stim_start=10, stim_duration=25, intensity='filtered', condor=False,
                      metric='euclidean', only_responsive=True, save=False, return_matrix=False):
    import numpy as np
    from scipy.spatial import distance
    import scipy.cluster.hierarchy as sch
    import pylab
    import matplotlib.pyplot as plt

    nlabels = detections_df.label.nunique()

    if only_responsive:
        # We want only the responsive neurons, to reduce dimensionality
        responsive_df = detections_df[detections_df['responsive'] == True]
    else:
        responsive_df = detections_df

    nresponsive = responsive_df.label.nunique()

    if verbose:
        print ('Checking ' + str(nlabels) + ' labels with ' + str(nresponsive) + ' responsive neurons...')

    # Checking how many stim we have
    nblocks = len(stim_sequence)
    nframes_check = nblocks * block_size
    nframes_real = responsive_df.t.nunique()
    nS = stim_sequence.count('S')
    nA = stim_sequence.count('A')
    nO = stim_sequence.count('O')
    nM = stim_sequence.count('M')
    if verbose:
        print ('Blocks count on ' + stim_sequence + ':')
        print (str(nA) + ' air'),
        print ('\t' + str(nO) + ' oct'),
        print ('\t' + str(nM) + ' mch'),
        print ('\t' + str(nS) + ' without')
        print ('Frames needed: ' + str(nframes_check) + '\tFrames found: ' + str(nframes_real))
        print ('')

    array_list = []
    current_frame = 0
    for stim in stim_sequence:
        frames_list = range(current_frame + stim_start, current_frame + stim_start + stim_duration)
        current_frame = current_frame + block_size
        if verbose:
            print ('Processing ' + stim + ','),
            print ('frames ' + str(min(frames_list)) + ' to ' + str(max(frames_list)))
        temp_df = responsive_df[['label', 't', intensity]]  # only the colums we need
        temp_df = temp_df[(temp_df['t'] >= min(frames_list)) & (temp_df['t'] < max(frames_list))]  # crop the time
        temp_df = temp_df.groupby(by='label').max()  # average grouped by label
        del (temp_df['t'])  # take time out because it doesnt make sense anymore (it's the average time)
        temp_df = temp_df.reset_index()  # reset index to have label back as column
        # if verbose:
        #    print (temp_df)
        array_list.append(temp_df[intensity].get_values())

    if verbose:
        print ('Done, array list with ' + str(np.shape(array_list)[0]) + ' elements and ' +
               str(np.shape(array_list)[1]) + ' dimensions')

    # Build distance matrix
    dist_matrix = np.zeros([nblocks, nblocks])
    i = 0
    j = 0
    for a1 in array_list:
        for a2 in array_list:
            dist_matrix[i, j] = distance.pdist([a1, a2], metric=metric)
            j += 1
        i += 1
        j = 0

    # create labels list
    labels_list = []
    label_counter = 1
    for label in stim_sequence:
        labels_list.append(str(label_counter) + '-' + label)
        label_counter += 1

    # set dendogram colors
    sch.set_link_color_palette(['#888888', '#AA00FF', '#777777', '#888888'])

    # Compute and plot first dendrogram.
    fig = pylab.figure(figsize=(10, 10))
    ax1 = fig.add_axes([0.11, 0.1, 0.2, 0.55])
    Y1 = sch.linkage(dist_matrix, method='complete', metric=metric)
    Z1 = sch.dendrogram(Y1, orientation='left', labels=labels_list, color_threshold=float('inf'))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.35, 0.68, 0.55, 0.2])
    Y2 = sch.linkage(dist_matrix, method='complete', metric=metric)
    Z2 = sch.dendrogram(Y2, labels=labels_list, color_threshold=float('inf'))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.35, 0.1, 0.55, 0.55])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    dist_matrix = dist_matrix[idx1, :]
    dist_matrix = dist_matrix[:, idx2]
    im = axmatrix.matshow(dist_matrix, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
    axmatrix.set_xticks(range(nblocks))
    axmatrix.set_yticks(range(nblocks))

    # create idx1 labels
    idx1_labels = []
    for idx in idx1:
        idx1_labels.append(labels_list[idx])

    # create idx1 labels
    idx2_labels = []
    for idx in idx2:
        idx2_labels.append(labels_list[idx])

    axmatrix.set_xticklabels(idx1_labels, minor=False)
    axmatrix.set_yticklabels(idx2_labels, minor=False)

    # Plot colorbar.
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.55])
    pylab.colorbar(im, cax=axcolor)

    # set title
    if only_responsive:
        pylab.ylabel(metric + ' distance for responsive neurons')
    else:
        pylab.ylabel(metric + ' distance for all neurons')

    if save:
        fig.savefig(save, bbox_inches='tight')
        plt.close()

    if return_matrix:
        return dist_matrix, stim_sequence


def report(file_path, trial_size=40, stim_delay=10, stim_duration=5, condor=False, save_figs=True, paperfig=True):
    import os
    import memotrack
    import sys
    import matplotlib.pyplot as plt
    import time

    print ('\nStarting report...')
    print (file_path)
    sys.stdout.flush()

    base_path, file_name = os.path.split(file_path)
    base_path2, stim = os.path.split(base_path)
    base_path3, fly = os.path.split(base_path2)
    nstim = len(stim)

    if paperfig:
        nstim = 3  # Manual override because of paper data

    stim_frames = range(stim_delay, nstim * trial_size, trial_size)

    save_path = base_path + os.sep + fly + '_' + stim

    # Writing to file that we finished this job
    f = open('/projects/memotrack/temp/report/finished_jobs.txt', 'a')
    f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    f.write(' | Job finished for fly {}'.format(fly))
    f.write('\n')
    f.close()

    if save_figs:
        save_path2 = '/projects/memotrack/temp/report/' + fly

        if os.path.isdir(save_path2) is False:
            os.mkdir(save_path2)

    print ('Fly: ' + fly)
    print ('Stim: ' + stim + ' (' + str(nstim) + ' trials)')
    print ('Stim frames: ' + str(stim_frames))

    detections_df = memotrack.io.read_df(file_path)
    sys.stdout.flush()

    nlabels = int(detections_df['label'].nunique())
    cmap = memotrack.display.rand_cmap(nlabels, type='super_bright', first_color_black=False, verbose=(not condor),
                                       condor=condor)

    # Signals Raw data
    memotrack.display.plot_1D_signals(detections_df, normalize=False, smooth=0,
                                      accent=True, only_responsive=False, only_positive=True,
                                      stim_frame=stim_frames,
                                      stim_duration=stim_duration,
                                      cmap=cmap, title=fly + '_' + stim,
                                      save=False, HighDPI=False, ymax=False)
    if save_figs:
        plt.savefig(save_path + '_RawSignals.png')
        plt.savefig(save_path2 + '/03_RawSignals.png')

    # Signals Filtered data
    memotrack.display.plot_1D_signals(detections_df, normalize='filtered', smooth=0,
                                      accent=True, only_responsive=False, only_positive=False,
                                      stim_frame=stim_frames,
                                      stim_duration=stim_duration,
                                      cmap=cmap, title=fly + '_' + stim,
                                      save=False, HighDPI=False, ymax=False)
    if save_figs:
        plt.savefig(save_path + '_FilteredSignals.png')
        plt.savefig(save_path2 + '/04_FilteredSignals.png')

    # Signals Normalized data
    memotrack.display.plot_1D_signals(detections_df, normalize='normalized', smooth=0,
                                      accent=True, only_responsive=False, only_positive=False,
                                      stim_frame=stim_frames,
                                      stim_duration=stim_duration,
                                      cmap=cmap, title=fly + '_' + stim,
                                      save=False, HighDPI=False, ymax=False)
    if save_figs:
        plt.savefig(save_path + '_NormSignals.png')
        plt.savefig(save_path2 + '/05_NormSignals.png')

    # Signals block
    memotrack.display.signals(detections_df, cmap=cmap, intensity='normalized', ymax=1.0,
                              stim_sequence=stim, only_responsive='comparison', only_positive=False,
                              block_size=trial_size, stim_start=stim_delay, stim_duration=stim_duration,
                              save=False, empty_plot=False)
    if save_figs:
        plt.savefig(save_path + '_NormSignals_Blocks.png')
        plt.savefig(save_path2 + '/06_NormSignals_Blocks.png')

    # Similarity Matrix
    if not paperfig:
        memotrack.analyse.similarity_matrix(detections_df, stim_sequence=stim,
                                            verbose=True, block_size=trial_size,
                                            stim_start=stim_delay - 0, stim_duration=15,
                                            intensity='norm_intensity', metric='cosine', only_responsive=True)
        if save_figs:
                plt.savefig(save_path + '_Matrix.png')
                plt.savefig(save_path2 + '/07_Matrix.png')

    # Scatter original coords
    memotrack.display.plot_from_df(detections_df, new_cmap=cmap, size=1, elev=15, azim=30,
                                   crop_data=True, auto_fit=True, one_frame=False,
                                   frame=9, time_color=True, intensity=False, borders=True,
                                   registered_coords='original', title=fly + '_' + stim,
                                   save=False, HighDPI=False, verbose=True,
                                   lim=10, zstart=268, ystart=338, xstart=38)
    if save_figs:
        plt.savefig(save_path + '_OriginalCoordinates.png')
        plt.savefig(save_path2 + '/01_OriginalCoordinates.png')

    # Sacatter clusters
    memotrack.display.plot_from_df(detections_df, new_cmap=cmap, size=1, elev=15, azim=30,
                                   crop_data=True, auto_fit=True, one_frame=False,
                                   frame=9, time_color=False, intensity=False, borders=True,
                                   registered_coords='cpd', title=fly + '_' + stim,
                                   save=False, HighDPI=False, verbose=True,
                                   lim=10, zstart=268, ystart=338, xstart=38)
    if save_figs:
        plt.savefig(save_path + '_Clusters.png')
        plt.savefig(save_path2 + '/02_Clusters.png')

    # Signal scatter corner
    memotrack.display.plot_from_df(detections_df, new_cmap=cmap, size=200, elev=15, azim=30,
                                   crop_data=True, auto_fit=True, one_frame=False,
                                   frame=9, time_color=False, intensity='df', borders=True,
                                   registered_coords=True, title=fly + '_' + stim,
                                   save=False, HighDPI=False, verbose=True,
                                   lim=10, zstart=268, ystart=338, xstart=38)
    if save_figs:
        plt.savefig(save_path + '_ScatterSignal.png')
        plt.savefig(save_path2 + '/08_ScatterSignal.png')

    # Signal scatter top
    memotrack.display.plot_from_df(detections_df, new_cmap=cmap, size=100, elev=-90, azim=-90,
                                   crop_data=False, auto_fit=False, one_frame=False,
                                   frame=9, time_color=False, intensity='df', borders=True,
                                   registered_coords=True, title=fly + '_' + stim,
                                   save=False, HighDPI=False, verbose=True,
                                   lim=512, zstart=0, ystart=0, xstart=0)
    if save_figs:
        plt.savefig(save_path + '_ScatterSignal_TopView.png')
        plt.savefig(save_path2 + '/09_ScatterSignal_TopView.png')


def group_results(files_path, ignore_list=False, stim_start=89, stim_duration=10, oct_ratio_thresh=0.0,
                  responsive_threshold=0.2, verbose=True, peak_fold=1.0, ndetection_norm=True, crosscheck=False):
    import memotrack
    import memotrack.analyse
    import sys
    import os
    import pandas as pd
    import numpy as np
    import scipy.ndimage

    # stim_blocks = [49, 89, 129, 169]
    stim_blocks = [49, 89]
    trial_size = 40  # This is the size of each trial (S, A, O or M)
    stim_delay = 9  # Delay in frames for the beginning of stimulus
    stim_duration = 5  # Duration of the stim
    stim = 'SAO'

    # Get list of paths
    text_file = open(files_path, 'r')
    path_list = text_file.read().split('\n')
    text_file.close()

    # Start dataframe
    results_df = pd.DataFrame()

    # Start counters
    analysed_files = 0
    missing_files = 0

    if not ignore_list:
        ignore_list = []

    for ignore in ignore_list:
        if ignore in path_list:
            path_list.remove(ignore)
            print ('Ignore list: {}'.format(ignore))

    if crosscheck:
        # Check paths for normalized dataframe
        path_list = [p for p in path_list if os.path.isfile((str(p[:-4]) + '_crosscheck.csv'))]
    else:
        path_list = [p for p in path_list if os.path.isfile((str(p[:-4]) + '_normalized.csv'))]

    # Here we want to update the ignore list with the "epileptic flies"
    stim_frames = []  # list of frames where we have either odor or air
    for block in stim_blocks:
        for step in range(stim_duration):
            stim_frames.append(block + step)
    if verbose:
        print ('\nResponsiveness check, using peak count fold of {}'.format(peak_fold))
        print ('Stim blocks starting at {}'.format(stim_blocks)),
        print ('with duration of {} frames'.format(stim_duration))
        print ('Threshold of {} for responsive neurons'.format(responsive_threshold))
        sys.stdout.flush()

    # Run every file

    for path in path_list:
        if crosscheck:
            df_path = (str(path[:-4]) + '_crosscheck.csv')
        else:
            df_path = (str(path[:-4]) + '_normalized.csv')

        fly_name = os.path.basename(path)[:-11]
        if verbose:
            print ('\n' + str(fly_name))
        sys.stdout.flush()

        fly_stim_peaks = 0
        fly_no_stim_peaks = 0

        # Load detections dataframe with normalized data
        detections_df = memotrack.io.read_df(df_path)

        nframes = detections_df['t'].nunique()
        nframes_stim = float(len(stim_frames))
        nframes_no_stim = float(nframes - nframes_stim)

        # Calculate the OCT ratio for this df
        oct_ratio = memotrack.analyse.oct_ratio(detections_df, stim_sequence=stim,
                                                verbose=verbose, block_size=trial_size,
                                                stim_start=stim_delay, stim_duration=stim_duration,
                                                intensity='norm_intensity', metric='cosine', only_responsive=True)

        # Check peaks in "stim region"
        stim_df = detections_df[detections_df['t'].isin(stim_frames)]
        no_stim_df = detections_df[~detections_df['t'].isin(stim_frames)]

        nlabels = int(detections_df['label'].nunique())

        # Run every label of this file
        for label in range(nlabels):

            stim_peak = stim_df[stim_df['label'] == label].norm_intensity.max()
            no_stim_peak = no_stim_df[no_stim_df['label'] == label].norm_intensity.max()

            if responsive_threshold < stim_peak < 100:
                fly_stim_peaks += 1

            if responsive_threshold < no_stim_peak < 100:
                fly_no_stim_peaks += 1

        if ndetection_norm:
            if verbose:
                print ('Total of {} detections'.format(nlabels))
            fly_stim_peaks /= float(nlabels)
            fly_no_stim_peaks /= float(nlabels)

        norm_stim_peaks = fly_stim_peaks / nframes_stim
        norm_no_stim_peaks = fly_no_stim_peaks / nframes_no_stim

        if verbose:
            print ('Inside stim peaks:  {:.3f}\t({:.3f})'.format(fly_stim_peaks, norm_stim_peaks))
            print ('Outside stim peaks: {:.3f}\t({:.3f})'.format(fly_no_stim_peaks, norm_no_stim_peaks))
            print ('Oct ratio: {:.3f}'.format(oct_ratio))

        if norm_no_stim_peaks * peak_fold >= norm_stim_peaks:
            # if oct_ratio > oct_ratio_thresh:  # Here "inf" to avoid problematic cases
            ignore_list.append(path)
            path_list.remove(path)
            if verbose:
                print ('\033[1mFly added to ignore list !\033[0m')

    # Run every file, measuring the peaks
    if verbose:
        print ('\n\nPeak count for the window {}-{}'.format(stim_start, stim_start + stim_duration))

    oct_ratio_list = []

    for path in path_list:
        if crosscheck:
            df_path = (str(path[:-4]) + '_crosscheck.csv')
        else:
            df_path = (str(path[:-4]) + '_normalized.csv')
        npeaks = 0

        # Check if we have results

        fly_name = os.path.basename(path)[:-11]
        if verbose:
            print ('\n' + str(fly_name))
        sys.stdout.flush()

        # Load detections dataframe with normalized data
        detections_df = memotrack.io.read_df(df_path)

        # Calculate the OCT ratio for this df
        oct_ratio = memotrack.analyse.oct_ratio(detections_df, stim_sequence=stim,
                                                verbose=verbose, block_size=trial_size,
                                                stim_start=stim_delay, stim_duration=stim_duration,
                                                intensity='norm_intensity', metric='cosine', only_responsive=True)
        oct_ratio_list.append(oct_ratio)

        # Cut the dataframe, to have only stim window
        detections_df = detections_df[
            (detections_df['t'] > stim_start) & (detections_df['t'] < stim_start + stim_duration)]

        # Get number of unique labels
        nlabels = int(detections_df['label'].nunique())
        name = path[-17:-4]

        if name[:2] == '00':
            condition = 'naive'
        if name[:2] == '01':
            condition = 'unpaired1'
        if name[:2] == '02':
            condition = 'unpaired2'
        if name[:2] == '10':
            condition = 'paired1'
        if name[:2] == '20':
            condition = 'paired2'

        # Run every label of this file
        # max_df = detections_df.groupby(by='label').max().reset_index()
        # print (max_df)
        for label in range(nlabels):
            # start results dictionary
            data_for_df = {'condition': condition, 'fly': name[:6]}

            peak = detections_df[detections_df['label'] == label].norm_intensity.max()
            raw_peak = detections_df[detections_df['label'] == label].raw_intensity.max()
            raw_mean = detections_df[detections_df['label'] == label].raw_intensity.mean()

            # First check if passed the crosstalk check:
            if crosscheck:
                if sum(detections_df[detections_df['label'] == label]['responsive_crosscheck']) > 0:
                    if responsive_threshold < peak < np.inf:  # Here "inf" to avoid problematic cases
                        data_for_df['peak'] = peak
                        data_for_df['raw_peak'] = raw_peak / raw_mean

                        if ndetection_norm:
                            npeaks += 1 / float(nlabels)
                            data_for_df['responsive'] = 1 / float(nlabels)
                        else:
                            npeaks += 1
                            data_for_df['responsive'] = 1

                        # Concatenate new data
                        temp_df = pd.DataFrame(data_for_df, index=[0])
                        results_df = pd.concat([results_df, temp_df], ignore_index=True)
            else:
                if responsive_threshold < peak < np.inf:  # Here "inf" to avoid problematic cases
                    data_for_df['peak'] = peak
                    data_for_df['raw_peak'] = raw_peak / raw_mean

                    if ndetection_norm:
                        npeaks += 1 / float(nlabels)
                        data_for_df['responsive'] = 1 / float(nlabels)
                    else:
                        npeaks += 1
                        data_for_df['responsive'] = 1

                    # Concatenate new data
                    temp_df = pd.DataFrame(data_for_df, index=[0])
                    results_df = pd.concat([results_df, temp_df], ignore_index=True)
        if verbose:
            print ('Total of {} neurons, with {} peaks ({:.2%})'.format(nlabels, npeaks, (npeaks / float(nlabels))))
        analysed_files += 1

    if verbose:
        print ('\n\nDone, total of ' + str(analysed_files) + ' files analyzed, missing (or ignoring) ' + str(
            missing_files) + ' results.')
        print ('Mean OCT ratio: {}'.format(np.mean(oct_ratio_list)))

    sys.stdout.flush()

    arg_sort = np.argsort(oct_ratio_list)
    top_worst = arg_sort[0:3]
    top_best = arg_sort[-3:]

    print ('\nRanked Flies')

    rank = len(arg_sort)
    for fly in arg_sort:
        print (str(rank) + ': '),
        print ('{} ({})'.format(path_list[fly], oct_ratio_list[fly]))
        rank -= 1

    print ('\nWorst flies:')
    for worst in top_worst:
        print ('{} ({})'.format(path_list[worst], oct_ratio_list[worst]))

    print ('\nBest flies:')
    for best in top_best:
        print ('{} ({})'.format(path_list[best], oct_ratio_list[best]))

    return results_df, oct_ratio_list


def fly_info(fly_name, stim_sequence='SAOAOS'):
    import os
    import time
    import math

    print ('Fly:\t\t{}'.format(fly_name))
    tif_path = '/projects/memotrack/data/' + fly_name + '/' + stim_sequence + '/' + fly_name + '_' + stim_sequence + '.tif'
    print ('Path:\t\t{}'.format(tif_path))
    print ('Import:\t\t{}'.format(time.strftime("%d %b %Y  |  %Hh%Mm", time.localtime(os.path.getmtime(tif_path)))))

    print (
        'Process:\t{}'.format(
            time.strftime("%d %b %Y  |  %Hh%Mm", time.localtime(os.path.getmtime(tif_path + '.out')))))

    report_path = '/projects/memotrack/temp/report/' + fly_name + '/01_OriginalCoordinates.png'

    if os.path.isfile(report_path):
        print (
            'Report:\t\t{}'.format(time.strftime("%d %b %Y  |  %Hh%Mm", time.localtime(os.path.getmtime(report_path)))))
    else:
        print ('\n\033[1mWarning\033[0;0m: Report not found, probably file process is not done yet !')
        return

    process_epoch = os.path.getmtime(tif_path + '.out')
    report_epoch = os.path.getmtime(report_path)

    if process_epoch - report_epoch > 3600 * 24:
        print ('\n\033[1mWarning\033[0;0m: Report is older than last process. '
               'Probably the newest version is still running !')


def oct_ratio(detections_df, stim_sequence, verbose=False,
              block_size=40, stim_start=10, stim_duration=10, intensity='norm_intensity',
              metric='cosine', only_responsive=True, debug=False):
    import numpy as np
    from scipy.spatial import distance

    nlabels = detections_df.label.nunique()

    if only_responsive:
        # We want only the responsive neurons, to reduce dimensionality
        responsive_df = detections_df[detections_df['responsive'] == True]
    else:
        responsive_df = detections_df

    nresponsive = responsive_df.label.nunique()

    if verbose > 1:
        print ('Checking ' + str(nlabels) + ' labels with ' + str(nresponsive) + ' responsive neurons...')

    # Checking how many stim we have
    nblocks = len(stim_sequence)
    nframes_check = nblocks * block_size
    nframes_real = responsive_df.t.nunique()
    nS = stim_sequence.count('S')
    nA = stim_sequence.count('A')
    nO = stim_sequence.count('O')
    nM = stim_sequence.count('M')
    if verbose > 1:
        print ('Blocks count on ' + stim_sequence + ':')
        print (str(nA) + ' air'),
        print ('\t' + str(nO) + ' oct'),
        print ('\t' + str(nM) + ' mch'),
        print ('\t' + str(nS) + ' without')
        print ('Frames needed: ' + str(nframes_check) + '\tFrames found: ' + str(nframes_real))
        print ('')

    array_list = []
    current_frame = 0
    for stim in stim_sequence:
        frames_list = range(current_frame + stim_start, current_frame + stim_start + stim_duration)
        current_frame += block_size

        if verbose > 1:
            print ('Processing ' + stim + ','),
            print ('frames ' + str(min(frames_list)) + ' to ' + str(max(frames_list)))
        temp_df = responsive_df[['label', 't', intensity]]  # only the colums we need
        temp_df = temp_df[(temp_df['t'] >= min(frames_list)) & (temp_df['t'] < max(frames_list))]  # crop the time
        temp_df = temp_df.groupby(by='label').max()  # average grouped by label
        del (temp_df['t'])  # take time out because it doesnt make sense anymore (it's the average time)
        temp_df = temp_df.reset_index()  # reset index to have label back as column

        array_list.append(temp_df[intensity].get_values())

    if verbose > 1:
        print ('Done, array list with ' + str(np.shape(array_list)[0]) + ' elements and ' +
               str(np.shape(array_list)[1]) + ' dimensions')

    # Build distance matrix
    dist_matrix = np.zeros([nblocks, nblocks])
    i = 0
    j = 0
    for a1 in array_list:
        for a2 in array_list:
            dist_matrix[i, j] = distance.pdist([a1, a2], metric=metric)
            j += 1
        i += 1
        j = 0

    #plt.imshow(dist_matrix, cmap='viridis_r')
    #plt.xticks(np.arange(6), ['S','A','O','A','O','S'])
    #plt.yticks(np.arange(6), ['S','A','O','A','O','S'])

    mtop = dist_matrix[2, 0]

    if debug:
        print (' \nmatrix shape: {}'.format(np.shape(dist_matrix)))
        print (dist_matrix)
        print ('mtop: {}'.format(mtop))

    ratio = mtop / np.sum(dist_matrix)

    if verbose:
        print ('OCT ratio: {:.5f}'.format(ratio))

    return ratio


def graph_features(df, max_dist=50, min_intensity=0.25, verbose=True):
    print ('Hello graphs !')

    return


def stim_intensity_array(detections_df, stim_sequence='SAO', verbose=False,
                         block_size=40, stim_start=8, stim_duration=10, intensity='norm_intensity',
                         only_responsive=False):
    import numpy as np
    from scipy.spatial import distance

    nlabels = detections_df.label.nunique()

    if only_responsive:
        # We want only the responsive neurons, to reduce dimensionality
        responsive_df = detections_df[detections_df['responsive'] == True]
    else:
        responsive_df = detections_df

    nresponsive = responsive_df.label.nunique()

    if verbose > 1:
        print ('Checking ' + str(nlabels) + ' labels with ' + str(nresponsive) + ' responsive neurons...')

    # Checking how many stim we have
    nblocks = len(stim_sequence)
    nframes_check = nblocks * block_size
    nframes_real = responsive_df.t.nunique()
    nS = stim_sequence.count('S')
    nA = stim_sequence.count('A')
    nO = stim_sequence.count('O')
    nM = stim_sequence.count('M')

    if verbose > 1:
        print ('Blocks count on ' + stim_sequence + ':')
        print (str(nA) + ' air'),
        print ('\t' + str(nO) + ' oct'),
        print ('\t' + str(nM) + ' mch'),
        print ('\t' + str(nS) + ' without')
        print ('Frames needed: ' + str(nframes_check) + '\tFrames found: ' + str(nframes_real))
        print ('')

    array_list = []
    current_frame = 0
    for stim in stim_sequence:
        frames_list = range(current_frame + stim_start, current_frame + stim_start + stim_duration)
        current_frame += block_size

        if verbose > 1:
            print ('Processing ' + stim + ','),
            print ('frames ' + str(min(frames_list)) + ' to ' + str(max(frames_list)))
        temp_df = responsive_df[['label', 't', intensity]]  # only the colums we need
        temp_df = temp_df[(temp_df['t'] >= min(frames_list)) & (temp_df['t'] < max(frames_list))]  # crop the time
        # temp_df = temp_df[temp_df[intensity] > 0.0]
        temp_df = temp_df.groupby(by='label').quantile(q=0.99)  # average grouped by label
        del (temp_df['t'])  # take time out because it doesnt make sense anymore (it's the average time)
        temp_df = temp_df.reset_index()  # reset index to have label back as column

        array_list.append(temp_df[intensity].get_values())

    if verbose > 1:
        print ('Done, array list with ' + str(np.shape(array_list)[0]) + ' elements and ' +
               str(np.shape(array_list)[1]) + ' dimensions')

    return array_list


def generate_ignore_list(files_path, display=False):
    import os
    import numpy as np
    import memotrack
    import matplotlib.pyplot as plt
    from scipy import stats
    import pandas as pd

    print ('Generating ignore list...\n')

    # Get list of paths
    text_file = open(files_path, 'r')
    path_list = text_file.read().split('\n')
    text_file.close()

    # Start dataframe
    results_df = pd.DataFrame()

    # Start counters
    analysed_files = 0
    missing_files = 0

    # Check paths for normalized dataframe
    path_list = [p for p in path_list if os.path.isfile((str(p[:-4]) + '_normalized.csv'))]

    ignore_list = []
    prob_list = []

    total_flies = 0
    ignored_flies = 0
    for path in path_list:
        print (path)
        total_flies += 1
        df_path = (str(path[:-4]) + '_normalized.csv')

        detections_df = memotrack.io.read_df(df_path)

        array = memotrack.analyse.stim_intensity_array(detections_df)

        print ('Total number of points : {}'.format(len(array[0])))

        subset = 10
        array1 = np.sort(array[0])[-subset:]
        array2 = np.sort(array[2])[-subset:]

        print ('(Using subset of {} points)'.format(len(array1)))

        # print ('Mean contol: {}'.format(np.mean(array1)))
        # print ('Mean oct:    {}'.format(np.mean(array2)))

        t, prob = stats.mannwhitneyu(array1, array2, alternative='less')
        prob_list.append(prob)

        if prob < 0.01:
            print ('\033[92m\033[1mp-value: ' + str(prob) + '\033[0m')

        elif prob < 0.05:
            print ('\033[93m\033[1mp-value: ' + str(prob) + '\033[0m')

        else:
            print ('\033[91m\033[1mp-value: ' + str(prob) + '\033[0m')

        if display:
            memotrack.display.boxplot([array1, array2], xticks=['Control', 'OCT'], xtitle=path,
                                      test_type='mw_less', log_transform=False)

        if prob > 0.01:
            ignored_flies += 1
            ignore_list.append(path)
            print ('\033[1mAdded to ignore list !\033[0m')

        print ('')
        plt.show()

    print ('\nTotal of {} flies before filter, {} being ignored (Thus, {} remain on the database)'.format(total_flies, ignored_flies, (total_flies-ignored_flies)))

    return ignore_list, prob_list


def graph_features(path_file, display=False, RT=0.10, frame_range=[89,95]):
    import memotrack
    import sys
    import os
    import pandas as pd
    import numpy as np
    import scipy.ndimage
    import csv
    import scipy.spatial
    import networkx as nx
    import scipy.spatial.distance as distance
    import itertools
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    with open(path_file, 'rb') as f:
        reader = csv.reader(f)
        path_list = list(reader)

    #path_list = path_list[:2]

    features_df = pd.DataFrame()

    for path in path_list:
        if len(path) > 0:
            path = path[0]
            path_data = path[:-4] + '_normalized.csv'
            fly = path[-17:-11]
            print ('\n')
            if os.path.isfile(path_data):
                features_temp_df = pd.DataFrame()
                df = memotrack.io.read_df(path_data)
                print (fly)
                features_temp_df['fly'] = [fly]
                sys.stdout.flush()

                temp_df = df[(df['t'] > frame_range[0]) & (df['t'] < frame_range[1])]
                temp_df = temp_df.groupby(by='label').quantile(q=0.99)
                temp_df.reset_index(inplace=True)

                #temp_df = df[df['t'] == frame].copy(deep=True)

                # Generate deaunay
                x_list = temp_df['x'].get_values()
                y_list = temp_df['y'].get_values()
                z_list = temp_df['z'].get_values()
                coords = np.transpose([x_list, y_list, z_list])
                delTri = scipy.spatial.Delaunay(coords)

                # Create networks

                G = nx.Graph()

                # Add nodes
                for n in xrange(len(delTri.points)):
                    G.add_node(n, attr_dict={'coords': list(delTri.points[n])})

                # Create edges
                max_dist = 50
                min_intensity = RT
                df_mean_intensity = temp_df['norm_intensity'].mean()
                df_max_intensity = temp_df['norm_intensity'].max()
                print ('Creating graph...'),
                for simplex in delTri.simplices:
                    sys.stdout.flush()
                    for u, v in itertools.permutations(range(4), 2):
                        # Distance threshold
                        if distance.pdist([delTri.points[simplex[u]], delTri.points[simplex[v]]]) < max_dist:

                            # Intensity threshold
                            u_intensity = temp_df[temp_df['label'] == simplex[u]]['norm_intensity'].get_values()
                            v_intensity = temp_df[temp_df['label'] == simplex[v]]['norm_intensity'].get_values()

                            if u_intensity > min_intensity and v_intensity > min_intensity:
                                # print ('!'),
                                mean_intensity = np.mean([u_intensity, v_intensity])
                                if mean_intensity < 0:
                                    mean_intensity = 0

                                # Add edge
                                G.add_edge(simplex[u], simplex[v],
                                           attr_dict={'intensity': mean_intensity,
                                                      'distance': -mean_intensity + df_max_intensity})

                print ('[Done]')

                if display:
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(111, projection='3d')
                    elev = 15
                    azim = 30
                    ax.view_init(elev, azim)
                    plt.ion()

                    # Draw dots
                    for n in range(len(G.node)):
                        x = G.node[n]['coords'][0]
                        y = G.node[n]['coords'][1]
                        z = G.node[n]['coords'][2]
                        ax.scatter(x, y, z, color='#FF773D', s=30, alpha=0.8)

                    # Draw edges
                    edge_accent = 'intensity'
                    for start in G.edge:
                        for end in G.edge[start].keys():
                            xs = [G.node[start]['coords'][0], G.node[end]['coords'][0]]
                            ys = [G.node[start]['coords'][1], G.node[end]['coords'][1]]
                            zs = [G.node[start]['coords'][2], G.node[end]['coords'][2]]

                            if edge_accent:

                                if G.edge[start][end][edge_accent] > 1:
                                    ax.plot(xs, ys, zs, color='#454545', alpha=1.0)
                                elif G.edge[start][end][edge_accent] < 0:
                                    ax.plot(xs, ys, zs, color='#454545', alpha=0.0)
                                else:
                                    ax.plot(xs, ys, zs, color='#454545', alpha=G.edge[start][end][edge_accent])
                            else:
                                ax.plot(xs, ys, zs, color='#454545', alpha=1.0)
                    plt.show()

                print ('Global features:')
                ecc = None
                print ('Total nodes:\t{}'.format(G.number_of_nodes()))
                features_temp_df['nnodes'] = G.number_of_nodes()
                print ('Total edges:\t{}'.format(G.number_of_edges()))
                features_temp_df['nedges'] = G.number_of_edges()
                if G.number_of_edges() > 0:

                    try:
                        ecc = nx.eccentricity(G)
                        print ('Diameter: {}'.format(nx.diameter(G, e=ecc)))
                        print ('Radius: {}'.format(nx.radius(G, e=ecc)))

                    except:
                        pass

                    print ('Density:\t{}'.format(nx.density(G)))
                    features_temp_df['density'] = nx.density(G)

                    print ('Avg Degree:\t{}'.format(np.mean(nx.average_neighbor_degree(G).values())))
                    features_temp_df['avg_degree'] = np.mean(nx.average_neighbor_degree(G).values())

                    print ('Assortativity:\t{}'.format(nx.degree_assortativity_coefficient(G)))
                    features_temp_df['assortativity'] = nx.degree_assortativity_coefficient(G)

                    print ('Centrality:\t{}'.format(np.mean(nx.degree_centrality(G).values())))
                    features_temp_df['centrality'] = np.mean(nx.degree_centrality(G).values())

                    print ('Edge load:\t{}'.format(np.mean(nx.edge_load(G).values())))
                    features_temp_df['edge_load'] = np.mean(nx.edge_load(G).values())

                    avg_cc_nodes = np.mean([len(n) for n in nx.connected_components(G) if len(n) > 1])
                    print ('Avg CC nodes:\t{}'.format(avg_cc_nodes))
                    features_temp_df['avg_cc_nodes'] = avg_cc_nodes

                    n_cc = len([n for n in nx.connected_components(G) if len(n) > 1])
                    print ('Number of CC:\t{}'.format(n_cc))
                    features_temp_df['n_cc'] = n_cc

                    conected_nodes = np.sum([len(n) for n in nx.connected_components(G) if len(n) > 1])
                    print ('Conected nodes:\t{}'.format(conected_nodes))
                    features_temp_df['conected_nodes'] = conected_nodes

                    single_nodes = np.sum([len(n) for n in nx.connected_components(G) if len(n) == 1])
                    print ('Single nodes:\t{}'.format(single_nodes))
                    features_temp_df['single_nodes'] = single_nodes

                features_df = pd.concat([features_df, features_temp_df], ignore_index=True)

            else:
                print ('File not found for {}'.format(fly))

    return features_df


def responsive_distance_distribution(path_file, display=False, RT=0.10, frame_range=[89,95], show_matrix=True,
                                     ignore_list=False):
    import memotrack
    import sys
    import os
    import pandas as pd
    import numpy as np
    import csv
    import scipy.spatial.distance as distance
    import matplotlib.pyplot as plt

    # Get list of paths
    text_file = open(path_file, 'r')
    path_list = text_file.read().split('\n')
    text_file.close()

    #path_list = path_list[:2]

    distribution_df = pd.DataFrame()

    if not ignore_list:
        print ('Empty ignore list !')
        ignore_list = []

    for ignore in ignore_list:
        if ignore in path_list:
            path_list.remove(ignore)
            print ('Ignore list: {}'.format(ignore))

    for path in path_list:
        if len(path) > 0:
            path = path
            path_data = path[:-4] + '_crosscheck.csv'
            fly = path[-17:-11]
            print ('\n')
            if os.path.isfile(path_data):
                distribution_temp_df = pd.DataFrame()
                df = memotrack.io.read_df(path_data)
                print ('\033[1m{}\033[0m'.format(fly))
                distribution_temp_df['fly'] = [fly]
                sys.stdout.flush()

                if fly[:2] == '00':
                    distribution_temp_df['group'] = 'naive'

                if fly[:2] == '10':
                    distribution_temp_df['group'] = 'paired1'

                if fly[:2] == '01':
                    distribution_temp_df['group'] = 'unpaired1'

                if fly[:2] == '20':
                    distribution_temp_df['group'] = 'paired2'

                if fly[:2] == '02':
                    distribution_temp_df['group'] = 'unpaired2'

                temp_df = df[(df['t'] > frame_range[0]) & (df['t'] < frame_range[1])]
                temp_df = temp_df.groupby(by='label').quantile(q=0.99)
                # temp_df.reset_index(inplace=True)

                print ('Total detections: {}'.format(len(temp_df)))

                temp_df = temp_df[(temp_df['norm_intensity'] > RT) & (temp_df['responsive_crosscheck'] == True)]

                print ('Responsive neurons: {}'.format(len(temp_df)))

                xs = temp_df.x.values
                ys = temp_df.y.values
                zs = temp_df.z.values

                coords = np.transpose([xs, ys, zs])

                dist_matrix = distance.cdist(coords, coords, 'euclidean')

                if len(dist_matrix) > 1:

                    sorted_matrix = np.sort(dist_matrix, axis=0)

                    #print (sorted_matrix[1])
                    mean_closest = np.mean(sorted_matrix[1])
                    mean_10closest = np.mean(sorted_matrix[1:11])
                    nclosest10 = len(sorted_matrix[1][sorted_matrix[1] < 10]) / float(len(sorted_matrix[1]))
                    nclosest20 = len(sorted_matrix[1][sorted_matrix[1] < 20]) / float(len(sorted_matrix[1]))
                    mean_global = np.mean(dist_matrix)

                else:
                    mean_closest = np.nan
                    mean_10closest = np.nan
                    nclosest10 = np.nan
                    nclosest20 = np.nan
                    mean_global = np.nan

                distribution_temp_df['mean_closest'] = mean_closest
                distribution_temp_df['mean_10closest'] = mean_10closest
                distribution_temp_df['nclosest10'] = nclosest10
                distribution_temp_df['nclosest20'] = nclosest20
                distribution_temp_df['mean_global'] = mean_global

                print ('Mean distance to closest responsive: {}'.format(mean_closest))
                print ('Mean distance to 10 closest responsive: {}'.format(mean_10closest))
                print ('Ratio of points closer than 10: {}'.format(nclosest10))
                print ('Ratio of points closer than 20: {}'.format(nclosest20))
                print ('Global mean: {}'.format(mean_global))
                if show_matrix:
                        plt.imshow(dist_matrix, vmin=0, vmax=20)
                        plt.axis('off')
                        plt.tight_layout()
                        plt.colorbar()
                        plt.show()

                distribution_df = pd.concat([distribution_df, distribution_temp_df], ignore_index=True)

            else:
                print ('File not found for {}'.format(fly))

    return distribution_df
