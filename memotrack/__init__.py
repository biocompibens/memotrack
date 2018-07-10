import memotrack.io
import memotrack.process
import memotrack.display
import memotrack.analyse
import memotrack.synthetic
import memotrack.registration


def version(verbose=True):
    memotrack_version = '0.5'
    if verbose:
        print('Memotrack v' + memotrack_version)
    return memotrack_version


def show_vars(v):
    for item in v:
        print (str(item) + ': ' + str(v[item]))


def run(img_path, verbose=True, display=True, debug=False, FWHMxy=1.21,
        iterations=10, signalFWHM=1.25, write_results=True, filter_std_thresh=0.2, condor=False,
        channel=0, min_verb=True, pixel_size=False, spline_order=3, save_raster=False, save_interpolated=True,
        detection=True, registration=True, DBSCAN=True, quality=True, fix=True, back_projection=True, fetch=True,
        get_signal=True, quality_check=True, normalize=True, filtering=True, crosscheck=True, report=True,
        extra=True, truncate=120, RT=0.1):

    # create dict from given variables
    v = locals()

    import memotrack
    import sys

    # Print header and adjust variables
    v = memotrack.run_header(v)

    # Start detection
    if detection:
        v = memotrack.run_detection(v)
    else:
        print ('WARNING: skipping detection phase !')
        v['current_df'] = v['base_path'] + v['file_name'] + '_detections.csv'

    # Start registration
    if registration:
        v = memotrack.run_registration(v)
    else:
        print ('WARNING: skipping registration phase !')
        v['current_df'] = v['base_path'] + v['file_name'] + '_registered.csv'

    # Start DBSCAN
    if DBSCAN:
        v = memotrack.run_DBSCAN(v)
    else:
        print ('WARNING: skipping clustering phase !')
        v['current_df'] = v['base_path'] + v['file_name'] + '_labels.csv'

    # Calculate detection quality
    if quality:
        v = memotrack.run_quality(v)
    else:
        print ('WARNING: skipping cluster fix phase !')
        v['current_df'] = v['base_path'] + v['file_name'] + '_quality.csv'

    # Fix clusters
    if fix:
        v = memotrack.run_ClusterFix(v)
    else:
        print ('WARNING: skipping cluster fix phase !')
        v['current_df'] = v['base_path'] + v['file_name'] + '_fixed.csv'

    # Back-projection of the coordinates
    if back_projection:
        v = memotrack.run_BackProjection(v)
    else:
        print ('WARNING: skipping back-projection !')
        v['current_df'] = v['base_path'] + v['file_name'] + '_reprojected.csv'

    # Fetch signal
    if get_signal:
        v = memotrack.run_GetSignal(v)
    else:
        print ('WARNING: skipping signal fetch  !')
        v['current_df'] = v['base_path'] + v['file_name'] + '_signal.csv'

    # Quality check
    if quality_check:
        v = memotrack.run_QualityCheck(v)
    else:
        print ('WARNING: skipping quality check  !')
        v['current_df'] = v['base_path'] + v['file_name'] + '_qcheck.csv'

    # Filter signal
    if filtering:
        v = memotrack.run_filter(v)
    else:
        print ('WARNING: skipping signal filtering  !')
        v['current_df'] = v['base_path'] + v['file_name'] + '_filtered.csv'

    # Normalize signal
    if normalize:
        v = memotrack.run_normalize(v)
    else:
        print ('WARNING: skipping signal normalization  !')
        v['current_df'] = v['base_path'] + v['file_name'] + '_normalized.csv'

    # Crosstalk check
    if crosscheck:
        v = memotrack.run_crosscheck(v)
    else:
        print ('WARNING: skipping crosstalk check  !')
        v['current_df'] = v['base_path'] + v['file_name'] + '_crosscheck.csv'

    # Generating analysis report
    if report:
        v = memotrack.run_report(v)
    else:
        print ('WARNING: skipping final report  !')
        v['current_df'] = v['base_path'] + v['file_name'] + '_report.csv'

    # Generate extra visualizations
    if extra:
        v = memotrack.run_visualizations(v)

    # Get final number of detections
    v = memotrack.nlabels(v)
    if verbose:
        print('[Done]')
        if v:
            if v['final_nlabels']:
                print ('\nAll done ! Process finished with ' + str(v['final_nlabels']) + ' detections')
                print ('')
                memotrack.show_vars(v)

    return v


def run_header(v):
    import time
    import memotrack
    import os
    import sys

    if v['verbose']:
        memotrack.version()
        print(time.ctime())
        print ('')

    if v['debug']:
        print ('*** Attention, debug mode enabled ! ***')
        print ('')
        sys.stdout.flush()

    # Split path
    v['base_path'], v['full_file_name'] = os.path.split(v['img_path'])
    v['base_path'] += os.path.sep  # Putting a slash on the end to create a full path
    v['file_name'] = v['full_file_name'][0:-4]  # Getting only the name of the file, without extension

    if v['verbose']:
        print ('Image path:\t\t\t' + str(v['img_path']))
        print ('FWHM for nuclei:\t\t' + str(v['FWHMxy']))
        print ('Signal FWHM increase factor:\t' + str(v['signalFWHM']))  # To fetch GCaMP signal
        print ('Iterations for detection:\t' + str(v['iterations']))
        print ('Working path:\t\t\t' + str(v['base_path']))
        print ('Working name:\t\t\t' + str(v['file_name']))

        sys.stdout.flush()

    # --- Reading metadata --- #
    if v['verbose']:
        print ('\nReading file metadata:')
    v['meta'] = memotrack.io.meta(v['img_path'], verbose=v['verbose'])

    # Setting variables for img size
    v['SizeT'] = v['meta']['SizeT']
    if v['debug']:
        v['SizeT'] = int(v['debug'])
        print ('\n*** Debug mode, using only ' + str(v['debug']) + ' frames ! ***')
        sys.stdout.flush()
    if v['truncate']:
        v['SizeT'] = int(v['truncate'])
        v['meta']['SizeT'] = int(v['truncate'])
        print ('WARNING: Truncating time frames, ending on frame {}'.format(v['truncate']))

    v['SizeZ'] = v['meta']['SizeZ']
    v['SizeY'] = v['meta']['SizeY']
    v['SizeX'] = v['meta']['SizeX']
    v['pixel_size'] = float(v['meta']['PhysicalSizeX'])
    if v['SizeT']:
        v['ref_frame'] = int(float(v['SizeT']) / 2)
    else:
        v['ref_frame'] = 0
        v['meta']['SizeT'] = 1

    if v['verbose']:
        print ('')
        sys.stdout.flush()

    return v


def run_detection(v):
    import numpy as np
    import scipy.ndimage
    import pandas as pd
    import sys
    import tifffile as tiff
    import os
    import scipy.spatial.distance
    import scipy.spatial
    import math
    import memotrack

    # Start dataframe
    detections_df = pd.DataFrame([], columns=['t', 'z', 'y', 'x'])

    # Setting variables for img size
    SizeT = v['meta']['SizeT']
    print ('SizeT: {}'.format(SizeT))
    SizeZ = v['meta']['SizeZ']
    SizeY = v['meta']['SizeY']
    SizeX = v['meta']['SizeX']
    SizeC = v['meta']['SizeC']
    PhysicalSizeX = float(v['meta']['PhysicalSizeX'])
    PhysicalSizeY = float(v['meta']['PhysicalSizeY'])
    PhysicalSizeZ = float(v['meta']['PhysicalSizeZ'])

    if PhysicalSizeX != PhysicalSizeY:
        print ('WARNING: Pixel dimensions are different on X and Y. Maybe something is wrong with the metadata ?')
        sys.stdout.flush()

    # Case pixel size is not supplied, use from metadata
    if v['pixel_size']:
        pixel_size = v['pixel_size']
    else:
        pixel_size = PhysicalSizeX

    # Initialize values
    sigmas_list = []
    num_detections_list = []

    # Load the full image
    full_img = memotrack.io.load_full_img(v['img_path'], verbose=True)
    print (np.shape(full_img))

    if v['truncate']:
        trunc_frame = int(v['truncate'])
        full_img = full_img[:trunc_frame]
        print ('Truncating image to frame {}'.format(trunc_frame))
        print ('New shape: {}'.format(np.shape(full_img)))
    # Generate list of sigma values
    target_size = (float(v['FWHMxy']) / pixel_size) / 2  # Converting diameter in microns to radius in pixels
    target_sigma = target_size / math.sqrt(
        2)  # * math.log(2))  # square root of 2*(ln(2)) for the equivalence with sigma

    scale_range = 0.5
    max_value = target_sigma + (target_sigma * scale_range)  # Maximum object size
    min_value = target_sigma - (target_sigma * scale_range)  # Minimum object size
    sigma_step = (max_value - min_value) / v['iterations']  # Sigma step for the desired number of iterations

    if v['verbose']:
        print ('\nStarting detection')
        print ('Target radius:\t' + str(target_size) + ' pixels')
        print ('Target sigma:\t' + str(target_sigma))
        print ('Min sigma:\t' + str(min_value))
        print ('Max sigma:\t' + str(max_value))
        print ('Sigma step:\t' + str(sigma_step))

    # runs through every frame of the image
    for ref_frame in range(SizeT):
        if v['verbose']:
            print ('\nFrame {}'.format(ref_frame))

        # Reading file. Default channel is zero
        # img = memotrack.io.read(v['img_path'], v['meta'], frame=ref_frame, channel=v['channel'], verbose=v['verbose'])

        # Mode for already loaded image (TZCYX)
        if len(np.shape(full_img)) == 3:
            img = full_img
        else:
            print ('- getting frame {} from {}'.format(ref_frame, np.shape(full_img)))

            img = full_img[ref_frame, :, 0, :, :]  # nuclei channel is zero

        # Convert to float
        img = np.float64(img)  # Converting to float, to be sure of precision level in next steps
        print ('\nImg max value: {}'.format(np.max(img)))
        if v['debug']:
            print ('Working on ' + str(img.dtype))
            sys.stdout.flush()

        # Interpolation. This step assures the isometric voxel
        if PhysicalSizeX != PhysicalSizeZ:  # Only do it if it's needed
            nslices = int((SizeZ * PhysicalSizeZ) / PhysicalSizeX)  # Number of slices we need for an isometric image
            if v['verbose']:
                print ('Interpolating ' + str(nslices) + ' slices from ' + str(SizeZ) + ' of the original image.'),
                sys.stdout.flush()
            zoom_ratio = float(nslices) / float(SizeZ)
            print ('\nImg max value: {}'.format(np.max(img)))
            img = scipy.ndimage.interpolation.zoom(np.float64(img), [zoom_ratio, 1, 1], order=v['spline_order'],
                                                   mode='nearest')
            print ('\nImg max value: {}'.format(np.max(img)))
            # Interpolation with order higher than 2 may cause negative values, fixing this here
            img[img < 0] = 0
            if v['verbose']:
                print ('[Done]')
                sys.stdout.flush()

            if v['save_interpolated']:
                save_path_interpolated = v['img_path'][:-4] + '_interpolated.tif'
                if v['debug']:
                    print ('Saving detections as raster to disk:')
                    print (str(save_path_interpolated)),
                    sys.stdout.flush()
                sys.stdout.flush()
                tiff.imsave(save_path_interpolated, np.float32(img))
                if v['debug']:
                    print ('[Done]')
                    sys.stdout.flush()

        # Add interpolated size to metadata
        v['meta']['SizeZinterpolated'] = np.shape(img)[0]

        # Counters and variables for this iteration
        total_iter = 0  # iteration counter
        accum_array = np.zeros(np.shape(img), dtype='float')  # initializing accumulator array
        max_intensity = np.max(img)
        min_intensity = np.min(img)
        avg_intensity = np.mean(img)
        med_intensity = np.median(img)
        if v['debug']:
            print ('max intensity: ' + str(max_intensity))
            print ('avg intensity: ' + str(avg_intensity))
            print ('med intensity: ' + str(med_intensity))
            print ('min intensity: ' + str(min_intensity))
            sys.stdout.flush()

        if v['debug']:
            print ('\nStarting filtered img for intensities...'),
            sys.stdout.flush()

        # Filtered image to fetch the intensities
        img_intensity_filter = scipy.ndimage.filters.gaussian_filter(img, sigma=target_sigma, mode='constant', cval=0)

        if v['debug']:
            print ('[Done]')
            sys.stdout.flush()

        # Iterate through all the scales (sigma values)
        for sigma in np.arange(min_value, max_value, sigma_step):
            total_iter += 1

            if v['verbose']:
                print ('\nScale #' + str(total_iter))
                print ('sigma ' + str(sigma))
                sys.stdout.flush()

            # Detection
            temp_df, num_detections, detections_array, img_filtered = memotrack.process.local_max(img, v['FWHMxy'],
                                                                                                  pixel_size=pixel_size,
                                                                                                  gauss_sigma=sigma,
                                                                                                  verbose=v[
                                                                                                      'verbose'],
                                                                                                  debug=v['debug'],
                                                                                                  save_raster=False,
                                                                                                  gauss_filter=True)

            temp_accum_array = detections_array * img_intensity_filter  # reliable but slow

            if v['debug']:
                print ('[Done]')
                sys.stdout.flush()

            # Filter temporary array
            # Here we use a GMM to separate values that should be only background noise
            if v['debug']:
                print ('Initializing GMM...'),
                sys.stdout.flush()
            temp_thresh = memotrack.process.GMM_thresh(temp_accum_array, debug=v['debug'], verbose=v['debug'])
            if v['debug']:
                print ('[Done]')
                sys.stdout.flush()
            if v['display']:
                # This histogram show the distribution of weights for this scale
                # Dark bars show values above the threshold
                memotrack.display.histogram(temp_accum_array[temp_accum_array > 0].flatten(), nbins=25, ylog=True,
                                            accent_after=temp_thresh[0], rotate_labels=True,
                                            title='Sigma: ' + str(sigma) + '   Thresh: ' + str(temp_thresh))

            # Values lower than the GMM threshold are removed
            if v['debug']:
                print ('Applying threshold...')
                sys.stdout.flush()

            if v['debug']:
                print ('min intensity before thresh: ' + str(np.min(temp_accum_array[np.nonzero(temp_accum_array)])))

            temp_accum_array[temp_accum_array < temp_thresh] = 0

            if v['debug']:
                print ('min value after thresh: ' + str(np.min(temp_accum_array[np.nonzero(temp_accum_array)])))

            if v['debug']:
                print ('[Done]')
                sys.stdout.flush()

            # Get number of detections after threshold
            foreground_ndetections = len(np.nonzero(np.ravel(temp_accum_array))[0])

            if v['verbose']:
                print (str(foreground_ndetections) + ' on foreground')
                sys.stdout.flush()

            # This scale is added to the global accumulator array
            accum_array += temp_accum_array

        # Save accumulator array to disk as raster image
        # This step is not needed for the rest of the process, It's only for later check
        if v['save_raster']:
            save_accum_path = v['base_path'] + 'accumulator.tif'
            print ('\nSaving accumulator array:')

            print (str(save_accum_path)),
            sys.stdout.flush()
            accum_img = accum_array.astype('float', copy=True)

            tiff.imsave(save_accum_path, np.float32(accum_img))
            print ('[Done]' + str(np.shape(accum_img)))

        # Create df from the iterations, for summary

        # Filtering the accumulator array, so that the final local maxima can be detected
        if v['debug']:
            print ('Filtering accumulator array...')
            sys.stdout.flush()
        array_blur = scipy.ndimage.filters.gaussian_filter(accum_array, sigma=target_sigma, mode='constant',
                                                           cval=0)
        if v['debug']:
            print ('[Done]')
            sys.stdout.flush()

        # Save filtered array
        # This step is not needed for the rest of the process, It's only for later check
        if v['save_raster']:
            save_accum_path = v['base_path'] + 'filtered.tif'
            print ('\nSaving filtered array:')
            sys.stdout.flush()
            print (str(save_accum_path)),
            sys.stdout.flush()
            tiff.imsave(save_accum_path, np.float32(array_blur))
            print ('[Done]' + str(np.shape(array_blur)))
            sys.stdout.flush()

        # Detect position on accumulator array
        if v['verbose']:
            print ('\nStarting detection on accumulator array')
            sys.stdout.flush()
        save_final_path = v['base_path'] + 'detections_final.tif'
        outputs = memotrack.process.local_max(array_blur, v['FWHMxy'], pixel_size=pixel_size,
                                              gauss_sigma=target_sigma, verbose=False, debug=v['debug'],
                                              save_raster=save_final_path, gauss_filter=False)

        # unpack outputs
        temp_df, num_detections, detections_array, img_filtered2 = outputs

        # Insert time
        temp_df['t'] = ref_frame

        # Concatenate temporary frame to the final
        detections_df = pd.concat([detections_df, temp_df])

        # Reset values for next image
        sigmas_list = []
        num_detections_list = []

        if v['verbose']:
            print ('Process finished with '),
            print (str(len(temp_df)) + ' detections')

            # Writing to disk the results from the detection
        if v['write_results']:
            if v['verbose']:
                print ('\nWriting dataframe to disk...'),
                sys.stdout.flush()
            detections_dataframe_path = v['base_path'] + v['file_name'] + '_detections.csv'
            detections_df.to_csv(detections_dataframe_path, index_label=False)
            if v['verbose']:
                print ('[Done]')
                print (detections_dataframe_path)
                sys.stdout.flush()

        v['current_df'] = v['base_path'] + v['file_name'] + '_detections.csv'

    # Deleting the big image, to release memory
    del full_img

    return v


def run_detection_old(v):
    import numpy as np
    import scipy.ndimage
    import pandas as pd
    import sys
    import tifffile as tiff
    import os
    import scipy.spatial.distance
    import scipy.spatial
    import math

    # Start dataframe
    detections_df = pd.DataFrame([], columns=['t', 'z', 'y', 'x'])

    # Setting variables for img size
    SizeT = v['meta']['SizeT']
    SizeZ = v['meta']['SizeZ']
    SizeY = v['meta']['SizeY']
    SizeX = v['meta']['SizeX']
    SizeC = v['meta']['SizeC']
    PhysicalSizeX = float(v['meta']['PhysicalSizeX'])
    PhysicalSizeY = float(v['meta']['PhysicalSizeY'])
    PhysicalSizeZ = float(v['meta']['PhysicalSizeZ'])

    if PhysicalSizeX != PhysicalSizeY:
        print ('WARNING: Pixel dimensions are different on X and Y. Maybe something is wrong with the metadata ?')
        sys.stdout.flush()

    # Case pixel size is not supplied, use from metadata
    if v['pixel_size']:
        pixel_size = v['pixel_size']
    else:
        pixel_size = PhysicalSizeX

    # Initialize values
    sigmas_list = []
    num_detections_list = []
    foreground_detections_list = []

    # runs through every frame of the image
    for ref_frame in range(SizeT):
        # Generate list of sigma values
        target_size = (float(v['FWHMxy']) / pixel_size) / 2  # Converting diameter in microns to radius in pixels
        target_sigma = target_size / math.sqrt(
            2 * math.log(2))  # square root of 2*(ln(2)) for the equivalence with sigma

        max_value = target_sigma + (target_sigma / 2)  # Maximum object size
        min_value = target_sigma - (target_sigma / 2)  # Minimum object size
        sigma_step = (max_value - min_value) / v['iterations']  # Sigma step for the desired number of iterations

        if v['verbose']:
            print ('Target radius:\t' + str(target_size) + ' pixels')
            print ('Target sigma:\t' + str(target_sigma))
            print ('Min sigma:\t' + str(min_value))
            print ('Max sigma:\t' + str(max_value))
            print ('Sigma step:\t' + str(sigma_step))

        # Reading file. Default channel is zero
        img = memotrack.io.read(v['img_path'], v['meta'], frame=ref_frame, channel=v['channel'],
                                verbose=v['verbose'])

        # Convert to float
        img = img.astype(float)  # Converting to float, to be sure of precision level in next steps
        if v['debug']:
            print ('Working on ' + str(img.dtype))
            sys.stdout.flush()

        # Interpolation. This step assures the isometric voxel
        if PhysicalSizeX != PhysicalSizeZ:  # Only do it if it's needed
            nslices = int((SizeZ * PhysicalSizeZ) / PhysicalSizeX)  # Number of slices we need for an isometric image
            if v['verbose']:
                print ('Interpolating ' + str(nslices) + ' slices from ' + str(SizeZ) + ' of the original image.'),
                sys.stdout.flush()
            zoom_ratio = float(nslices) / float(SizeZ)
            img = scipy.ndimage.interpolation.zoom(img, [zoom_ratio, 1, 1], order=v['spline_order'], mode='nearest')
            # Interpolation with order higher than 2 may cause negative values, fixing this here
            img[img < 0] = 0
            if v['verbose']:
                print ('[Done]')
                sys.stdout.flush()

            if v['save_interpolated']:
                save_path_interpolated = v['img_path'][:-4] + '_interpolated.tif'
                if v['debug']:
                    print ('Saving detections as raster to disk:')
                    print (str(save_path_interpolated)),
                    sys.stdout.flush()
                sys.stdout.flush()
                img_to_save = img.astype('uint16', copy=True)
                tiff.imsave(save_path_interpolated, np.float32(img_to_save))
                if v['debug']:
                    print ('[Done]')
                    sys.stdout.flush()

        # Add interpolated size to metadata
        v['meta']['SizeZinterpolated'] = np.shape(img)[0]

        # Counters and variables for this iteration
        total_iter = 0  # iteration counter
        accum_array = np.zeros(np.shape(img), dtype='float')  # initializing accumulator array
        max_intensity = np.max(img)
        min_intensity = np.min(img)
        avg_intensity = np.mean(img)
        med_intensity = np.median(img)
        if v['debug']:
            print ('max intensity: ' + str(max_intensity))
            print ('avg intensity: ' + str(avg_intensity))
            print ('med intensity: ' + str(med_intensity))
            print ('min intensity: ' + str(min_intensity))

        img_filtered = scipy.ndimage.filters.gaussian_filter(img, sigma=target_sigma, mode='constant', cval=0)
        foreground_mask = img_filtered >= 0
        mask_img = foreground_mask.astype('uint8', copy=True)

        # Iterate through all the scales (sigma values)
        for sigma in np.arange(min_value, max_value, sigma_step):
            total_iter += 1

            if v['verbose']:
                print ('\nIteration #' + str(total_iter))
                print ('sigma ' + str(sigma))
                sys.stdout.flush()

            # Detection
            temp_df, num_detections, detections_array, img_filtered = memotrack.process.local_max(img, v['FWHMxy'],
                                                                                                  pixel_size=pixel_size,
                                                                                                  gauss_sigma=sigma,
                                                                                                  verbose=v[
                                                                                                      'verbose'],
                                                                                                  debug=v['debug'],
                                                                                                  save_raster=False,
                                                                                                  gauss_filter=True,
                                                                                                  non_zero_mask=True,
                                                                                                  mask=mask_img,
                                                                                                  show_time=False)
            # Insert time
            temp_df['t'] = ref_frame

            # -- Weight calculation -- #
            # get w2 (based on intensity)
            positions = temp_df.as_matrix(columns=['z', 'y', 'x'])
            intensity_list = []

            point_counter = 0

            '''
            # Create the labeled array of the constrained voronoi regions
            voronoi_max_size = (v['FWHMxy'] / v['pixel_size'])
            labeled_array = memotrack.process.generate_voronoi(temp_df, v['meta'], max_distance=voronoi_max_size,
                                                               verbose=v['verbose'], debug=v['debug'])
            '''
            if v['debug']:
                print ('Generating quick labels...'),
                print (np.shape(detections_array))
                sys.stdout.flush()
            labeled_array, num_features = scipy.ndimage.label(detections_array)
            if v['debug']:
                print ('[Done]')
                print (np.shape(labeled_array))
                print ('num features: ' + str(num_features))
                sys.stdout.flush()

            if v['debug']:
                print ('Fetching signal from labels...'),
                sys.stdout.flush()
            intensity_list = scipy.ndimage.measurements.mean(img_filtered, labels=labeled_array,
                                                             index=range(num_detections))

            if v['debug']:
                print ('[Done]')
                sys.stdout.flush()

            '''
            # Here we define a sphere centered on the detection, to calculate the average value
            for label in range(num_detections):
                if v['debug']:
                    if point_counter >= 100:
                        print ('[' + str(int(float(label) / num_detections * 100)) + '%]'),
                        sys.stdout.flush()
                        point_counter = 0

                label_mask = labeled_array == label

                coords_list = np.where(label_mask)
                label_avg_intensity = np.mean(img[coords_list[0], coords_list[1], coords_list[2]])
                intensity_list.append(label_avg_intensity)
                point_counter += 1
            '''

            # normalizing intensity list
            # w2 is the average of the sphere, divided by the average of the image
            intensity_list /= avg_intensity
            temp_df['w2'] = intensity_list

            if v['debug']:
                print ('')
                print (temp_df.sort('w2'))
                sys.stdout.flush()

            # Append lists
            sigmas_list.append(sigma)
            num_detections_list.append(num_detections)

            # Generate accumulator array for this scale
            temp_accum_array = np.zeros(np.shape(img))
            for i in range(len(temp_df)):
                x = int(temp_df.loc[i, 'x'])
                y = int(temp_df.loc[i, 'y'])
                z = int(temp_df.loc[i, 'z'])
                w2 = temp_df.loc[i, 'w2']
                temp_accum_array[z, y, x] = w2  # Multiplication of both weights for the accumulator array

            # Filter temporary array
            # Here we use a GMM to separate values that should be only background noise
            if v['debug']:
                print ('Initializing GMM...'),
                sys.stdout.flush()
            temp_thresh = memotrack.process.GMM_thresh(temp_accum_array, debug=v['debug'], verbose=v['debug'])
            if v['debug']:
                print ('[Done]')
                sys.stdout.flush()
            if v['display']:
                # This histogram show the distribution of weights for this scale
                # Dark bars show values above the threshold
                memotrack.display.histogram(temp_accum_array[temp_accum_array > 0].flatten(), nbins=25, ylog=True,
                                            accent_after=temp_thresh[0][0], rotate_labels=True,
                                            title='Sigma: ' + str(sigma) + '   Thresh: ' + str(temp_thresh[0][0]))

            # Values lower than the GMM threshold are removed
            temp_accum_array[temp_accum_array < temp_thresh] = 0
            foreground_detections = np.sum(i > 0 for i in temp_accum_array.flatten())
            foreground_detections_list.append(foreground_detections)

            # This scale is added to the global accumulator array
            accum_array += temp_accum_array

        # Save accumulator array to disk as raster image
        # This step is not needed for the rest of the process, It's only for later check
        if v['save_raster']:
            save_accum_path = v['base_path'] + 'accumulator.tif'
            print ('\nSaving accumulator array:')
            print (str(save_accum_path)),
            sys.stdout.flush()
            accum_img = accum_array.astype('float', copy=True)
            tiff.imsave(save_accum_path, np.float32(accum_img))
            print ('[Done]' + str(np.shape(accum_img)))

        # Create df from the iterations, for summary
        '''
        if vars['verbose']:
            list_matrix = [sigmas_list, num_detections_list, foreground_detections_list]
            progress_df = pd.DataFrame(np.transpose(list_matrix),
                                       columns=['sigma', 'detections', 'only foreground'])
            print('\nScales summary')
            print(progress_df)
        '''

        # Filtering the accumulator array, so that the final local maxima can be detected
        array_blur = scipy.ndimage.filters.gaussian_filter(accum_array, sigma=target_sigma, mode='constant',
                                                           cval=0)
        # Save filtered array
        # This step is not needed for the rest of the process, It's only for later check
        if v['save_raster']:
            save_accum_path = v['base_path'] + 'filtered.tif'
            print ('\nSaving filtered array:')
            print (str(save_accum_path)),
            sys.stdout.flush()
            tiff.imsave(save_accum_path, np.float32(array_blur))
            print ('[Done]' + str(np.shape(array_blur)))

        # Detect position on accumulator array
        if v['verbose']:
            print ('\nStarting detection on accumulator array')
            sys.stdout.flush()
        save_final_path = v['base_path'] + 'detections_final.tif'
        outputs = memotrack.process.local_max(array_blur, v['FWHMxy'], pixel_size=pixel_size,
                                              gauss_sigma=target_sigma, verbose=False, debug=v['debug'],
                                              save_raster=save_final_path, gauss_filter=False, non_zero_mask=True,
                                              show_time=False, mask=True)

        # unpack outputs
        temp_df, num_detections, detections_array, img_filtered2 = outputs

        # Insert time
        temp_df['t'] = ref_frame

        # Concatenate temporary frame to the final
        detections_df = pd.concat([detections_df, temp_df])

        # Reset values for next image
        sigmas_list = []
        num_detections_list = []

        if v['min_verb']:
            print ('Process finished with '),
            print (str(len(temp_df)) + ' detections')

            # Writing to disk the results from the detection
        if v['write_results']:
            if v['verbose']:
                print ('\nWriting dataframe to disk...'),
                sys.stdout.flush()
            detections_dataframe_path = v['base_path'] + v['file_name'] + '_detections.csv'
            detections_df.to_csv(detections_dataframe_path, index_label=False)
            if v['verbose']:
                print ('[Done]')
                print (detections_dataframe_path)
                sys.stdout.flush()

    return v


def run_registration(v):
    import memotrack
    reg_step = 10
    # load dataframe
    df = memotrack.io.read_df(v['current_df'])

    # Affine registration method
    df, matrix_list, middle_frame = memotrack.registration.chained_registration(df, reg_step=reg_step,
                                                                                verbose=v['verbose'],
                                                                                display=v['display'],
                                                                                debug=v['debug'])

    # FOR TESTS ONLY
    # matrix_list = memotrack.registration.generate_test_matrix(df, matrix_list, middle_frame)

    df = memotrack.registration.align_frames(df, matrix_list, middle_frame, reg_step=reg_step,
                                             verbose=v['verbose'],
                                             display=v['display'], debug=v['debug'])

    # Here we set the affine registration with the temporary names.
    # This makes easier in case we want to change back to only affine
    df['xaff'] = df['xreg']
    df['yaff'] = df['yreg']
    df['zaff'] = df['zreg']

    # Starts the non-rigid step of the registration
    # Here we use the chained affine registration as starting point
    df = memotrack.registration.cpd(df, verbose=v['verbose'])

    # set new path
    new_path = v['base_path'] + v['file_name'] + '_registered.csv'

    # writing registered set to disk
    memotrack.io.write_df(df, new_path)
    v['current_df'] = new_path

    return v


def run_DBSCAN(v):
    import sys
    import memotrack
    import numpy as np

    # load dataframe
    df = memotrack.io.read_df(v['current_df'])
    df_analysis = memotrack.analyse.info_df(df)

    if v['verbose']:
        print ('\nStarting cluster stage...')
        print ('\nEstimation of eps for DBSCAN:')
        sys.stdout.flush()

    # eps estimation
    # estimated_eps = memotrack.process.estimate_DBSCAN_eps(df, verbose=v['verbose'], debug=False)
    estimated_eps = 2 * float(v['FWHMxy'])
    if v['verbose']:
        print ('\nEstimation of min_samples for DBSCAN:')
        sys.stdout.flush()

    # Estimates the value for min_samples, based on the average number of detections
    avg_detections = np.average(df_analysis['avg'])  # Average number of detections is the goal number of clusters
    estimated_min_samples = memotrack.process.estimate_DBSCAN_min_samples(df, estimated_eps,
                                                                          avg_detections, verbose=v['verbose'],
                                                                          display=v['display'])

    # Running DBSCAN with estimated parameters
    if v['verbose']:
        print ('\nInitializing final DBSCAN...')
        print ('eps: ' + str(estimated_eps))
        print ('min_samples: ' + str(estimated_min_samples))
        sys.stdout.flush()

    df, nlabels = memotrack.process.dbscan(df, eps=estimated_eps, min_samples=estimated_min_samples,
                                           FWHM=False, verbose=v['verbose'])

    v['nlabels'] = nlabels
    # Showing the dataframe with the labels
    if v['verbose']:
        print ('')
        print ('\nDetection dataframe after DBSCAN:')
        print ('')
        print (df)
        print ('')
        sys.stdout.flush()

    # Writing dataframe with labels to disk

    if v['verbose']:
        print ('\nWriting dataframe with labels to disk...'),
        sys.stdout.flush()
    label_dataframe_path = v['base_path'] + v['file_name'] + '_labels.csv'
    memotrack.io.write_df(df, label_dataframe_path)
    v['current_df'] = label_dataframe_path

    if v['verbose']:
        print ('[Done]')
        print (label_dataframe_path)
        sys.stdout.flush()

    if v['display']:
        # Raw clusters plots
        if v['verbose']:
            print ('\nGenerating plots...')
            sys.stdout.flush()

        # Generating colormap
        raw_clusters_cmap = memotrack.display.rand_cmap(nlabels, type='super_bright',
                                                        first_color_black=True, verbose=False)

        # All frames, original coordinates
        memotrack.display.plot_from_df(df, new_cmap=raw_clusters_cmap, size=2, elev=15, azim=30,
                                       crop_data=True, auto_fit=True, one_frame=False,
                                       frame=v['ref_frame'], time_color=False, intensity=False, borders=False,
                                       registered_coords=False,
                                       title=v['file_name'] + ' | Original detections before fix')

        # All frames, registered coordinates
        memotrack.display.plot_from_df(df, new_cmap=raw_clusters_cmap, size=2, elev=15, azim=30,
                                       crop_data=True, auto_fit=True, one_frame=False,
                                       frame=v['ref_frame'], time_color=False, intensity=False, borders=False,
                                       registered_coords=True,
                                       title=v['file_name'] + ' | Registered detections before fix')
        # Detection matrix
        memotrack.display.detection_matrix(df, color_scale=False, verbose=v['verbose'],
                                           title='Original detection matrix')

    return v


def run_quality(v):
    import sys
    import memotrack
    # load dataframe
    df = memotrack.io.read_df(v['current_df'])

    df = memotrack.process.check_quality(df, threshold=0.8, verbose=v['verbose'], visualization=False)

    # Writing fixed dataframe to disk
    if v['verbose']:
        print ('\nWriting dataframe with detection quality to disk...'),
        sys.stdout.flush()
    quality_path = v['base_path'] + v['file_name'] + '_quality.csv'
    memotrack.io.write_df(df, quality_path)
    v['current_df'] = quality_path
    if v['verbose']:
        print ('[Done]')
        print (quality_path)
        sys.stdout.flush()

    v['final_nlabels'] = df['label'].nunique()

    return v


def run_ClusterFix(v):
    import sys
    import memotrack

    # load dataframe
    df = memotrack.io.read_df(v['current_df'])

    print ('\nBeginning of cluster fix')
    print (df)
    # start remaining duplicates
    remaining_duplicates = 1

    # Reseting index of df. Duplicated index problem !
    del (df['level_0'])
    df = df.reset_index()

    if v['verbose']:
        print ('\nChecking and fixing clusters...')
        sys.stdout.flush()
    if v['debug']:
        print ('\ndf before fixing duplicates:')
        print (df)
        sys.stdout.flush()

    # Run while there are duplicates
    counter = 0

    while remaining_duplicates > 0:
        if counter < 25:
            df, remaining_duplicates = memotrack.process.handle_duplicates(df, verbose=v['verbose'], debug=False)
        else:
            remaining_duplicates = 0
        counter += 1

    if v['display']:
        if v['verbose']:
            print ('\nPlotting detection matrix...')
            sys.stdout.flush()
            # Plot fixed detection matrix

        memotrack.display.detection_matrix(df, color_scale=False, verbose=v['verbose'],
                                           title='Detection matrix with fixed clusters')

    # Filling detection gaps
    if v['verbose']:
        print ('\nInterpolating positions on detection gaps...')
        sys.stdout.flush()

    df = memotrack.process.fill_gaps(df)

    # Quickfix for strange column
    del (df['level_0'])

    # Showing the dataframe with the fixed clusters
    if v['debug']:
        print ('')
        print ('\nDetection dataframe with fixed clusters:')
        print ('')
        print (df)
        print ('')
        sys.stdout.flush()

    if v['display']:
        if v['verbose']:
            print ('\nGenerating plots...')
            sys.stdout.flush()

        # Detection matrix
        memotrack.display.detection_matrix(df, color_scale=False, verbose=v['verbose'],
                                           title='Detection matrix with interpolated missing positions')

        # Generating colormap
        final_clusters_cmap = memotrack.display.rand_cmap(v['nlabels'], type='super_bright',
                                                          first_color_black=False, verbose=False)
        v['final_clusters_cmap'] = final_clusters_cmap

        # All frames, original coords
        memotrack.display.plot_from_df(df, new_cmap=final_clusters_cmap, size=2, elev=15, azim=30,
                                       crop_data=True, auto_fit=True, one_frame=False,
                                       frame=v['ref_frame'], time_color=False, intensity=False, borders=False,
                                       registered_coords=False,
                                       title=v['file_name'] + ' | Original detections after fix')

        # All frames, registered coords
        memotrack.display.plot_from_df(df, new_cmap=final_clusters_cmap, size=2, elev=15, azim=30,
                                       crop_data=True, auto_fit=True, one_frame=False,
                                       frame=v['ref_frame'], time_color=False, intensity=False, borders=False,
                                       registered_coords=True,
                                       title=v['file_name'] + ' | Registered detections after fix')

    # --- Smoothing coordinates before signal check --- #
    # here we have the clustered dataset. Before getting the signal from the GCaMP channel,
    # we can try to smooth the xyz coordinates. Remember that here we smooth the original coordinates,
    # not the registered ones, because the signal is fetched from the original coords
    if v['verbose']:
        print ('\nSmoothing tracks...'),
        sys.stdout.flush()

    df = memotrack.process.smooth_track(df, sigma=0.5, verbose=v['debug'], debug=False)

    if v['verbose']:
        print ('[Done]')
        sys.stdout.flush()

    if v['debug']:
        print ('\ndf after smoothing')
        print (df)
        sys.stdout.flush()

    # We don't need the Ws anymore, they were used only during registration
    del df['w']
    del df['wreg']

    # Set labels as int, They were float for no reason
    df.label = df.label.astype(int)

    # Writing fixed dataframe to disk
    if v['verbose']:
        print ('\nWriting dataframe with fixed clusters to disk...'),
        sys.stdout.flush()
    fixed_clusters_path = v['base_path'] + v['file_name'] + '_fixed.csv'
    memotrack.io.write_df(df, fixed_clusters_path)
    v['current_df'] = fixed_clusters_path
    if v['verbose']:
        print ('[Done]')
        print (fixed_clusters_path)
        sys.stdout.flush()

    v['final_nlabels'] = df['label'].nunique()
    return v


def run_BackProjection(v):
    import sys
    import memotrack
    import numpy as np
    import pandas as pd
    import sklearn.neighbors
    import scipy.spatial
    import itertools

    verbose = v['verbose']
    # load dataframe
    df = memotrack.io.read_df(v['current_df'])

    print ('\nStarting back-projection...')
    sys.stdout.flush()

    # Making backup of original coords
    df['xorig'] = df['x']
    df['yorig'] = df['y']
    df['zorig'] = df['z']

    # Coords for reporjection
    df['xreproject'] = df['x']
    df['yreproject'] = df['y']
    df['zreproject'] = df['z']

    # Get number of labels and frames
    nlabels = df.label.nunique()
    nframes = df.t.nunique()
    # nlabels = 10
    # nframes = 3

    print ('{} frames with {} labels'.format(nframes, nlabels))
    sys.stdout.flush()

    # Calculate centroids. They will be used for the
    centroidX = df.groupby('label').xreg.mean()
    centroidY = df.groupby('label').yreg.mean()
    centroidZ = df.groupby('label').zreg.mean()

    # Here the index position of the array corresponds to the label of the centroid
    centroid_coords = (centroidX, centroidY, centroidZ)

    # Calculate the centroid KDTree.
    centroid_tree = sklearn.neighbors.KDTree(np.transpose(centroid_coords))

    for t in range(nframes):
        print ('\n\n==== Frame {} ===='.format(t))

        # Get only the dataframe for this time frame
        temp_df = df[df['t'] == t].copy(deep=True)

        ''''
        # Using smooth coordiantes
        xs = temp_df.xsmooth.get_values()
        ys = temp_df.ysmooth.get_values()
        zs = temp_df.zsmooth.get_values()
        coords = (xs, ys, zs)  # These are the original coordinates, and index is the label
        '''''

        # Using real coordianates
        xs = temp_df.x.get_values()
        ys = temp_df.y.get_values()
        zs = temp_df.z.get_values()
        coords = (xs, ys, zs)  # These are the original coordinates, and index is the label

        delta_change = [0, 0, 0]
        for label in range(nlabels):
            # Take the coordinates of the centroid for this label
            label_centroid_coord = np.transpose(centroid_coords)[label]

            # labels_ref are the closest labels to the point we're going to fix
            dist, labels_ref = centroid_tree.query(label_centroid_coord.reshape(1, -1), k=100)

            # Remove itself from list. (the closest)
            labels_ref = labels_ref[0][1:]
            # print (labels_ref)

            # An simplex with small volume is almost coplanar, we need to check this
            simplex_volume = 0
            idx = 0
            coords_range = 8
            best_coords = []
            best_volume = 0
            best_vertices_centroids = []
            start_distance = 2

            for neighbours in itertools.combinations(range(start_distance, start_distance + coords_range), 4):
                # print (neighbours)

                # Take the coordinates of the reference labels
                c1 = np.transpose(centroid_coords)[labels_ref[neighbours[0]]]
                c2 = np.transpose(centroid_coords)[labels_ref[neighbours[1]]]
                c3 = np.transpose(centroid_coords)[labels_ref[neighbours[2]]]
                c4 = np.transpose(centroid_coords)[labels_ref[neighbours[3]]]
                vertices_centroids = [c1, c2, c3, c4]

                simplex_volume = memotrack.process.tetrahedron_volume(vertices=np.asarray(vertices_centroids))

                if simplex_volume > best_volume:
                    best_volume = simplex_volume
                    best_coords = [neighbours[0], neighbours[1], neighbours[2], neighbours[3]]
                    best_vertices_centroids = vertices_centroids

            # Calculate the barycentric weights
            b_weights = memotrack.process.barycentric_coords(best_vertices_centroids, label_centroid_coord)

            # Now we get the coordinates of the reference labels on the original space
            r1 = np.transpose(coords)[labels_ref[best_coords[0]]]
            r2 = np.transpose(coords)[labels_ref[best_coords[1]]]
            r3 = np.transpose(coords)[labels_ref[best_coords[2]]]
            r4 = np.transpose(coords)[labels_ref[best_coords[3]]]
            vertices = [r1, r2, r3, r4]

            adjusted_coords = np.dot(np.transpose(vertices), b_weights)
            original_coords = np.transpose(coords)[label]

            # print (original_coords, adjusted_coords)

            # Update coordinates
            df.loc[
                (df['label'] == label) & (df['t'] == t), ['xreproject', 'yreproject', 'zreproject']] = adjusted_coords

            # Lazy check
            deltaX = abs(original_coords[0] - adjusted_coords[0])
            deltaY = abs(original_coords[1] - adjusted_coords[1])
            deltaZ = abs(original_coords[2] - adjusted_coords[2])

            if verbose > 1:
                print ('')
                print ('Label:      \t{}'.format(label))
                print ('Best coords:\t{}'.format(best_coords))
                print ('Volume:     \t{}'.format(best_volume))
                print ('Distances:  \t{}'.format(dist[0][best_coords]))
                print ('Coords Orig:\t{0:.2f}\t{1:.2f}\t{2:.2f}'.format(original_coords[0], original_coords[1],
                                                                        original_coords[2]))
                print ('Coords Back:\t{0:.2f}\t{1:.2f}\t{2:.2f}'.format(adjusted_coords[0], adjusted_coords[1],
                                                                        adjusted_coords[2]))
                sys.stdout.flush()
            else:
                if verbose:
                    print ('.'),

            thresh = 50
            if deltaX > thresh or deltaY > thresh or deltaZ > thresh:
                print ('\n\n**** WARNING: High variation after backprojection ! ****')
                print ('Label:      \t{}'.format(label))
                print ('Best coords:\t{}'.format(best_coords))
                print ('Volume:     \t{}'.format(best_volume))
                print ('Distances:  \t{}'.format(dist[0][best_coords]))
                print ('Coords Orig:\t{0:.2f}\t{1:.2f}\t{2:.2f}'.format(original_coords[0], original_coords[1],
                                                                        original_coords[2]))
                print ('Coords Back:\t{0:.2f}\t{1:.2f}\t{2:.2f}'.format(adjusted_coords[0], adjusted_coords[1],
                                                                        adjusted_coords[2]))
                print ('')

            delta_change[0] = delta_change[0] + deltaX
            delta_change[1] = delta_change[1] + deltaY
            delta_change[2] = delta_change[2] + deltaZ

        # print ('\nx: {0:.2f}\ty: {1:.2f}\tz: {2:.2f}'.format(delta_change[0], delta_change[1], delta_change[2]))
        sys.stdout.flush()

    print ('[Done]')
    sys.stdout.flush()

    # Writing dataframe to disk
    if v['verbose']:
        print ('\nWriting dataframe with back projection to disk...'),
        sys.stdout.flush()
    reprojected_path = v['base_path'] + v['file_name'] + '_reprojected.csv'
    memotrack.io.write_df(df, reprojected_path)
    v['current_df'] = reprojected_path
    if v['verbose']:
        print ('[Done]')
        print (reprojected_path)
        sys.stdout.flush()

    v['final_nlabels'] = df['label'].nunique()
    return v


def run_GetSignal(v):
    import sys
    import memotrack
    import tifffile as tiff

    # load dataframe
    df = memotrack.io.read_df(v['current_df'])
    if v['debug']:
        print ('current_df: ' + v['current_df'])

    if v['verbose']:
        print ('\nReading signal from detected neurons...')
        sys.stdout.flush()

    df = memotrack.process.get_signal_voronoi(df, v['img_path'], FWHMxy=v['FWHMxy'], signalFWHM=v['signalFWHM'],
                                              verbose=v['verbose'], debug=v['debug'], truncate=v['truncate'])

    '''
    df = memotrack.process.get_signal_old(df, v['img_path'], FWHMxy=v['FWHMxy'] * v['signalFWHM'], verbose=v['verbose'],
                                          debug=v['debug'])
    '''
    # Writing dataframe with signal to disk
    if v['verbose']:
        print ('\nWriting dataframe with signal to disk...'),
        sys.stdout.flush()
    fetched_signal_path = v['base_path'] + v['file_name'] + '_signal.csv'
    memotrack.io.write_df(df, fetched_signal_path)
    if v['verbose']:
        print ('[Done]')
        print (fetched_signal_path)
        sys.stdout.flush()

    v['current_df'] = fetched_signal_path

    return v


def run_QualityCheck(v):
    import sys
    import memotrack
    import numpy as np

    if v['verbose']:
        print ('\nApplying quality filter...'),
        sys.stdout.flush()

    # load dataframe
    df = memotrack.io.read_df(v['current_df'])
    if v['debug']:
        print ('current_df: ' + v['current_df'])

    qthresh = 0.95

    # Q interpolation bug fix
    nframes = df.t.nunique()
    for frame in range(nframes):
        thresh = df[df['t'] == frame].Q.median()
        df.loc[df['t'] == frame, 'Q'] = thresh
        if thresh < qthresh:
            if v['verbose']:
                print ('Frame ' + str(frame) + ' under quality threshold')

    # Save raw intensities in case needed for the future
    df['raw_intensity'] = df['intensity']

    # Interpolate when quality is smaller than threshold
    print ('NaNs before Qthresh: '),
    print (df.intensity.isnull().any().any())

    df.loc[df['Q'] < qthresh, 'intensity'] = np.nan

    print ('NaNs after Qthresh: '),
    print (df.intensity.isnull().any().any())

    nlabels = df.label.nunique()

    for label in range(nlabels):
        # print (label),
        temp_df = df[df['label'] == label].copy(deep=True)
        # print(temp_df.intensity.isnull().sum()),

        # Do interpolation
        temp_df.interpolate(method='linear', inplace=True)

        # print(temp_df.intensity.isnull().sum())

        intensity_list = temp_df.intensity.get_values()

        # Here we have a pandas bug in case the first value was NaN; Pandas doesnt interpolate backwards
        # Also, a case were the value was zero was causing infinite value during normalization
        intensity_median = np.nanmedian(intensity_list)
        for i in range(len(intensity_list)):
            if np.isnan(intensity_list[i]) or intensity_list[i] == 0:
                intensity_list[i] = intensity_median

        df.loc[df.label == label, 'intensity'] = intensity_list

    if v['verbose']:
        print ('[Done]')
        sys.stdout.flush()

    # Writing dataframe with signal to disk
    if v['verbose']:
        print ('\nWriting dataframe with quality check to disk...'),
        sys.stdout.flush()
    qcheck_path = v['base_path'] + v['file_name'] + '_qcheck.csv'
    memotrack.io.write_df(df, qcheck_path)
    if v['verbose']:
        print ('[Done]')
        print (qcheck_path)
        sys.stdout.flush()

    v['current_df'] = qcheck_path

    return v


def run_filter(v):
    import sys
    import memotrack

    # load dataframe
    df = memotrack.io.read_df(v['current_df'])

    if v['verbose']:
        print ('\nInitializing bandpass filtering')
    detections_df = memotrack.process.apply_filter(df, lowcut=0.0, highcut=0.2, verbose=v['verbose'],
                                                   thresh=v['filter_std_thresh'])

    # Writing dataframe with normalized signal to disk
    if v['verbose']:
        print ('\nWriting dataframe with filtered signal to disk...'),
        sys.stdout.flush()
    filtered_signal_path = v['base_path'] + v['file_name'] + '_filtered.csv'
    memotrack.io.write_df(detections_df, filtered_signal_path)
    v['current_df'] = filtered_signal_path

    if v['verbose']:
        print ('[Done]')
        print (filtered_signal_path)
        sys.stdout.flush()

    # Showing the dataframe with the signal intensity
    if v['debug']:
        print ('')
        print ('\nDetection dataframe with average and normalized signal intensity:')
        print ('')
        print (detections_df)
        print ('')
        sys.stdout.flush()

    # Plotting Signal lines
    if v['display']:

        if v['verbose']:
            print ('\nPlotting signal from detections...'),
            sys.stdout.flush()

        memotrack.display.plot_1D_signals(detections_df, normalize=False, accent=False,
                                          stim_frame=[25], stim_duration=10,
                                          cmap=v['final_clusters_cmap'], title=v['file_name'])

        memotrack.display.plot_1D_signals(detections_df, normalize='df', accent=False,
                                          stim_frame=[25], stim_duration=10,
                                          cmap=v['final_clusters_cmap'], title=v['file_name'])

        memotrack.display.plot_1D_signals(detections_df, normalize='filtered', accent=False,
                                          stim_frame=[25], stim_duration=10, only_responsive=True,
                                          cmap=v['final_clusters_cmap'], title=v['file_name'])

        memotrack.display.signal_matrix(detections_df, normalized=True, title='Normalized signal matrix')

        memotrack.display.plot_from_df(detections_df, new_cmap=v['final_clusters_cmap'], size=200, elev=15, azim=30,
                                       crop_data=False, auto_fit=True, one_frame=False,
                                       frame=0, time_color=False, intensity=True, borders=False,
                                       registered_coords=False, title='Normalized signal')

        if v['verbose']:
            print ('[Done]')
            sys.stdout.flush()

    return v


def run_normalize(v):
    import sys
    import memotrack

    # load dataframe
    df = memotrack.io.read_df(v['current_df'])

    detections_df = memotrack.process.intensity_normalization_deltaf(df, verbose=v['verbose'], debug=v['debug'])

    if v['verbose']:
        print ('\nWriting dataframe with normalized signal to disk...'),
        sys.stdout.flush()
    normalized_signal_path = v['base_path'] + v['file_name'] + '_normalized.csv'
    memotrack.io.write_df(detections_df, normalized_signal_path)
    if v['verbose']:
        print ('[Done]')
        print (normalized_signal_path)
        sys.stdout.flush()

    v['current_df'] = normalized_signal_path

    return v


def run_crosscheck(v):
    import sys
    import memotrack

    # load dataframe
    df = memotrack.io.read_df(v['current_df'])

    detections_df = memotrack.process.crosscheck(df, RT=v['RT'], diameter=v['FWHMxy'], pixel_size=v['meta']['PhysicalSizeX'],
                                                 verbose=v['verbose'], debug=v['debug'])

    if v['verbose']:
        print ('\nWriting dataframe with crosstalk check to disk...'),
        sys.stdout.flush()
    crosscheck_path = v['base_path'] + v['file_name'] + '_crosscheck.csv'
    memotrack.io.write_df(detections_df, crosscheck_path)
    if v['verbose']:
        print ('[Done]')
        print (crosscheck_path)
        sys.stdout.flush()

    v['current_df'] = crosscheck_path

    return v


def run_report(v):
    import memotrack
    v['trial_size'] = 40
    v['stim_delay'] = 10
    v['stim_duration'] = 5
    memotrack.analyse.report(v['current_df'], trial_size=v['trial_size'], stim_delay=v['stim_delay'],
                             stim_duration=v['stim_duration'], condor=v['condor'])
    return v


def run_visualizations(v):
    import numpy as np
    import memotrack
    import scipy.ndimage
    import matplotlib.pyplot as plt
    import sys
    import os

    df_path = v['current_df']
    file_path = v['img_path']
    meta = v['meta']

    full_img = memotrack.io.load_full_img(file_path, verbose=True)

    # Starting projections
    nframes = np.shape(full_img)[0]
    nslices = np.shape(full_img)[1]
    nchannels = np.shape(full_img)[2]
    ysize = np.shape(full_img)[3]
    xsize = np.shape(full_img)[4]

    XYmax_img = np.zeros((nframes, nchannels, ysize, xsize))
    ZYmax_img = np.zeros((nframes, nchannels, nslices, xsize))
    ZXmax_img = np.zeros((nframes, nchannels, ysize, nslices))

    nslices_interpolated = int((meta['SizeZ'] * meta['PhysicalSizeZ']) / meta['PhysicalSizeX'])
    zoom_ratio = float(nslices_interpolated) / float(meta['SizeZ'])

    # Z Projection
    for c in range(nchannels):
        print ('\nChannel {}'.format(c))
        for mframe in range(nframes):
            print ('.'),
            for mslice in range(nslices):
                inds = full_img[mframe][mslice][c] > XYmax_img[mframe][c]
                XYmax_img[mframe][c][inds] = full_img[mframe][mslice][c][inds]

    print ('\n[Done]')

    # Y projection
    swap_img = np.swapaxes(full_img, 1, 3)
    print (np.shape(swap_img))
    for c in range(nchannels):
        print ('\nChannel {}'.format(c))
        for mframe in range(nframes):
            print ('.'),
            for mY in range(ysize):
                inds = swap_img[mframe][mY][c] > ZYmax_img[mframe][c]
                ZYmax_img[mframe][c][inds] = swap_img[mframe][mY][c][inds]

    # Interpolation
    ZYmax_img = scipy.ndimage.interpolation.zoom(ZYmax_img, [1, 1, zoom_ratio, 1], order=1)
    print ('\n[Done]')

    # X Projection
    swap_img = np.swapaxes(full_img, 1, 4)
    print (np.shape(swap_img))
    for c in range(nchannels):
        print ('\nChannel {}'.format(c))
        for mframe in range(nframes):
            print ('.'),
            for mX in range(xsize):
                inds = swap_img[mframe][mX][c] > ZXmax_img[mframe][c]
                ZXmax_img[mframe][c][inds] = swap_img[mframe][mX][c][inds]

    # Interpolation
    ZXmax_img = scipy.ndimage.interpolation.zoom(ZXmax_img, [1, 1, 1, zoom_ratio], order=1)

    print ('\n[Done]')

    # Save movie to disk
    print ('\nStarting movie...')
    print (file_path)
    sys.stdout.flush()

    base_path, file_name = os.path.split(file_path)
    base_path2, stim = os.path.split(base_path)
    base_path3, fly = os.path.split(base_path2)

    save_path = '/projects/memotrack/temp/report/' + fly + '/projections/'

    if os.path.isdir(save_path) is False:
        os.mkdir(save_path)

    detections_df = memotrack.io.read_df(file_path[:-4] + '_normalized.csv')
    nlabels = int(detections_df['label'].nunique())
    cmap = memotrack.display.rand_cmap(nlabels, type='super_bright', first_color_black=False, verbose=True)

    for frame in range(nframes):
        fig = memotrack.display.XYZprojections_and_signal(XYmax_img, ZYmax_img, ZXmax_img,
                                                          detections_df, cmap, frame=frame, path=file_path,
                                                          peak_threshold=0.1, verbose=False)

        fig.savefig(save_path + 'OnlyResponsive_' + str(frame).zfill(3) + '.png')

        fig = memotrack.display.XYZprojections_and_signal(XYmax_img, ZYmax_img, ZXmax_img,
                                                          detections_df, cmap, frame=frame, path=file_path,
                                                          peak_threshold=0.0, verbose=False)

        fig.savefig(save_path + 'AllNeurons_' + str(frame).zfill(3) + '.png')

        # Please close
        fig.clear()
        plt.close('all')
        del (fig)
        plt.clf()

        print ('[{}]'.format(frame)),
        sys.stdout.flush()

    print ('\n[Done]')


def nlabels(v):
    import memotrack
    # Load dataframe

    while True:
        try:
            df = memotrack.io.read_df(v['current_df'])

            # Count unique labels
            v['final_nlabels'] = df['label'].nunique()

            return v
        except IOError:
            print ('Report dataframe not found !')

            return


def run_detection_w1w2(img_path, FWHMxy=1.21, iterations=10, channel=0, verbose=True, debug=False, min_verb=True,
                       pixel_size=False, spline_order=3, display=False, save_raster=False, write_results=True,
                       save_interpolated=True):
    """
    Starts the object detection for IterativeMax
    :param img_path: path for tiff image containing the spots to be detected
    :param FWHMxy: Average diamenter of the desired object, in microns
    :param iterations: Number of scales. Default is 10
    :param channel: channel within the image were the objects are
    :param verbose: True or False, toggles the output during detection
    :param debug: Shows more detailed information. If is given T, it limits the process to the first T time frames
    :param min_verb: True or False, for the one line output "Process finished with n detections"
    :param pixel_size: Override the pixel size contained on the metadata
    :param spline_order: Spline order for axial interpolation. Default is 3
    :param display: True or False, to plot histograms of detections for each scale
    :param save_raster: If True, saves the accumulator array as Tiff, on the same folder as the original image
    :param write_results: save results to disk in a csv file, on the same folder as the image
    :return: Pandas dataframe, containing the 3D coordinates of the detections
    """
    import numpy as np
    import scipy.ndimage
    import pandas as pd
    import sys
    import tifffile as tiff
    import os
    import scipy.spatial.distance
    import scipy.spatial
    import math

    # Split path
    base_path, full_file_name = os.path.split(img_path)
    base_path += os.sep  # Putting a slash on the end to create a full path
    file_name = full_file_name[0:-4]  # Getting only the name of the file, without extension

    if verbose:
        print ('Starting detection...')
        print ('Image path:\t\t' + str(img_path))
        print ('FWHMxy of object:\t' + str(FWHMxy))
        print ('Number of iterations:\t' + str(iterations))
        sys.stdout.flush()

    # Start dataframe
    detections_df = pd.DataFrame([], columns=['t', 'z', 'y', 'x'])

    # --- Reading metadata --- #
    if verbose:
        print ('\nReading file metadata:')
        sys.stdout.flush()
    meta = memotrack.io.meta(img_path, verbose=verbose)

    # Setting variables for img size
    SizeT = meta['SizeT']
    if debug:
        print ('\n*** Debug mode, using only ' + str(debug) + ' frames ! ***')
        sys.stdout.flush()
        SizeT = int(debug)
    SizeZ = meta['SizeZ']
    SizeY = meta['SizeY']
    SizeX = meta['SizeX']
    SizeC = meta['SizeC']
    PhysicalSizeX = float(meta['PhysicalSizeX'])
    PhysicalSizeY = float(meta['PhysicalSizeY'])
    PhysicalSizeZ = float(meta['PhysicalSizeZ'])

    if PhysicalSizeX != PhysicalSizeY:
        print ('WARNING: Pixel dimensions are different on X and Y. Maybe something is wrong with the metadata ?')
        sys.stdout.flush()

    # Case pixel size is not supplied, use from metadata
    if pixel_size is False:
        pixel_size = PhysicalSizeX

    # Initialize values
    sigmas_list = []
    num_detections_list = []
    foreground_detections_list = []

    # runs through every frame of the image
    for ref_frame in range(SizeT):
        # Generate list of sigma values
        target_size = (float(FWHMxy) / pixel_size) / 2  # Converting diameter in microns to radius in pixels
        target_sigma = target_size / math.sqrt(2)  # square root of two for the equivalence between size and sigma

        max_value = target_sigma + (target_sigma / 2)  # Maximum object size
        min_value = target_sigma - (target_sigma / 2)  # Minimum object size
        sigma_step = (max_value - min_value) / iterations  # Sigma step for the desired number of iterations

        if verbose:
            print ('Target radius:\t' + str(target_size) + ' pixels')
            print ('Target sigma:\t' + str(target_sigma))
            print ('Min sigma:\t' + str(min_value))
            print ('Max sigma:\t' + str(max_value))
            print ('Sigma step:\t' + str(sigma_step))

        # Reading file. Default channel is zero
        img = memotrack.io.read(img_path, meta, frame=ref_frame, channel=channel, verbose=verbose)

        # Convert to float
        img = img.astype(float)  # Converting to float, to be sure of precision level in next steps
        if debug:
            print ('Working on ' + str(img.dtype))
            sys.stdout.flush()

        # Interpolation. This step assures the isometric voxel
        if PhysicalSizeX != PhysicalSizeZ:  # Only do it if it's needed
            nslices = int((SizeZ * PhysicalSizeZ) / PhysicalSizeX)  # Number of slices we need for an isometric image
            if verbose:
                print ('Interpolating ' + str(nslices) + ' slices from ' + str(SizeZ) + ' of the original image.'),
                sys.stdout.flush()
            zoom_ratio = float(nslices) / float(SizeZ)
            img = scipy.ndimage.interpolation.zoom(img, [zoom_ratio, 1, 1], order=spline_order, mode='nearest')
            # Interpolation with order higher than 2 may cause negative values, fixing this here
            img[img < 0] = 0
            if verbose:
                print ('[Done]')
                sys.stdout.flush()

            if save_interpolated:
                save_path_interpolated = img_path[:-4] + '_interpolated.tif'
                if debug:
                    print ('Saving detections as raster to disk:')
                    print (str(save_path_interpolated)),
                    sys.stdout.flush()
                sys.stdout.flush()
                img_to_save = img.astype('uint16', copy=True)
                tiff.imsave(save_path_interpolated, np.float32(img_to_save))
                if debug:
                    print ('[Done]')
                    sys.stdout.flush()

        # Add interpolated size to metadata
        meta['SizeZinterpolated'] = np.shape(img)[0]

        # Counters and variables for this iteration
        total_iter = 0  # iteration counter
        accum_array = np.zeros(np.shape(img), dtype='float')  # initializing accumulator array
        max_intensity = np.max(img)
        min_intensity = np.min(img)
        avg_intensity = np.mean(img)
        med_intensity = np.median(img)
        if debug:
            print ('max intensity: ' + str(max_intensity))
            print ('avg intensity: ' + str(avg_intensity))
            print ('med intensity: ' + str(med_intensity))
            print ('min intensity: ' + str(min_intensity))

        img_filtered = scipy.ndimage.filters.gaussian_filter(img, sigma=target_sigma, mode='constant', cval=0)
        foreground_mask = img_filtered >= 0
        mask_img = foreground_mask.astype('uint8', copy=True)

        # Iterate through all the scales (sigma values)
        for sigma in np.arange(min_value, max_value, sigma_step):
            total_iter += 1

            if verbose:
                print ('\nIteration #' + str(total_iter))
                print ('sigma ' + str(sigma))
                sys.stdout.flush()

            # Detection
            temp_df, num_detections, detections_array, img_filtered = memotrack.process.local_max(img, FWHMxy,
                                                                                                  pixel_size=pixel_size,
                                                                                                  gauss_sigma=sigma,
                                                                                                  verbose=verbose,
                                                                                                  debug=debug,
                                                                                                  save_raster=False,
                                                                                                  gauss_filter=True,
                                                                                                  non_zero_mask=True,
                                                                                                  mask=mask_img,
                                                                                                  show_time=False)
            # Insert time
            temp_df['t'] = ref_frame

            # -- Weight calculation -- #
            # Get w1 (based on distances)
            positions = temp_df.as_matrix(columns=['z', 'y', 'x'])
            tree = scipy.spatial.cKDTree(positions)  # distance tree between the detections
            dist = []

            # Look for the closest neighbour for every detection
            for i in range(len(positions)):
                dd, ii = tree.query(positions[i], k=2)
                dist.append(dd[1])

            if debug:
                dist_thresholded = dist < sigma
                dist_global_thresh = dist < target_sigma
                print ('\nw1 stats:')
                print ('closest mean dist: ' + str(np.average(dist)))
                print ('closest min dist: ' + str(np.min(dist)))
                print ('closest max dist: ' + str(np.max(dist)))
                print ('Points closer than current sigma: ' + str(np.sum(dist_thresholded)))
                print ('Points closer than target sigma:  ' + str(np.sum(dist_global_thresh)))

            # Calculate final weight
            # Values closer than the object size are adjusted between 0 and 1.
            wlist = []
            for i in dist:
                if i > target_size:
                    wlist.append(1)
                else:
                    wtemp = (float(i) / target_size)
                    wlist.append(wtemp)
                    # wlist.append(1)

            temp_df['w1'] = wlist

            # get w2 (based on intensity)
            positions = temp_df.as_matrix(columns=['z', 'y', 'x'])
            intensity_list = []

            point_counter = 0

            # Here we define a sphere centered on the detection, to calculate the average value
            for coords in positions:
                if debug:
                    if point_counter >= 100:
                        print ('.'),
                        sys.stdout.flush()
                        point_counter = 0
                # coordinates of current point
                z = int(coords[0])
                y = int(coords[1])
                x = int(coords[2])

                # Get the coordinates of the sphere
                coords_list = memotrack.process.coords_in_range([z, y, x], meta, FWHMxy, pixel_size=pixel_size)

                # Fetch intensities from the image
                coord_intensity_list = []
                for ZYXcoords in coords_list:
                    coord_intensity_list.append(img[ZYXcoords[0], ZYXcoords[1], ZYXcoords[2]])

                # Calculate avergage of the sphere
                intensity_list.append(np.average(coord_intensity_list))
                point_counter += 1

            # normalizing intensity list
            # w2 is the average of the sphere, divided by the average of the image
            intensity_list /= avg_intensity
            temp_df['w2'] = intensity_list

            if debug:
                print ('')
                print (temp_df.sort('w1'))

            # Append lists
            sigmas_list.append(sigma)
            num_detections_list.append(num_detections)

            # Generate accumulator array for this scale
            temp_accum_array = np.zeros(np.shape(img))
            for i in range(len(temp_df)):
                x = int(temp_df.loc[i, 'x'])
                y = int(temp_df.loc[i, 'y'])
                z = int(temp_df.loc[i, 'z'])
                w1 = temp_df.loc[i, 'w1']
                w2 = temp_df.loc[i, 'w2']
                temp_accum_array[z, y, x] = w1 * w2  # Multiplication of both weights for the accumulator array

            # Filter temporary array
            # Here we use a GMM to separate values that should be only background noise
            temp_thresh = memotrack.process.GMM_thresh(temp_accum_array)
            if display:
                # This histogram show the distribution of weights for this scale
                # Dark bars show values above the threshold
                memotrack.display.histogram(temp_accum_array[temp_accum_array > 0].flatten(), nbins=25, ylog=True,
                                            accent_after=temp_thresh[0][0], rotate_labels=True,
                                            title='Sigma: ' + str(sigma) + '   Thresh: ' + str(temp_thresh[0][0]))

            # Values lower than the GMM threshold are removed
            temp_accum_array[temp_accum_array < temp_thresh] = 0
            foreground_detections = np.sum(i > 0 for i in temp_accum_array.flatten())
            foreground_detections_list.append(foreground_detections)

            # This scale is added to the global accumulator array
            accum_array += temp_accum_array

        # Save accumulator array to disk as raster image
        # This step is not needed for the rest of the process, It's only for later check
        if save_raster:
            save_accum_path = base_path + 'accumulator.tif'
            print ('\nSaving accumulator array:')
            print (str(save_accum_path)),
            sys.stdout.flush()
            accum_img = accum_array.astype('float', copy=True)
            tiff.imsave(save_accum_path, np.float32(accum_img))
            print ('[Done]' + str(np.shape(accum_img)))

        # Create df from the iterations, for summary
        if verbose:
            list_matrix = [sigmas_list, num_detections_list, foreground_detections_list]
            progress_df = pd.DataFrame(np.transpose(list_matrix),
                                       columns=['sigma', 'detections', 'only foreground'])
            print('\nScales summary')
            print(progress_df)

        # Filtering the accumulator array, so that the final local maxima can be detected
        array_blur = scipy.ndimage.filters.gaussian_filter(accum_array, sigma=target_sigma, mode='constant',
                                                           cval=0)
        # Save filtered array
        # This step is not needed for the rest of the process, It's only for later check
        if save_raster:
            save_accum_path = base_path + 'filtered.tif'
            print ('\nSaving filtered array:')
            print (str(save_accum_path)),
            sys.stdout.flush()
            tiff.imsave(save_accum_path, np.float32(array_blur))
            print ('[Done]' + str(np.shape(array_blur)))

        # Detect position on accumulator array
        if verbose:
            print ('\nStarting detection on accumulator array')
            sys.stdout.flush()
        save_final_path = base_path + 'detections_final.tif'
        outputs = memotrack.process.local_max(array_blur, FWHMxy, pixel_size=pixel_size, gauss_sigma=target_sigma,
                                              verbose=False, debug=debug, save_raster=save_final_path,
                                              gauss_filter=False, non_zero_mask=True, show_time=False, mask=True)

        # unpack outputs
        temp_df, num_detections, detections_array, img_filtered2 = outputs

        # Insert time
        temp_df['t'] = ref_frame

        # Concatenate temporary frame to the final
        detections_df = pd.concat([detections_df, temp_df])

        # Reset values for next image
        sigmas_list = []
        num_detections_list = []

        if min_verb:
            print ('Process finished with '),
            print (str(len(temp_df)) + ' detections')

            # Writing to disk the results from the detection
        if write_results:
            if verbose:
                print ('\nWriting dataframe to disk...'),
                sys.stdout.flush()
            detections_dataframe_path = base_path + file_name + '_detections.csv'
            detections_df.to_csv(detections_dataframe_path, index_label=False)
            if verbose:
                print ('[Done]')
                print (detections_dataframe_path)
                sys.stdout.flush()

    return detections_df


def run_only_analysis(img_path, verbose=True, display=True, debug=False, FWHMxy=1.337, FWHMz=5.581, pixel_size=0.16125,
                      tolerance=0.01, initial_sigma=0, sigma_step=0.01, signalFWHM=1.1, write_results=True,
                      copy_to_home=False, home_path='/users/biocomp/delestro/tmp/', filter_thresh=0.2):
    import memotrack
    import time
    import numpy as np
    import os
    import sys
    import datetime
    import pandas as pd
    import javabridge
    import shutil

    # Start clock
    start_time = time.time()

    # Print memotrack version
    memotrack.version(verbose=verbose)

    # Time counters
    total_load_time = 0
    total_process_time = 0
    total_plot_time = 0
    total_DBSCAN_time = 0
    total_fixing_clusters_time = 0
    total_get_signal_time = 0

    # Split path
    base_path, full_file_name = os.path.split(img_path)
    base_path += '/'  # Putting a slash on the end to create a full path - Linux only !
    file_name = full_file_name[0:-4]  # Getting only the name of the file, without extension

    if copy_to_home:
        new_path = home_path + full_file_name
        print ('Copying file to ' + str(home_path)),
        sys.stdout.flush()
        shutil.copyfile(img_path, new_path)
        print ('[Done]')
        sys.stdout.flush()
        img_path = new_path

    if debug:
        print ('*** Attention, debug mode enabled ! ***')
        print ('')
        sys.stdout.flush()

    if verbose:
        print ('Image path:\t\t\t' + str(img_path))
        print ('FWHMxy for nuclei:\t\t' + str(FWHMxy))
        print ('FWHMz for nuclei:\t\t' + str(FWHMz))
        print ('Signal FWHM increase factor:\t' + str(signalFWHM))  # To fetch GCaMP signal
        print ('Error tolerance (alpha):\t' + str(tolerance))
        print ('Initial sigma for estimation:\t' + str(initial_sigma))
        print ('Sigma step for estimation:\t' + str(sigma_step))
        print ('Working path:\t\t\t' + str(base_path))
        print ('Working name:\t\t\t' + str(file_name))
        sys.stdout.flush()
    '''
    # Loading detections
    detections_df = memotrack.io.read_df(base_path + file_name + '_detections_labels_fixed_signal.csv')

    # --- Signal normalization and filtering --- #
    detections_df = memotrack.process.intensity_normalization_amount(detections_df, k=12, FWHM=1.33, pixel_size=0.16125,
                                                                     verbose=True, debug=False)
    # Writing dataframe with normalized signal to disk
    if write_results:
        if verbose:
            print ('\nWriting dataframe with normalized signal to disk...'),
            sys.stdout.flush()
        normalized_signal_path = base_path + file_name + '_detections_labels_fixed_signal_normalized.csv'
        memotrack.io.write_df(detections_df, normalized_signal_path)
        if verbose:
            print ('[Done]')
            print (normalized_signal_path)
            sys.stdout.flush()
    '''
    detections_df = memotrack.io.read_df(base_path + file_name +
                                         '_detections_labels_fixed_signal_normalized_filtered.csv')
    if verbose:
        print ('\nInitializing bandpass filtering')

    detections_df = memotrack.process.apply_filter(detections_df, lowcut=0.02, highcut=0.3333, verbose=verbose,
                                                   thresh=filter_thresh)

    # Writing dataframe with normalized signal to disk
    if write_results:
        if verbose:
            print ('\nWriting dataframe with filtered signal to disk...'),
            sys.stdout.flush()
        normalized_signal_path = base_path + file_name + '_detections_labels_fixed_signal_normalized_filtered.csv'
        memotrack.io.write_df(detections_df, normalized_signal_path)
        if verbose:
            print ('[Done]')
            print (normalized_signal_path)
            sys.stdout.flush()

    # Showing the dataframe with the signal intensity
    if verbose:
        print ('')
        print ('\nDetection dataframe with average and normalized signal intensity:')
        print ('')
        print (detections_df)
        print ('')
        sys.stdout.flush()

    # Plotting Signal lines
    if display:

        if verbose:
            print ('\nPlotting signal from detections...'),
            sys.stdout.flush()
        temp_timer_start = time.time()
        memotrack.display.plot_1D_signals(detections_df, normalize=False, accent=False,
                                          stim_frame=[25], stim_duration=10,
                                          cmap=final_clusters_cmap, title=file_name)

        memotrack.display.plot_1D_signals(detections_df, normalize='df', accent=False,
                                          stim_frame=[25], stim_duration=10,
                                          cmap=final_clusters_cmap, title=file_name)

        memotrack.display.plot_1D_signals(detections_df, normalize='filtered', accent=False,
                                          stim_frame=[25], stim_duration=10, only_responsive=True,
                                          cmap=final_clusters_cmap, title=file_name)

        memotrack.display.signal_matrix(detections_df, normalized=True, title='Normalized signal matrix')

        memotrack.display.plot_from_df(detections_df, new_cmap=final_clusters_cmap, size=200, elev=15, azim=30,
                                       crop_data=False, auto_fit=True, one_frame=False,
                                       frame=0, time_color=False, intensity=True, borders=False,
                                       registered_coords=False, title='Normalized signal')

        total_plot_time += (time.time() - temp_timer_start)
        if verbose:
            print ('[Done]')
            sys.stdout.flush()

    # --- Finalizing function --- #

    # Kill the JVM
    if verbose:
        print ('Killing JVM'),
        sys.stdout.flush()
    javabridge.kill_vm()
    if verbose:
        print('[Done]')
        sys.stdout.flush()

    # Getting final number of labels.
    final_nlabels = detections_df['label'].nunique()

    if verbose:
        print ('\nAll done ! Process finished with ' + str(final_nlabels) + ' detected neurons')
        print ('*** Attention: this was only the post-detection step !')
        sys.stdout.flush()

    return detections_df


def load_test(path):
    import tifffile
    import sys
    import numpy as np
    import time

    print ('Starting test...')
    print ('Loading file: {}'.format(path)),
    sys.stdout.flush()

    start_time = time.time()
    with tifffile.TiffFile(path, fastij=True) as tif:
        img = tif.asarray()
    end_time = time.time()
    elapsed_time = end_time - start_time
    size = (sys.getsizeof(img)) / float(1024 ** 3)

    print (' [Done]')
    print (np.shape(img))
    print ('\nTotal of {:.2f} GB'.format(size))
    print ('Elapsed time: {} seconds'.format(elapsed_time))
    sys.stdout.flush()

    return
