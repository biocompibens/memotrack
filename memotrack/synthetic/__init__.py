def get_volume(img_base, meta, display=True, verbose=False, volume=9000, sub_sampling=0.5, pixel_size=0.16125,
               max_num_samples=250000):
    from matplotlib import pyplot as plt
    import numpy as np
    import scipy.ndimage

    if verbose:
        if sub_sampling <= 1:
            print ('Using volume of ' + str(volume) + ' micro m3 and sub sampling of ' + str(sub_sampling * 100) + '%')

        if sub_sampling > 1:
            print ('Using volume of ' + str(volume) + ' micro m3 and sub sampling of ' + str(sub_sampling) + ' voxels')
    # Interpolation
    SizeZ = meta['SizeZ']
    SizeY = meta['SizeY']
    SizeX = meta['SizeX']
    PhysicalSizeX = float(meta['PhysicalSizeX'])
    PhysicalSizeY = float(meta['PhysicalSizeY'])
    PhysicalSizeZ = float(meta['PhysicalSizeZ'])

    if PhysicalSizeX != PhysicalSizeZ:
        nslices = int((SizeZ * PhysicalSizeZ) / PhysicalSizeX)
        if verbose:
            print ('Interpolating ' + str(nslices) + ' slices from ' + str(SizeZ) + ' of the original image.')
        zoom_ratio = float(nslices) / float(SizeZ)
        img_base = scipy.ndimage.interpolation.zoom(img_base, [zoom_ratio, 1, 1], order=1)

    img_base = scipy.ndimage.gaussian_filter(img_base, sigma=3.78)

    # get shape
    img_shape = np.shape(img_base)
    img_z = img_shape[0]
    img_y = img_shape[1]
    img_x = img_shape[2]

    if verbose:
        print('Image shape (ZYX):' + str(img_shape))

    # Get thresholded volume
    # Check if desired volume is possible
    max_possible_volume = (img_z * img_y * img_x) * (pixel_size ** 3)
    if volume > (max_possible_volume):
        print ('WARNING: Desired volume is bigger than the image. using '
               + str(max_possible_volume) + ' micro m3 instead.')
        volume = max_possible_volume

    # Threshold to get the volume
    thresh = 65535  # Initiate with highest 16bit threshold, to get a blank image
    current_volume = 0  # Initiate with a zero volume

    if verbose:
        print('Estimating threshold value for desired volume'),
    while_counter = 0
    while current_volume < volume:
        if while_counter > 10:
            if verbose:
                print('.'),
                while_counter = 0

        thresh -= 100  # Precision to get the volume
        img_thresh = img_base > thresh
        pixel_sum = np.sum(img_thresh)
        current_volume = pixel_sum * (pixel_size ** 3)
        while_counter += 1

    if verbose:
        print('')
        print ('Resulted volume of ' + str(current_volume) + ' micro m3.\t Error of ' + str(
            ((current_volume - volume) / volume) * 100) + '%')

    # Subsample
    # Check if using percentage or exact number of samples
    if sub_sampling <= 1:
        thresh_volume = np.sum(img_thresh)
        num_samples = sub_sampling * thresh_volume
    else:
        num_samples = sub_sampling

    # Check if the number of samples isn't too big,
    # otherwise k-means will take too long
    if num_samples > max_num_samples:
        if verbose:
            print ('\nDesired number of samples is too high (' + str(num_samples) + ')'),
            print ('using ' + str(max_num_samples) + ' instead')
        num_samples = max_num_samples


    # Copy volume to work on
    img_thresh_copy = np.array(img_thresh)

    # Create blank sampled image
    img_sampled = np.zeros(img_shape, dtype=bool)

    # Start counter
    sub_count = 0

    # Check if number of samples isn't ridiculously low
    if verbose:
        if num_samples < 2000:
            print ('\nWARNING: Subsample generated only '
                   + str(num_samples) + ' samples ! K-means probably is going to fail.')
        else:
            print('\nStarting subsample with ' + str(num_samples) + ' samples')

    # Start getting random samples
    while sub_count < num_samples:
        rand_x = np.random.random_integers(0, img_x - 1)  # Minus one otherwise gets out of bound
        rand_y = np.random.random_integers(0, img_y - 1)
        rand_z = np.random.random_integers(0, img_z - 1)

        if img_thresh_copy[rand_z, rand_y, rand_x] == True:
            # sets equal 2 for the second threshold
            img_sampled[rand_z, rand_y, rand_x] = True
            img_thresh_copy[rand_z, rand_y, rand_x] = False
            sub_count = sub_count + 1

    # Displays
    if display:
        plt.figure(figsize=(20, 50))

        plt.subplot(1, 3, 1)
        plt.imshow(img_base[(img_z / 2)], cmap=plt.cm.CMRmap, vmin=0, vmax=65535)
        plt.axis('off')
        plt.title('Original')

        plt.subplot(1, 3, 2)
        plt.imshow(img_thresh[(img_z / 2)], cmap=plt.cm.CMRmap, vmin=0, vmax=1)
        plt.axis('off')
        plt.title('Extracted volume')

        plt.subplot(1, 3, 3)
        plt.imshow(img_sampled[(img_z / 2)], cmap=plt.cm.CMRmap, vmin=0, vmax=1)
        plt.axis('off')
        plt.title('Subsampled Image')

    if verbose:
        print('\nVolume extraction done, ' + str(np.sum(img_sampled)) + ' data points.')

    return img_base, img_thresh, img_sampled


def kmeans_centroids(img_sampled, display=True, verbose=True, num_centroids=2000, nuclei_diameter=1.337,
                     pixel_size=0.16125):
    from sklearn.cluster import KMeans
    from matplotlib import pyplot as plt
    import numpy as np
    import memotrack.process

    # Checks if not wanting only one cluster (it crashes, minimum of 1)
    if num_centroids < 2:
        print ('WARNING: Number of centroids too low. Using 2 instead of ' + str(num_centroids))
        num_centroids = 2

    if verbose:
        print ('Extracting coordinates from image...')
    # Gets coordinates from the subsampled Image
    img_sampled_coords = np.nonzero(img_sampled)
    img_sampled_coords = np.transpose(img_sampled_coords)

    if verbose:
        print ('Starting K-Means...')
    # Start K-Means (n_init=number of different initializations    max_iter: max number of iterations)
    estimator = KMeans(n_clusters=num_centroids, n_jobs=5, max_iter=1, n_init=3)  # fast
    # estimator = KMeans(n_clusters=num_centroids, n_jobs=5, max_iter=10, n_init=10 ) # accurate
    estimator.fit_predict(img_sampled_coords)

    # Get the centroids coordinates
    centroids = estimator.cluster_centers_

    # Creates blank image
    img_shape = np.shape(img_sampled)  # get shape
    img_z = img_shape[0]
    img_y = img_shape[1]
    img_x = img_shape[2]

    img_centroids = np.zeros(img_shape, dtype='uint8')

    # Create binary image from coordinates
    for position in centroids:
        z = int(position[0])
        y = int(position[1])
        x = int(position[2])
        img_centroids[z, y, x] = 1

    # Check number of centers in image
    check = np.sum(img_centroids)

    if check == num_centroids:
        if verbose:
            print ('K-Means okay, ' + str(check) + ' centroids generated !')
    else:
        print ('WARNING: wanted ' + str(num_centroids) + ' centroids but got ' + str(check) + ' instead.')

    # Converting the centroids coordinates to pandas dataframe
    # This isn't the best way to do it, but it is the easiest as I used the previous memotrack function
    if verbose:
        print('\nConverting centroids to dataframe...')
    img_centroids_temp = np.zeros([1, img_z, img_y, img_x], dtype='uint8')
    img_centroids_temp[0] = img_centroids
    centroids_df = memotrack.process.create_dataframe(img_centroids_temp, verbose=verbose)

    # Create time and labels
    centroids_df['t'] = 0  # Create time zero for all points
    centroids_df['label'] = range(num_centroids)  # create label for all points

    # Code below to display the centroids
    if display:
        memotrack.display.plot_from_df(centroids_df, time_color=True, borders=True)

    # Work on centroid distances
    from scipy.spatial.distance import pdist, squareform

    # pairwise distances between all centroids
    dist = pdist(centroids)

    # creates square matrix
    square_matrix = squareform(dist)

    # print(square_matrix) # Show the square distance matrix. Use only for tests

    # gets minimum distance for each centroid
    minimum_distances = []  # initiate list for append
    for i in range(num_centroids):
        # Get the distances for one centroid
        distances_array = square_matrix[i]
        sorted_distances_array = np.sort(distances_array)
        minimum_distances.append(sorted_distances_array[1])  # Second element, because first is zero (itself)

    minimum_distances_sorted = np.sort(minimum_distances)  # sort in ascending order

    minimum_distances_sorted_micron = [i * pixel_size for i in minimum_distances_sorted]  # makes list in microns

    # print('Minimum distance array: '+str(minimum_distances_sorted_micron)) # Just for tests

    # Get info from the array of minimum distances:
    minimum = np.min(minimum_distances_sorted_micron)
    median = np.median(minimum_distances_sorted_micron)
    average = np.average(minimum_distances_sorted_micron)

    # Percentiles
    percentile_1 = minimum_distances_sorted_micron[int(num_centroids * 0.01)]
    percentile_5 = minimum_distances_sorted_micron[int(num_centroids * 0.05)]
    percentile_10 = minimum_distances_sorted_micron[int(num_centroids * 0.10)]

    # Calculates number of touching nuclei
    num_touching = sum(i < nuclei_diameter for i in minimum_distances_sorted_micron)  # count touching

    if verbose:
        print ('')
        print ('\nCalculations on minimum distances:')
        print ('Minimum distance: ' + str(minimum))
        print ('Median distance: ' + str(median))
        print ('Average distance: ' + str(average))
        print ('Percentile 1: ' + str(percentile_1))
        print ('Percentile 5: ' + str(percentile_5))
        print ('Percentile 10: ' + str(percentile_10))
        print ('Number of touching nuclei: ' + str(num_touching))
        print ('Percentage of touching nuclei: ' + str((float(num_touching) / float(num_centroids)) * 100) + '%')

    # Plot histogram of distances
    if display:
        plt.figure(figsize=(15, 8))
        # plt.hist(minimum_distances_micron, 100, histtype='stepfilled', color='blue', label='blue')
        plt.hist(minimum_distances_sorted_micron, 100, color='green', alpha=0.5)
        plt.axvline(x=nuclei_diameter, linewidth=3, color='orange')
        plt.title('Distribution of minimum distances (100 bins)')
        plt.xlabel('Distance in micrometers')

    return img_centroids, centroids_df


def make_image(img_centroids, centroids_df, meta_original, PSF_img, original_max_intensity,
               fake_PSF=False, synhtetic_PSF=True, verbose=True, size_range=3,
               debug=False, FWHMxy=1.21, pixel_size=0.16125, SNR=4):
    import numpy as np
    import skimage.morphology
    import scipy.ndimage
    import scipy.signal
    import sys

    # Getting info

    SizeZ = np.shape(img_centroids)[0]
    SizeY = np.shape(img_centroids)[1]
    SizeX = np.shape(img_centroids)[2]
    Original_SizeZ = meta_original['SizeZ']
    Original_SizeT = meta_original['SizeT']
    nlabels = centroids_df['label'].nunique()

    # Normalizing PSF
    PSF_float = np.asarray(PSF_img, dtype='float')
    PSF_max = np.max(PSF_float)
    PSF_float = PSF_float / PSF_max

    # Creates blank image
    img = np.zeros([SizeZ, SizeY, SizeX], dtype='uint16')

    avg_radius = ((FWHMxy*1.4142)/2)/pixel_size
    print ('Average nuclei radius: ' + str(avg_radius) + ' pixels')
    print ('Range from ' + str(avg_radius-size_range) + ' to ' + str(avg_radius+size_range) + ' pixels')
    # Adding values from each coordinate
    if verbose:
        print ('Adding centroids to image'),
        sys.stdout.flush()

    for label in range(nlabels):
        temp_df = centroids_df[centroids_df['label'] == label]
        x = int(temp_df['x'].values)
        y = int(temp_df['y'].values)
        z = int(temp_df['z'].values)

        if size_range > 0:
            kernel_radius = np.random.randint(low=avg_radius-size_range, high=avg_radius+size_range)
        else:
            kernel_radius = avg_radius
        nuclei = skimage.morphology.ball(kernel_radius)
        nuclei_center_z = (np.shape(nuclei)[0])/2
        nuclei_center_y = (np.shape(nuclei)[1])/2
        nuclei_center_x = (np.shape(nuclei)[2])/2

        # Adding to img
        for xpos in range(-nuclei_center_x, (nuclei_center_x+1)):
            for ypos in range(-nuclei_center_y, (nuclei_center_y+1)):
                for zpos in range(-nuclei_center_z, (nuclei_center_z+1)):
                    value = nuclei[zpos+nuclei_center_z, ypos+nuclei_center_y, xpos+nuclei_center_x]
                    if img[z+zpos, y+ypos, x+xpos] == 0:
                        img[z+zpos, y+ypos, x+xpos] = value

    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    img_nuclei = img
    # Signal heterogeneity
    if verbose:
        print ('Generating signal heterogeneity'),
        sys.stdout.flush()
    img = img_nuclei * (np.random.rand(SizeZ, SizeY, SizeX))

    if verbose:
        print ('[Done]')
        sys.stdout.flush()
    if debug:
        print ('***Shape before convolution: ' + str(np.shape(img)))
    # Convolution with PSF
    if fake_PSF:
        if verbose:
            print ('*** Using fake PSF mode !'),
            sys.stdout.flush()
        sig_z = 15
        sig_xy = 2
        img *= 400
        img = scipy.ndimage.filters.gaussian_filter(img, sigma=[sig_z, sig_xy, sig_xy])
        if verbose:
            print ('[Done]')
            sys.stdout.flush()
    else:
        if verbose:
            print ('Convolution with PSF'),
            sys.stdout.flush()
        # Convolution using FFT
        img = scipy.signal.fftconvolve(img, PSF_float, mode='same')
        if verbose:
            print ('[Done]')
            sys.stdout.flush()
    if debug:
        print ('***Shape after convolution: ' + str(np.shape(img)))

    # Glow
    if verbose:
        print ('Generating background glow'),
        sys.stdout.flush()
    glow = scipy.ndimage.filters.gaussian_filter(img, sigma=[30, 30, 30])
    glow = scipy.ndimage.filters.gaussian_filter1d(glow, sigma=120, axis=0)
    img = img + (glow * 3)

    #back_level = np.ones(np.shape(img)) * 1
    #img += back_level
    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    # Getting temp image for output
    img_temp = img

    # Negative value correction
    img[img<0] = 0

    if debug:
        print ('\n***img shape: ' + str(np.shape(img)))
        print ('***img max: ' + str(np.max(img)))
        print ('***img avg: ' + str(np.mean(img)))
        print ('***img min: ' + str(np.min(img)))


    # Apply poisson noise
    if verbose:
        print ('Applying poisson noise'),
        sys.stdout.flush()
    # Get noise factor from given SNR

    # Values from fitted equation
    a = 1.1624989283988068
    b = 1.0103057007957579
    c = 26.194262107957016
    d = 4.3279958004125936
    noise_factor = c*(((a-d)/(SNR-d))-1)**(1/b)


    img = img.astype(float)
    img /= np.max(img)
    img *= noise_factor
    img = np.random.poisson(lam=img)
    img = img.astype(float)  # Converting because it was int64

    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    # Final scale adjustment
    if verbose:
        print ('Adjusting intensity scale'),
        sys.stdout.flush()
    img -= np.min(img)
    max_intensity = float(np.max(img))
    img /= max_intensity
    img *= float(original_max_intensity)
    img = np.asarray(img, dtype='uint16')
    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    # Check SNR
    if verbose:
        print ('Checking noise level...'),
        sys.stdout.flush()
    img_SNR = img_nuclei * img  # masking with the dilated nuclei
    img_temp = img_SNR
    img_SNR = img_SNR.flatten()  # flattening values to use list comprehension
    values_list = [i for i in img_SNR if i > 0]
    real_SNR = np.mean(values_list) / np.std(values_list)
    if verbose:
        print ('Obtained SNR of ' + str(real_SNR))

    # Resizing image
    if verbose:
        print ('Adjusting size'),
        sys.stdout.flush()
    shrink_ratio = float(Original_SizeZ) / float(SizeZ)
    img_small = scipy.ndimage.interpolation.zoom(img, [shrink_ratio, 1, 1])

    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    if verbose:
        print ('\nAll done !')

    return img_small, img_temp

def get_volume2(img_base, meta, display=True, verbose=False, volume=9000, sub_sampling=0.1, pixel_size=0.16125,
               max_num_samples = 4800000):

    from matplotlib import pyplot as plt
    import numpy as np
    import scipy.ndimage

    if verbose:
        if sub_sampling <= 1:
            print ('Using volume of ' + str(volume) + ' micro m3 and sub sampling of ' + str(sub_sampling * 100) + '%')

        if sub_sampling > 1:
            print ('Using volume of ' + str(volume) + ' micro m3 and sub sampling of ' + str(sub_sampling) + ' voxels')
    # Interpolation
    SizeZ = meta['SizeZ']
    SizeY = meta['SizeY']
    SizeX = meta['SizeX']
    PhysicalSizeX = float(meta['PhysicalSizeX'])
    PhysicalSizeY = float(meta['PhysicalSizeY'])
    PhysicalSizeZ = float(meta['PhysicalSizeZ'])

    if PhysicalSizeX != PhysicalSizeZ:
        nslices = int((SizeZ * PhysicalSizeZ) / PhysicalSizeX)
        if verbose:
            print ('Interpolating ' + str(nslices) + ' slices from ' + str(SizeZ) + ' of the original image.')
        zoom_ratio = float(nslices) / float(SizeZ)
        img_base = scipy.ndimage.interpolation.zoom(img_base, [zoom_ratio, 1, 1], order=1)


    # smooth the image with Gaussian_filter
    img_smooth = scipy.ndimage.gaussian_filter(img_base, sigma=3.78)
    if verbose :
        print ('\nHere begin the filter with sigma = 3.78 \n')

    # get shape
    img_shape = np.shape(img_smooth)
    img_z = img_shape[0]
    img_y = img_shape[1]
    img_x = img_shape[2]



    if verbose:
        print('Image shape (ZYX):' + str(img_shape))

    # Get thresholded volume
    # Check if desired volume is possible
    max_possible_volume = (img_z * img_y * img_x) * (pixel_size ** 3)
    if volume > (max_possible_volume):
        print ('WARNING: Desired volume is bigger than the image. using '
               + str(max_possible_volume) + ' micro m3 instead.')
        volume = max_possible_volume

    # Threshold to get the volume
    thresh = 65535  # Initiate with highest 16bit threshold, to get a blank image
    current_volume = 0  # Initiate with a zero volume

    if verbose:
        print('Estimating threshold value for desired volume'),
    while_counter = 0
    while current_volume < volume:
        if while_counter > 10:
            if verbose:
                print('.'),
                while_counter = 0

        thresh -= 100  # Precision to get the volume
        img_thresh = img_smooth > thresh
        pixel_sum = np.sum(img_thresh)
        current_volume = pixel_sum * (pixel_size ** 3)
        while_counter += 1


    # get the biggest component after the treatment of extracted volume
    labeled_array, num_features = scipy.ndimage.label(img_thresh)
    print '\n',num_features, '\n'

    # create blank image
    img_create = np.shape(img_thresh)
    img_z = img_create[0]
    img_y = img_create[1]
    img_x = img_create[2]

    # img_component = np.zeros(img_create)
    img_component = np.array(labeled_array)

    i = 0
    num_count = 0
    while num_count < max_num_samples * 0.8:
        i = i + 1
        num_count = 0
        for x in range(img_x):
            for y in range(img_y):
                for z in range(img_z):
                    if labeled_array[z, y, x] == i:
                        num_count = num_count + 1

    print '\nnum_count = ', num_count
    print '\ni = ', i

    for x in range(img_x):
        for y in range(img_y):
            for z in range(img_z):
                if labeled_array[z, y, x] != i:
                    img_component[z, y, x] = 0

    labeled_array00, num_features00 = scipy.ndimage.label(img_component)

    print 'now the number of feature = ',num_features00, '\n'
    # end of modification


    if verbose:
        print('')
        print ('Resulted volume of ' + str(current_volume) + ' micro m3.\t Error of ' + str(
            ((current_volume - volume) / volume) * 100) + '%')


    # Subsample
    # Check if using percentage or exact number of samples
    if sub_sampling <= 1:
        thresh_volume = np.sum(img_thresh)
        num_samples = sub_sampling * thresh_volume
    else:
        num_samples = sub_sampling

    # Check if the number of samples isn't too big,
    # otherwise k-means will take too long
    #if num_samples > max_num_samples :
    #    if verbose:
    #        print ('\nDesired number of samples is too high ( '+ str(num_samples) + ')')
    #        print ('using '+ str(max_num_samples)+ ' instead')
    #    num_samples = max_num_samples

    # Copy volume to work on
    #img_thresh_copy = np.array(img_thresh)

    # Create blank sampled image
    #img_sampled = np.zeros(img_shape, dtype=bool)

    # Start counter
    #sub_count = 0

    # Check if number of samples isn't ridiculously low
    #if verbose:
    #    if num_samples < 2000:
    #        print ('\nWARNING: Subsample generated only '+ str(num_samples) + ' samples ! K-means probably is going to fail.')
    #    else:
    #        print('\nStarting subsample with ' + str(num_samples) + ' samples')

    # Start getting random samples
    #while sub_count < num_samples:
    #    rand_x = np.random.random_integers(0, img_x - 1)  # Minus one otherwise gets out of bound
    #    rand_y = np.random.random_integers(0, img_y - 1)
    #    rand_z = np.random.random_integers(0, img_z - 1)

    #    if img_thresh_copy[rand_z, rand_y, rand_x] == True:
    # sets equal 2 for the second threshold
    #        img_sampled[rand_z, rand_y, rand_x] = True
    #        img_thresh_copy[rand_z, rand_y, rand_x] = False
    #        sub_count = sub_count + 1


    # Start to check all the points in the img_sampled
    #for i in np.arange(img_x):
    #    for j in np.arange(img_y):
    #        for k in np.arange(img_z):
    #            if img_thresh_copy[k,j,i] == True:
    #                img_sampled[k,j,i] = True
    #                img_thresh_copy[k,j,i] =False
    #                sub_count = sub_count + 1


    # Start to get the random but representitive samples



    #print('\n\n Ending subsample with ' + str(sub_count) + ' samples')
    #ends of modification



    # Displays
    if display:
        plt.figure(figsize=(20, 20))

        plt.subplot(2, 3, 1)
        plt.imshow(img_base[(img_z / 2)], cmap=plt.cm.CMRmap, vmin=0, vmax=65535)
        plt.axis('off')
        plt.title('Original')

        plt.subplot(2, 3, 2)
        plt.imshow(img_smooth[(img_z / 2)], cmap=plt.cm.CMRmap, vmin=0, vmax=65535)
        plt.axis('off')
        plt.title('Smoothed image')

        plt.subplot(2, 3, 3)
        plt.imshow(img_thresh[(img_z / 2)], cmap=plt.cm.CMRmap,vmin=0, vmax=1)
        plt.axis('off')
        plt.title('Extracted volume')

        plt.subplot(2, 3, 4)
        plt.imshow(img_thresh[(img_z / 2)], cmap=plt.cm.CMRmap, vmin=0, vmax=1)
        plt.axis('off')
        plt.title('Extracted volume')

        plt.subplot(2, 3, 5)
        plt.imshow(img_component[(img_z / 2)], cmap=plt.cm.CMRmap, vmin=0, vmax=1)
        plt.axis('off')
        plt.title('The biggest component')



    #if verbose:
    #    print('\nVolume extraction done, ' + str(np.sum(img_sampled)) + ' data points.')

    return img_base, img_thresh, img_component

