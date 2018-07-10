def centroid_registration(detections_df, verbose=True, skip_registration=False):
    # For tests, the skip registration flag
    if skip_registration:
        print ('\n' + ('*' * 37))
        print ('WARNING: Ignoring registration step !')
        print ('*' * 37)
        detections_df['xreg'] = detections_df['x']
        detections_df['yreg'] = detections_df['y']
        detections_df['zreg'] = detections_df['z']
        return detections_df

    nframes = detections_df['t'].nunique()

    if verbose:
        print ('\nInitializing registration of dataset on ' + str(nframes) + ' frames')

    # Create columns for registered coordinates
    detections_df['xreg'] = 0
    detections_df['yreg'] = 0
    detections_df['zreg'] = 0

    for frame in range(nframes):
        # Get the centroid of the current frame
        centroidx = (detections_df[detections_df['t'] == frame].loc[:, 'x'].mean())
        centroidy = (detections_df[detections_df['t'] == frame].loc[:, 'y'].mean())
        centroidz = (detections_df[detections_df['t'] == frame].loc[:, 'z'].mean())

        # Set registered coordinates on the dataframe
        detections_df.loc[detections_df['t'] == frame, 'xreg'] = detections_df.loc[
                                                                     detections_df['t'] == frame, 'x'] - centroidx
        detections_df.loc[detections_df['t'] == frame, 'yreg'] = detections_df.loc[
                                                                     detections_df['t'] == frame, 'y'] - centroidy
        detections_df.loc[detections_df['t'] == frame, 'zreg'] = detections_df.loc[
                                                                     detections_df['t'] == frame, 'z'] - centroidz

    return detections_df


def create_cost_matrix(detections_df, frame=0, data_fraction=0.1, verbose=True,
                       debug=False, display=False, method='center'):
    # Generates the subsampling matrix to be used together with the cost function
    import memotrack
    import numpy as np
    import munkres
    import scipy.spatial.distance as dist
    from matplotlib import pyplot as plt
    import sys
    import time

    nframes = detections_df['t'].nunique()
    work_frame = frame

    # Check for the reference frame
    ref_frame = int(nframes / 2)  # Old method
    # if frame < nframes:
    #     ref_frame = frame + 10
    # else:
    #     ref_frame = frame

    if verbose:
        print('Registering frame ' + str(work_frame) + ' to frame ' + str(ref_frame))

    import scipy.spatial.distance as dist
    from matplotlib import pyplot as plt

    # Gets matrix of coordinates
    ref_frame_df = detections_df[detections_df['t'] == ref_frame]
    x_ref = ref_frame_df['x'].values
    y_ref = ref_frame_df['y'].values
    z_ref = ref_frame_df['z'].values
    ref_coords = [x_ref, y_ref, z_ref]
    ref_coords = np.transpose(ref_coords)

    work_frame_df = detections_df[detections_df['t'] == work_frame]
    x_work = work_frame_df['xreg'].values
    y_work = work_frame_df['yreg'].values
    z_work = work_frame_df['zreg'].values
    work_coords = [x_work, y_work, z_work]
    work_coords = np.transpose(work_coords)

    # Calculate distance matrix
    dist_matrix = dist.cdist(ref_coords, work_coords)
    min_detections = min(np.shape(dist_matrix))
    npoints = min_detections * data_fraction

    if display:
        plt.figure(figsize=(8.7, 7))
        plt.imshow(dist_matrix, cmap='RdYlBu_r', interpolation='nearest')
        plt.title('Raw distance matrix')
        plt.colorbar()

    # If the user gives a value between 0 and 1, uses a fraction of data.
    # If the value is bigger than 1, use the exact number of points given
    if data_fraction > 1:
        npoints = int(data_fraction)

    if method == 'center':
        # Get only the center region of the distance matrix, based on desired "data_fraction"
        final_matrix = dist_matrix[int((min_detections / 2) - npoints / 2):int((min_detections / 2) + npoints / 2),
                       int((min_detections / 2) - npoints / 2):int((min_detections / 2) + npoints / 2)]

    if method == 'closest':
        # Get the npoints smallest values (in a flatten array)
        smallest_coords_flat = np.argpartition(dist_matrix, npoints, axis=None)
        smallest_coords_flat = smallest_coords_flat[:npoints]

        # indices from flatten to 2D matrix
        smallest_coords = np.unravel_index(smallest_coords_flat, np.shape(dist_matrix))
        if debug:
            print ('Coordinates of closest points')
            for coords in smallest_coords:
                print (coords)
            print ('\nValues from closest points')
            print(dist_matrix[smallest_coords])

        # Expand list of coordinates
        expanded_list_row = np.zeros(npoints * npoints, dtype=int)
        expanded_list_col = np.zeros(npoints * npoints, dtype=int)
        pos_counter = 0
        pos2_counter = 0
        for pos in range(len(smallest_coords[0])):
            value1 = smallest_coords[0][pos]
            for pos2 in range(len(smallest_coords[1])):
                value2 = smallest_coords[1][pos2]

                expanded_list_row[pos_counter] = int(value1)
                expanded_list_col[pos2_counter] = int(value2)

                pos_counter += 1
                pos2_counter += 1

        expanded_list = [expanded_list_row, expanded_list_col]
        final_matrix = dist_matrix[expanded_list]

        if debug:
            print('\nExpanded list:')
            for coords in expanded_list:
                print (coords)
            print ('\nValues from expanded list')

        final_matrix = final_matrix.reshape(npoints, npoints)

        if debug:
            print ('\nShape of final matrix: ' + str(np.shape(final_matrix)))

    # Distance Matrix
    if display:
        plt.figure(figsize=(8.7, 7))
        plt.imshow(final_matrix, cmap='RdYlBu_r', interpolation='nearest')
        plt.title('Distance matrix for ' + str(data_fraction) + ' points (method: ' + str(method) + ')')
        plt.colorbar()

    return final_matrix


def cost_function(cost_matrix, verbose=True):
    import memotrack
    import numpy as np
    import munkres
    import scipy.spatial.distance as dist
    from matplotlib import pyplot as plt
    import sys
    import time

    work_matrix = np.copy(cost_matrix)

    if verbose:
        print ('Using ' + str(np.shape(work_matrix)[0]) + ' points for cost calculation.')
        sys.stdout.flush()
    # Calculate cost
    m = munkres.Munkres()
    coords = m.compute(work_matrix)
    cost = 0  # Set initial cost of zero
    for coord in coords:
        # Test changes
        if coord[0] == coord[1]:
            print ('.'),
        else:
            print ('!'),
        sys.stdout.flush()

        cost += cost_matrix[coord]  # Get the cost for each assigned coordinate

    if verbose:
        print('Final cost of ' + str(cost))
        sys.stdout.flush()

    return cost


def cost_function_old(detections_df, frame=0, data_fraction=0.1, verbose=True,
                      debug=False, display=False, method='center'):
    import memotrack
    import numpy as np
    import munkres
    import scipy.spatial.distance as dist
    from matplotlib import pyplot as plt
    import sys

    nframes = detections_df['t'].nunique()
    work_frame = frame
    ref_frame = int(nframes / 2)

    import scipy.spatial.distance as dist
    from matplotlib import pyplot as plt

    # Gets matrix of coordinates
    ref_frame_df = detections_df[detections_df['t'] == ref_frame]
    x_ref = ref_frame_df['x'].values
    y_ref = ref_frame_df['y'].values
    z_ref = ref_frame_df['z'].values
    ref_coords = [x_ref, y_ref, z_ref]
    ref_coords = np.transpose(ref_coords)

    work_frame_df = detections_df[detections_df['t'] == work_frame]
    x_work = work_frame_df['xreg'].values
    y_work = work_frame_df['yreg'].values
    z_work = work_frame_df['zreg'].values
    work_coords = [x_work, y_work, z_work]
    work_coords = np.transpose(work_coords)

    # Calculate distance matrix
    dist_matrix = dist.cdist(ref_coords, work_coords)
    min_detections = min(np.shape(dist_matrix))
    npoints = min_detections * data_fraction
    if display:
        plt.figure(figsize=(8.7, 7))
        plt.imshow(dist_matrix, cmap='RdYlBu_r', interpolation='nearest')
        plt.title('Raw distance matrix')
        plt.colorbar()

    # If the user gives a value between 0 and 1, uses a fraction of data.
    # If the value is bigger than 1, use the exact number of points given
    if data_fraction > 1:
        npoints = int(data_fraction)

    if method == 'center':
        # Get only the center region of the distance matrix, based on desired "data_fraction"
        final_matrix = dist_matrix[int((min_detections / 2) - npoints / 2):int((min_detections / 2) + npoints / 2),
                       int((min_detections / 2) - npoints / 2):int((min_detections / 2) + npoints / 2)]

    if method == 'closest':
        # Get the npoints smallest values (in a flatten array)
        smallest_coords_flat = np.argpartition(dist_matrix, npoints, axis=None)
        smallest_coords_flat = smallest_coords_flat[:npoints]

        # indices from flatten to 2D matrix
        smallest_coords = np.unravel_index(smallest_coords_flat, np.shape(dist_matrix))
        if debug:
            print ('Coordinates of closest points')
            for coords in smallest_coords:
                print (coords)
            print ('\nValues from closest points')
            print(dist_matrix[smallest_coords])

        # Expand list of coordinates
        expanded_list_row = np.zeros(npoints * npoints, dtype=int)
        expanded_list_col = np.zeros(npoints * npoints, dtype=int)
        pos_counter = 0
        pos2_counter = 0
        for pos in range(len(smallest_coords[0])):
            value1 = smallest_coords[0][pos]
            for pos2 in range(len(smallest_coords[1])):
                value2 = smallest_coords[1][pos2]

                expanded_list_row[pos_counter] = int(value1)
                expanded_list_col[pos2_counter] = int(value2)

                pos_counter += 1
                pos2_counter += 1

        expanded_list = [expanded_list_row, expanded_list_col]
        final_matrix = dist_matrix[expanded_list]

        if debug:
            print('\nExpanded list:')
            for coords in expanded_list:
                print (coords)
            print ('\nValues from expanded list')
            print (final_matrix)

        final_matrix = final_matrix.reshape(npoints, npoints)
        if debug:
            print ('\nShape of final matrix: ' + str(np.shape(final_matrix)))

    # Distance Matrix
    if display:
        plt.figure(figsize=(8.7, 7))
        plt.imshow(final_matrix, cmap='RdYlBu_r', interpolation='nearest')
        plt.title('Distance matrix for ' + str(data_fraction) + ' points (method: ' + str(method) + ')')
        plt.colorbar()

    # Get work matrix for LAP
    work_matrix = np.copy(final_matrix)

    if verbose:
        print ('Using ' + str(np.shape(work_matrix)[0]) + ' points for cost calculation.')
        sys.stdout.flush()

    # Calculate cost
    m = munkres.Munkres()
    coords = m.compute(work_matrix)
    cost = 0  # Set initial cost of zero
    for coord in coords:
        cost += final_matrix[coord]  # Get the cost for each assigned coordinate

    if verbose:
        print('Final cost of ' + str(cost))
        sys.stdout.flush()
    return cost


def transform_frame(transform_matrix, detections_df, frame, data_fraction=100, debug=False, verbose=False):
    import memotrack.registration
    import numpy as np

    # aa, ab, ac, ba, bb, bc, ca, cb, cc = transform_matrix
    aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd = transform_matrix

    if debug:
        print (np.around(aa, decimals=3)),
        print (np.around(ab, decimals=3)),
        print (np.around(bb, decimals=3)),
        print (np.around(cc, decimals=3)),

    # Create array for transformation
    # matrix = np.array([[aa, ab, ac], [ba, bb, bc], [ca, cb, cc]])
    matrix = np.array([[aa, ab, ac, ad],
                       [ba, bb, bc, bd],
                       [ca, cb, cc, cd],
                       [da, db, dc, dd]])

    # Get the indices we're working on
    work_frame_df = detections_df[detections_df['t'] == frame]
    frame_indices = detections_df[detections_df['t'] == frame].index

    # Apply transformation
    detections_df.loc[frame_indices, ('xreg', 'yreg', 'zreg', 'wreg')] = np.dot(work_frame_df[['x', 'y', 'z', 'w']],
                                                                                matrix)

    # Generate again the cost matrix
    cost_matrix = memotrack.registration.create_cost_matrix(detections_df, frame=frame, data_fraction=data_fraction,
                                                            method='closest', verbose=False)
    # Calculate cost for the registered frame
    cost = memotrack.registration.cost_function(cost_matrix, verbose=False)

    '''
    cost = memotrack.registration.cost_function(detections_df, frame=frame, data_fraction=data_fraction,
                                                method='closest', verbose=False, display=False)
    '''
    if verbose:
        print ('\nCost: ' + str(np.around(cost, decimals=5)))
        print ('Matrix:\n' + str(np.around(matrix, decimals=5)))

    # Set the transform matrix as global, for checking later
    global current_matrix
    current_matrix = matrix

    return cost


def register_frame(detections_df, frame=0, data_fraction=100, verbose=False):
    import numpy as np
    import memotrack.registration
    import scipy.optimize
    import sys
    import time

    temp_timer_start = time.time()

    if verbose:
        print ('\nRegistering frame ' + str(frame)),
        sys.stdout.flush()

    # First time only. Creating registered coords as copy and reseting index
    if 'xreg' not in detections_df.columns:
        detections_df = detections_df.reset_index()
        detections_df['w'] = 1
        detections_df['wreg'] = 1
        detections_df['xreg'] = detections_df['x']
        detections_df['yreg'] = detections_df['y']
        detections_df['zreg'] = detections_df['z']

    # Identity matrix as initial parameters
    # initial = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    initial = np.array([1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1])

    # Limit the boundaries of search (transformation matrix)
    d = 0.2  # For matrix diagonal
    v = 1.0  # For other values
    t = 10
    # bounds = [[0, d], [-v, v], [-v, v], [-v, v], [0, d], [-v, v], [-v, v], [-v, v], [0, d]]
    bounds = [[1 - d, 1 + d], [-v, v], [-v, v], [-v, v],
              [-v, v], [1 - d, 1 + d], [-v, v], [-v, v],
              [-v, v], [-v, v], [1 - d, 1 + d], [-v, v],
              [-t, t], [-t, t], [-t, t], [1 - d, 1 + d]]

    cost_matrix = memotrack.registration.create_cost_matrix(detections_df, frame=frame, data_fraction=data_fraction,
                                                            method='closest', verbose=False, display=False)

    if verbose:
        print ('\nInitial cost: '),
        sys.stdout.flush()

        initial_cost = memotrack.registration.cost_function(cost_matrix, verbose=False)

        '''
        initial_cost = memotrack.registration.cost_function(detections_df, frame=frame, data_fraction=data_fraction,
                                                            method='closest', verbose=False, display=False)
        '''
        print (initial_cost)
        sys.stdout.flush()

    # Optimization
    result = scipy.optimize.minimize(memotrack.registration.transform_frame, initial,
                                     args=(detections_df, frame, cost_matrix, data_fraction),
                                     bounds=bounds, method='SLSQP')

    total_time = (time.time() - temp_timer_start)
    if verbose:
        print ('Final cost of ' + str(result.fun) + ' using ' + str(result.nit) + ' iterations.')
        print ('Matrix:\n' + str(np.around(current_matrix, decimals=3)))
        print ('Elapsed time: ' + str(total_time) + ' seconds')
        sys.stdout.flush()

    return detections_df


def all_frames(detections_df, data_fraction=100, verbose=False):
    import memotrack.registration
    import sys
    nframes = detections_df['t'].nunique()
    if verbose:
        print ('\nRegistration on ' + str(nframes) + ' frames'),
        sys.stdout.flush()

    for frame in range(nframes):
        if verbose:
            print ('.'),
            sys.stdout.flush()
        detections_df = memotrack.registration.register_frame(detections_df, frame=frame,
                                                              data_fraction=data_fraction, verbose=verbose)
    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    return detections_df


def chained_registration_backup_munkres(detections_df, data_fraction, verbose=False, display=False, debug=False):
    import memotrack.registration
    import sys
    import numpy as np
    import scipy.spatial.distance as dist
    from matplotlib import pyplot as plt
    import munkres

    # Get total number of frames
    nframes = detections_df['t'].nunique()

    if verbose:
        print ('\nRegistration on ' + str(nframes) + ' frames'),
        sys.stdout.flush()

    # Add the registered columns to the df. Here we have "W" because of the 4x4 transformation matrix
    detections_df = detections_df.reset_index()
    detections_df['w'] = 1
    detections_df['wreg'] = 1
    detections_df['xreg'] = detections_df['x']
    detections_df['yreg'] = detections_df['y']
    detections_df['zreg'] = detections_df['z']

    for frame in range(3):  # nframes-1):  # Here we use "-1" because the last frame doesn't need to be registered !
        if verbose:
            print ('.'),
            sys.stdout.flush()

        # Gets matrix of coordinates for reference frame
        ref_frame_df = detections_df[detections_df['t'] == frame + 1]
        x_ref = ref_frame_df['x'].values
        y_ref = ref_frame_df['y'].values
        z_ref = ref_frame_df['z'].values
        ref_coords = [x_ref, y_ref, z_ref]
        ref_coords = np.transpose(ref_coords)

        # Gets matrix of coordinates for frame that will be registered
        work_frame_df = detections_df[detections_df['t'] == frame]
        x_work = work_frame_df['xreg'].values
        y_work = work_frame_df['yreg'].values
        z_work = work_frame_df['zreg'].values
        work_coords = [x_work, y_work, z_work]
        work_coords = np.transpose(work_coords)

        # Calculate distance matrix
        dist_matrix = dist.cdist(ref_coords, work_coords)
        min_detections = min(np.shape(dist_matrix))
        npoints = data_fraction

        if display:
            plt.figure(figsize=(8.7, 7))
            plt.imshow(dist_matrix, cmap='RdYlBu_r', interpolation='nearest')
            plt.title('Frame ' + str(frame) + '  | Raw distance matrix')
            plt.colorbar()

        # Get coordinates pairs to be used for minimization
        # Get the npoints smallest values (in a flatten array)
        smallest_coords_flat = np.argpartition(dist_matrix, npoints, axis=None)
        smallest_coords_flat = smallest_coords_flat[:npoints]
        print ('\nSmallest coords flat:')
        print (np.shape(smallest_coords_flat))

        # indices from flatten to 2D matrix
        smallest_coords = np.unravel_index(smallest_coords_flat, np.shape(dist_matrix))

        print ('\nSmallest coords:')
        print (np.shape(smallest_coords))

        if debug:
            print ('Coordinates of closest points')
            for coords in smallest_coords:
                print (coords)
            print ('\nValues from closest points')
            print(dist_matrix[smallest_coords])

        # Expand list of coordinates
        expanded_list_row = np.zeros(npoints * npoints, dtype=int)
        expanded_list_col = np.zeros(npoints * npoints, dtype=int)
        pos_counter = 0
        pos2_counter = 0
        for pos in range(len(smallest_coords[0])):
            value1 = smallest_coords[0][pos]
            for pos2 in range(len(smallest_coords[1])):
                value2 = smallest_coords[1][pos2]

                expanded_list_row[pos_counter] = int(value1)
                expanded_list_col[pos2_counter] = int(value2)

                pos_counter += 1
                pos2_counter += 1

        expanded_list = [expanded_list_row, expanded_list_col]
        final_matrix = dist_matrix[expanded_list]

        if debug:
            print('\nExpanded list:')
            for coords in expanded_list:
                print (coords)
            print ('\nValues from expanded list')

        # Here we have the matrix to be used with the Munkres algorithmn
        final_matrix = final_matrix.reshape(npoints, npoints)

        if debug:
            print ('\nShape of final matrix: ' + str(np.shape(final_matrix)))

        # Distance Matrix
        if display:
            plt.figure(figsize=(8.7, 7))
            plt.imshow(final_matrix, cmap='RdYlBu_r', interpolation='nearest')
            plt.title('Frame ' + str(frame) + '  | Distance matrix for ' + str(data_fraction))
            plt.colorbar()

        work_matrix = np.copy(final_matrix)

        if verbose:
            print ('Using ' + str(np.shape(work_matrix)[0]) + ' points for cost calculation.')
            sys.stdout.flush()

        # Calculate Munkres
        m = munkres.Munkres()
        coords = m.compute(work_matrix)

        # Check current cost
        cost = 0  # Set initial cost of zero
        for coord in coords:
            sys.stdout.flush()

            cost += final_matrix[coord]  # Get the cost for each assigned coordinate

        if verbose:
            print('Initial cost: ' + str(cost))
            sys.stdout.flush()

        print ('\nCoordinates pairs:')
        print (coords)



        # detections_df = memotrack.registration.register_frame(detections_df, frame=frame, data_fraction=data_fraction, verbose=verbose)

    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    return detections_df


def chained_registration(detections_df, reg_step=10, verbose=False, display=False, debug=False):
    import memotrack.registration
    import sys
    import numpy as np
    import math

    # Get total number of frames
    nframes = detections_df['t'].nunique()

    # Get the reference frame for the required registration step
    middle_frame = int(math.floor(nframes / (2 * reg_step)) * reg_step)

    if verbose:
        print ('Registration step of ' + str(reg_step))
        print ('Using frame ' + str(middle_frame) + ' as reference')
        print ('\nRegistration on ' + str(nframes) + ' frames \nCost ratios:')
        sys.stdout.flush()

    # Creating registered coords as copy and reseting index
    if 'xreg' not in detections_df.columns:
        detections_df = detections_df.reset_index()
        detections_df['w'] = 1
        detections_df['wreg'] = 1
        detections_df['xreg'] = detections_df['x']
        detections_df['yreg'] = detections_df['y']
        detections_df['zreg'] = detections_df['z']

        # Add the registered columns to the df. Here we have "W" because of the 4x4 transformation matrix
        detections_df = detections_df.reset_index()
        detections_df['w'] = 1
        detections_df['wreg'] = 1
        detections_df['xreg'] = detections_df['x']
        detections_df['yreg'] = detections_df['y']
        detections_df['zreg'] = detections_df['z']

    matrix_list = []

    for frame in range(nframes):
        # if verbose:
        #    print ('.'),
        #    sys.stdout.flush()

        if frame < middle_frame:
            ref_frame = frame + reg_step
        else:
            ref_frame = frame - reg_step

        initial_cost = memotrack.registration.simple_cost(detections_df, frame, ref_frame)

        if debug:
            print ('Initial cost: ' + str(initial_cost))
            sys.stdout.flush()

        detections_df, final_matrix = memotrack.registration.register_frame_simple(detections_df, frame, ref_frame,
                                                                                   verbose=False)

        matrix_list.append(final_matrix)

    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    return detections_df, matrix_list, middle_frame


def register_frame_simple(detections_df, frame, ref_frame, data_fraction=100, verbose=False):
    import numpy as np
    import memotrack.registration
    import scipy.optimize
    import sys
    import time

    temp_timer_start = time.time()

    if verbose:
        print ('\nRegistering frame ' + str(frame)),
        sys.stdout.flush()

    # Identity matrix as initial parameters
    # initial = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    initial = np.array([1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1])

    # Limit the boundaries of search (transformation matrix)
    d = 0.2  # For matrix diagonal
    v = 0.5  # For other values
    t = 10  # For translation
    # bounds = [[0, d], [-v, v], [-v, v], [-v, v], [0, d], [-v, v], [-v, v], [-v, v], [0, d]]
    bounds = [[1 - d, 1 + d], [-v, v], [-v, v], [-v, v],
              [-v, v], [1 - d, 1 + d], [-v, v], [-v, v],
              [-v, v], [-v, v], [1 - d, 1 + d], [-v, v],
              [-t, t], [-t, t], [-t, t], [1 - d, 1 + d]]

    initial_cost = memotrack.registration.simple_cost(detections_df, frame, ref_frame, verbose=False, display=False)

    if verbose:
        print ('\nInitial cost: '),
        print (initial_cost)
        sys.stdout.flush()

    # Optimization
    result = scipy.optimize.minimize(memotrack.registration.transform_frame_simple, initial,
                                     args=(detections_df, frame, ref_frame, data_fraction),
                                     bounds=bounds, method='SLSQP')

    total_time = (time.time() - temp_timer_start)
    if verbose:
        print ('Final cost of ' + str(result.fun) + ' using ' + str(result.nit) + ' iterations.')
        print ('Matrix:\n' + str(np.around(result.x, decimals=3)))
        print ('Elapsed time: ' + str(total_time) + ' seconds')
        sys.stdout.flush()

    final_matrix = result.x

    cost_ratio = result.fun / initial_cost

    print ('%.2f' % cost_ratio),

    return detections_df, final_matrix


def simple_cost(detections_df, frame, ref_frame, verbose=False, display=False):
    import numpy as np
    import scipy.spatial.distance as dist
    from matplotlib import pyplot as plt

    work_frame = frame

    # Gets matrix of coordinates
    ref_frame_df = detections_df[detections_df['t'] == ref_frame]
    x_ref = ref_frame_df['x'].values
    y_ref = ref_frame_df['y'].values
    z_ref = ref_frame_df['z'].values
    ref_coords = [x_ref, y_ref, z_ref]
    ref_coords = np.transpose(ref_coords)

    work_frame_df = detections_df[detections_df['t'] == work_frame]
    x_work = work_frame_df['xreg'].values
    y_work = work_frame_df['yreg'].values
    z_work = work_frame_df['zreg'].values
    work_coords = [x_work, y_work, z_work]
    work_coords = np.transpose(work_coords)

    # Calculate distance matrix
    dist_matrix = dist.cdist(ref_coords, work_coords)

    # Sorted distance
    sorted_matrix = np.sort(dist_matrix, axis=0)

    if display:
        plt.figure(figsize=(8.7, 7))
        plt.imshow(sorted_matrix, cmap='RdYlBu_r', interpolation='nearest')
        plt.title('Frame ' + str(frame) + ' to ' + str(ref_frame))
        plt.colorbar()

    # Calculate cost
    cost = np.sum(sorted_matrix[0])

    if verbose:
        print('   cost: ' + str(cost))

    return cost


def transform_frame_simple(transform_matrix, detections_df, frame, ref_frame, debug=False, verbose=False):
    import memotrack.registration
    import numpy as np

    # aa, ab, ac, ba, bb, bc, ca, cb, cc = transform_matrix
    aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd = transform_matrix

    # Create array for transformation
    # matrix = np.array([[aa, ab, ac], [ba, bb, bc], [ca, cb, cc]])
    matrix = np.array([[aa, ab, ac, ad],
                       [ba, bb, bc, bd],
                       [ca, cb, cc, cd],
                       [da, db, dc, dd]])

    # Get the indices we're working on
    work_frame_df = detections_df[detections_df['t'] == frame]
    frame_indices = detections_df[detections_df['t'] == frame].index

    # Apply transformation
    detections_df.loc[frame_indices, ('xreg', 'yreg', 'zreg', 'wreg')] = np.dot(work_frame_df[['x', 'y', 'z', 'w']],
                                                                                matrix)

    # Calculate cost for the registered frame
    cost = memotrack.registration.simple_cost(detections_df, frame, ref_frame, verbose=False, display=False)

    if verbose:
        print ('\nCost: ' + str(np.around(cost, decimals=5)))
        print ('Matrix:\n' + str(np.around(matrix, decimals=5)))

    '''
    # Set the transform matrix as global, for checking later
    global current_matrix
    current_matrix = matrix
    '''

    return cost


def generate_test_matrix(detections_df, matrix_list, middle_frame):
    import memotrack.registration
    import sys
    import numpy as np

    # Uses a sub-list to generate a full list, for testing purpuses only
    # Get total number of frames
    nframes = detections_df['t'].nunique()

    avg_matrix = np.mean(matrix_list, axis=0)

    print ('\nUSING TEST MATRICES !')
    print ('Avg matrix:')
    print (avg_matrix)

    x_translation = avg_matrix[12]
    y_translation = avg_matrix[13]
    z_translation = avg_matrix[14]

    print ('\nX translation: ' + str(x_translation))
    print ('Y translation: ' + str(y_translation))
    print ('Z translation: ' + str(z_translation))

    reverse_matrix = avg_matrix
    reverse_matrix[12] = -reverse_matrix[12]
    reverse_matrix[13] = -reverse_matrix[13]
    reverse_matrix[14] = -reverse_matrix[14]

    print ('\nReverse matrix:')
    print (reverse_matrix)

    # Generate fake list
    full_list = []
    for frame in range(nframes):
        if frame < middle_frame:
            full_list.append(avg_matrix)
        else:
            full_list.append(reverse_matrix)

    return full_list


def align_frames(detections_df, matrix_list, middle_frame, reg_step=10, verbose=False, display=False, debug=False):
    import memotrack.registration
    import sys
    import numpy as np

    # Get total number of frames
    nframes = detections_df['t'].nunique()

    if verbose:
        print ('\nTotal of ' + str(nframes) + ' frames and ' + str(len(matrix_list)) + ' matrices')
        print ('\nAverage transoformation matrix:')
        print (np.mean(matrix_list, axis=0))
        print ('Using as reference frame ' + str(middle_frame))
        print ('\nAligning frames: ')
        sys.stdout.flush()

    for frame in range(nframes):
        if verbose:
            print ('\n[' + str(frame) + ']')
        nsteps = abs(frame - middle_frame)

        for step in range(0, nsteps, reg_step):

            # Get the transform matrix
            if frame < middle_frame:
                aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd = matrix_list[frame + step]
                print (int(frame + step)),
            else:
                aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd = matrix_list[frame - step]
                print (int(frame - step)),

            # Create array for transformation
            matrix = np.array([[aa, ab, ac, ad],
                               [ba, bb, bc, bd],
                               [ca, cb, cc, cd],
                               [da, db, dc, dd]])

            # Get the indices we're working on
            work_frame_df = detections_df[detections_df['t'] == frame]
            frame_indices = detections_df[detections_df['t'] == frame].index

            # Apply transformation
            detections_df.loc[frame_indices, ('xreg', 'yreg', 'zreg', 'wreg')] = np.dot(
                work_frame_df[['xreg', 'yreg', 'zreg', 'wreg']], matrix)

    return detections_df


def pycpd(detections_df, verbose=False, display=True):
    '''
    THIS DOESN'T WORK ! pycpd IS NOT WELL IMPLEMENTED IN 3D !
    :param detections_df:
    :param verbose:
    :param display:
    :return:
    '''
    import memotrack.registration
    import sys
    import numpy as np
    import pycpd
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from functools import partial

    # This is the callback for the cpd class
    def visualize(iteration, error, X, Y):

        if display:
            print ('error: {}'.format(error))
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='#2233FF')
            ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='#22FF33')
            ax.view_init(elev=25, azim=45)
            plt.title('error: {}'.format(error))
            plt.savefig('/projects/memotrack/temp/' + str(iteration) + '.png')

        print ('error: {}'.format(error))

    # Get total number of frames
    nframes = detections_df['t'].nunique()

    # Get the reference frame for the required registration step
    middle_frame = int(nframes / 2.0)
    # middle_frame = 5  # FOR DEBUG

    if verbose:
        print ('\nStarting registration using Coherent Point Drift')
        print ('Using frame {} as reference for registration.'.format(middle_frame))

    # Creating registered coords as copy and reseting index
    if 'xreg' not in detections_df.columns:
        detections_df = detections_df.reset_index()
        detections_df['xreg'] = detections_df['x']
        detections_df['yreg'] = detections_df['y']
        detections_df['zreg'] = detections_df['z']

    # Get the referance timeframe as a numpy array
    ref_df = detections_df[detections_df['t'] == middle_frame]
    ref = ref_df[['x', 'y', 'z']].as_matrix().astype('float')

    for frame in range(nframes):

        if verbose:
            print ('\nFrame: {}'.format(frame))

        # Get current frame as numpy array
        current_df = detections_df[detections_df['t'] == frame]
        current = current_df[['x', 'y', 'z']].as_matrix().astype('float')

        if verbose:
            print ('{} to {}'.format(np.shape(current), np.shape(ref)))

        print ('Current: {} \t Ref: {}'.format(np.average(current), np.average(ref)))

        reg = pycpd.rigid_registration(current, ref, sigma2=0.01)

        orig_z = reg.TY[:, 2]

        # ## Plot before
        # fig = plt.figure(figsize=(20, 20))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(reg.X[:, 0], reg.X[:, 1], reg.X[:, 2], color='#2233FF')
        # ax.scatter(reg.Y[:, 0], reg.Y[:, 1], reg.Y[:, 2], color='#22FF33')
        # ax.view_init(elev=30, azim=45)

        # Register
        reg.register(partial(visualize))

        # ## Plot after
        # fig2 = plt.figure(figsize=(20, 20))
        # ax2 = fig2.add_subplot(111, projection='3d')
        # ax2.scatter(reg.X[:, 0], reg.X[:, 1], reg.X[:, 2], color='#2233FF')
        # ax2.scatter(reg.TY[:, 0], reg.TY[:, 1], reg.TY[:, 2], color='#22FF33')
        # ax2.view_init(elev=30, azim=45)

        print ('Current: {} \t Ref: {}'.format(np.average(reg.X), np.average(reg.TY)))

        new_z = reg.TY[:, 2]

        diff_z = abs(orig_z - new_z)

        print (np.sort(diff_z))

    return detections_df


def cpd(detections_df, verbose=False, display=True):
    import memotrack.registration
    import sys
    import numpy as np
    import cpd
    from cpd.cpd_plot import cpd_plot
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # Get total number of frames
    nframes = detections_df['t'].nunique()

    # Get the reference frame for the required registration step
    middle_frame = int(nframes / 2.0)

    if verbose:
        print ('\n\n\nStarting registration using Coherent Point Drift')
        print ('Using frame {} as reference for registration.'.format(middle_frame))

    # Creating registered coords as copy and reseting index
    if 'xreg' not in detections_df.columns:
        detections_df = detections_df.reset_index()
        detections_df['xreg'] = detections_df['x']
        detections_df['yreg'] = detections_df['y']
        detections_df['zreg'] = detections_df['z']

    # Get the referance timeframe as a numpy array
    ref_df = detections_df[detections_df['t'] == middle_frame]
    # ref = ref_df[['x', 'y', 'z']].as_matrix().astype('float')  # old reference, using original coordinates
    ref = ref_df[['xaff', 'yaff', 'zaff']].as_matrix().astype('float')  # reference chained affine registration

    for frame in range(nframes):

        if verbose:
            print ('\nFrame: {}'.format(frame))

        # Get current frame as numpy array
        current_df = detections_df[detections_df['t'] == frame]
        # current = current_df[['x', 'y', 'z']].as_matrix().astype('float')  # old reference, using original coordinates
        current = current_df[['xaff', 'yaff', 'zaff']].as_matrix().astype('float')  # affine reference

        if verbose:
            print ('{} to {}'.format(np.shape(current), np.shape(ref)))

        T = memotrack.registration.register_nonrigid(ref, current, 0.0, lamb=0.1, beta=30, display=display)

        # Setting coordinates back to dataframe
        detections_df.loc[detections_df['t'] == frame, 'xreg'] = T[:, 0]
        detections_df.loc[detections_df['t'] == frame, 'yreg'] = T[:, 1]
        detections_df.loc[detections_df['t'] == frame, 'zreg'] = T[:, 2]

    return detections_df


def register_nonrigid(x, y, w, lamb=3.0, beta=2.0, max_it=50, verbose=True, display=True):
    """
    Registers Y to X using the Coherent Point Drift algorithm, in non-rigid fashion.
    Note: For affine transformation, t = y+g*wc(* is dot).
    Parameters
    ----------
    x : ndarray
        The static shape that Y will be registered to. Expected array shape is [n_points_x, n_dims]
    y : ndarray
        The moving shape. Expected array shape is [n_points_y, n_dims]. Note that n_dims should be equal for X and Y,
        but n_points does not need to match.
    w : float
        Weight for the outlier suppression. Value is expected to be in range [0.0, 1.0].
    lamb : float, optional
        lamb represents the trade-off between the goodness of maximum likelihood fit and regularization.
        Default value is 3.0.
    beta : float, optional
        beta defines the model of the smoothness regularizer(width of smoothing Gaussian filter in
        equation(20) of the paper).Default value is 2.0.
    max_it : int, optional
        Maximum number of iterations. Used to prevent endless looping when the algorithm does not converge.
        Default value is 150.
    tol : float

    Returns
    -------
    t : ndarray
        The transformed version of y. Output shape is [n_points_y, n_dims].
    """

    import numpy as np
    import numpy.matlib
    import scipy.sparse
    from cpd.cpd_p import cpd_p
    import matplotlib.pyplot as plt

    # Construct G:
    g = y[:, np.newaxis, :] - y
    g = g * g
    g = np.sum(g, 2)
    g = np.exp(-1.0 / (2 * beta * beta) * g)
    [n, d] = x.shape
    [m, d] = y.shape
    t = y
    # initialize sigma^2
    sigma2 = (m * np.trace(np.dot(np.transpose(x), x)) + n * np.trace(np.dot(np.transpose(y), y)) -
              2 * np.dot(sum(x), np.transpose(sum(y)))) / (m * n * d)
    iter = 0

    while (iter < max_it) and (sigma2 > 1.0e-5):
        if verbose and (iter == 0 or iter == max_it):
            print ('error: {}'.format(sigma2)),

        [p1, pt1, px] = cpd_p(x, t, sigma2, w, m, n, d)
        # precompute diag(p)
        dp = scipy.sparse.spdiags(p1.T, 0, m, m)
        # wc is a matrix of coefficients
        wc = np.dot(np.linalg.inv(dp * g + lamb * sigma2 * np.eye(m)), (px - dp * y))
        t = y + np.dot(g, wc)
        Np = np.sum(p1)
        sigma2 = np.abs((np.sum(x * x * np.matlib.repmat(pt1, 1, d)) + np.sum(t * t * np.matlib.repmat(p1, 1, d)) -
                         2 * np.trace(np.dot(px.T, t))) / (Np * d))
        iter += 1
        if verbose and (iter == 0 or iter == max_it):
            print ('\tcentroid: x:{0:.3f} y:{1:.3f} z:{2:.3f}'.format(np.average(t[:, 0]), np.average(t[:, 1]),
                                                                      np.average(t[:, 2])))

        if display:
            fig = plt.figure(figsize=(14, 14))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], color='#2299FF')
            ax.scatter(t[:, 0], t[:, 1], t[:, 2], color='#222222', marker='x', s=50)
            ax.view_init(elev=25, azim=45)
            plt.title('error: {}'.format(sigma2))
            plt.savefig('/projects/memotrack/temp/' + str(iter) + '.png')
            plt.close()

    return t
