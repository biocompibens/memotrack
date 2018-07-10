# Imports that must be at the beginning
from ipywidgets import interact, interactive, fixed


# Generate random colormap
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True, condor=False,
              accent_label=False, rand_state=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np
    from matplotlib import pyplot as plt

    if condor:
        plt.ioff()

    nlabels = int(nlabels)
    if type not in ('bright', 'soft', 'super_bright'):
        print ('Please choose "bright", "soft" or "super_bright" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    if rand_state:
        np.random.seed(seed=rand_state)

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.75, high=1)) for i in xrange(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        if accent_label:
            for label in range(nlabels):
                if label is not accent_label:
                    newR = randRGBcolors[label][0] / 5.0
                    newG = randRGBcolors[label][1] / 5.0
                    newB = randRGBcolors[label][2] / 5.0
                    randRGBcolors[label] = [newR, newG, newB]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    if type == 'super_bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.9, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in xrange(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        if accent_label:
            for label in range(nlabels):
                if label is not accent_label:
                    newR = randRGBcolors[label][0] / 5.0
                    newG = randRGBcolors[label][1] / 5.0
                    newB = randRGBcolors[label][2] / 5.0
                    randRGBcolors[label] = [newR, newG, newB]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in xrange(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        if accent_label:
            for label in range(nlabels):
                if label is not accent_label:
                    newR = randRGBcolors[label][0] / 5.0
                    newG = randRGBcolors[label][1] / 5.0
                    newB = randRGBcolors[label][2] / 5.0
                    randRGBcolors[label] = [newR, newG, newB]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


# Basic visualization of ZYX image
def basic(img, z, z_slice, t=0, auto_scale=True, vmin=0, vmax=65535, cmap='CMRmap'):
    """
    Basic display of the stack. Doesn't work with time yet.
    :param img: Stack in ZYX
    :param z: Slice to display
    :param scale_factor: Adjusts image size

    """

    from matplotlib import pyplot as plt
    import numpy as np

    # Check for time
    if len(np.shape(img)) == 4:
        time = True
    elif len(np.shape(img)) == 3:
        time = False
    else:
        print ('Are you sure this is a 3D or 4D image ?')
        return

    # Get image size
    if time:
        zsize = np.shape(img[t])[0]
        ysize = np.shape(img[t])[1]
        xsize = np.shape(img[t])[2]
    else:
        zsize = np.shape(img)[0]
        ysize = np.shape(img)[1]
        xsize = np.shape(img)[2]

    # Set parameters
    plt.rcParams['image.interpolation'] = 'none'
    plt.rcParams['figure.figsize'] = 15, 8
    plt.set_cmap(cmap)

    # Create XY image
    if time:
        xy_img = img[t][z]
    else:
        xy_img = img[z]

    # Create YZ image
    if time:
        yz_img = np.zeros([ysize, zsize])
        for y_pos in range(ysize):
            for z_pos in range(zsize):
                yz_img[y_pos, z_pos] = img[t][z_pos][y_pos][z_slice]
    else:
        yz_img = np.zeros([ysize, zsize])
        for y_pos in range(ysize):
            for z_pos in range(zsize):
                yz_img[y_pos, z_pos] = img[z_pos][y_pos][z_slice]

    # Plot
    plt.subplot(1, 2, 1)
    if auto_scale:
        plt.imshow(xy_img)
    else:
        plt.imshow(xy_img, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    if auto_scale:
        plt.imshow(yz_img)
    else:
        plt.imshow(yz_img, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar()


# Basic plot with interact
def basic_interact(img):
    """
    Calls memotrack.display.basic on interactive mode
    :param img:
    :return:
    """
    import memotrack.display
    import numpy as np

    # Image with Time
    if len(np.shape(img)) == 4:
        interact(memotrack.display.basic,
                 img=fixed(img),
                 t=(0, (len(img) - 1)),
                 z=(0, (len(img[0]) - 1)),
                 z_slice=(0, (len(img[0][0][0]) - 1)),
                 auto_scale=True,
                 vmin=(0, 65535),
                 vmax=(0, 65535))

    # Image without Time
    if len(np.shape(img)) == 3:
        interact(memotrack.display.basic,
                 img=fixed(img),
                 z=(0, (len(img) - 1)),
                 z_slice=(0, (len(img[0][0]) - 1)),
                 auto_scale=True,
                 vmin=(0, 65535),
                 vmax=(0, 65535))


# Plot from pandas dataframe
def plot_from_df(detections_df, new_cmap=False, size=60, elev=15, azim=45, lim=430, zstart=-7, ystart=-5, xstart=-6,
                 crop_data=True, one_frame=False, frame=0, time_color=False, intensity=False, auto_fit=True,
                 borders=False, registered_coords='cpd', title=False, save=False, HighDPI=False, zlim=False,
                 verbose=False, line_tracks=False, axis_off=False):
    """
    Used to plot the datapoints from the detection in 3D
    If the dataframe have the column label it uses colors from new_cmap for each label.
    Otherwise it understand that the data wasnt labeled yet, and uses color_map to label the time information

    set for zoom:
    detections_df, new_cmap, size=30, elev=15, azim=30, lim=120, zstart=70, ystart=70, xstart=120,
                 one_frame=False, frame=0, crop_data=True

    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import memotrack.display
    import matplotlib

    nframes_original = detections_df['t'].max()
    if intensity:
        max_norm_intensity = detections_df['norm_intensity'].max()
        min_norm_intensity = detections_df['norm_intensity'].min()

    if title is False:
        title = ''
    if registered_coords is 'cpd':
        x_coord = 'xreg'
        y_coord = 'yreg'
        z_coord = 'zreg'

    elif registered_coords is 'affine':
        x_coord = 'xaff'
        y_coord = 'yaff'
        z_coord = 'zaff'

    elif registered_coords is 'original':
        x_coord = 'xorig'
        y_coord = 'yorig'
        z_coord = 'zorig'

    elif registered_coords is 'smooth':
        x_coord = 'xsmooth'
        y_coord = 'ysmooth'
        z_coord = 'zsmooth'

    elif registered_coords is 'back':
        x_coord = 'xreproject'
        y_coord = 'yreproject'
        z_coord = 'zreproject'

    elif registered_coords is 'xyz':
        x_coord = 'x'
        y_coord = 'y'
        z_coord = 'z'

    else:
        print ('Didnt understood coords label, using xyz coords')
        x_coord = 'x'
        y_coord = 'y'
        z_coord = 'z'

    # Setting size for border
    if borders:
        if borders > 1:
            width = borders
        else:
            width = 1
    else:
        width = 0

    if HighDPI:
        plt.rcParams['figure.dpi'] = 1200
        plt.rcParams['savefig.dpi'] = 1200

    if auto_fit:
        border = 10
        # Getting the biggest and lowest values for each dimension, to scale the axis accordingly
        xstart = detections_df[x_coord].min() - border
        xend = detections_df[x_coord].max() + border

        ystart = detections_df[y_coord].min() - border
        yend = detections_df[y_coord].max() + border

        zstart = detections_df[z_coord].min() - border
        zend = detections_df[z_coord].max() + border

        lim = (max([xend, yend, zend]) - min([xstart, ystart, zstart]))

        if verbose:
            print ('xstart: ' + str(xstart))
            print ('ystart: ' + str(ystart))
            print ('zstart: ' + str(zstart))
            print ('lim: ' + str(lim))

    # We need to get here the number of labels, before cropping the data in case of plotting only one frame
    if 'label' in detections_df:
        nlabels = detections_df['label'].nunique()
        # This is a fix for the cases where only noise (Label = -1) exists:
        if nlabels < 0:
            nlabels = 0
    # not sure if I really need this
    else:
        nlabels = len(detections_df)
    if verbose:
        print (str(nlabels) + ' clusters')

    # Get max intensity before using only one frame data
    if 'intensity' in detections_df:
        max_intensity = float(detections_df['intensity'].max())

    if one_frame:
        detections_df = detections_df[detections_df['t'] == frame].copy(deep=True)

    nframes = detections_df['t'].max()

    # Start axes
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev, azim)
    if axis_off:
        ax.set_axis_off()

    # Cropping the data avoids plotting outside the axis, and make the process much faster
    if crop_data:
        detections_df = detections_df[detections_df[x_coord] < (lim + xstart)]
        detections_df = detections_df[detections_df[x_coord] > (xstart)]

        detections_df = detections_df[detections_df[y_coord] < (lim + ystart)]
        detections_df = detections_df[detections_df[y_coord] > (ystart)]

        detections_df = detections_df[detections_df[z_coord] < (lim + zstart)]
        detections_df = detections_df[detections_df[z_coord] > (zstart)]

    # Plot data. If no label is found, use colormap for frames

    # Check if we need to generate a colormap
    if new_cmap is False:
        new_cmap = memotrack.display.rand_cmap(nlabels, type='bright', first_color_black=True, verbose=False)

    # Plot using intensity information
    if 'label' and 'intensity' in detections_df and time_color is False and intensity is 'norm':
        color_map = 'viridis'

        # Averaging each label
        temp_df = detections_df.groupby(['label']).min()
        temp2_df = detections_df.groupby(['label']).mean()

        intensity_list = temp2_df['intensity'] / temp_df['intensity']
        max_intensity = intensity_list.max()
        min_intensity = intensity_list.min()
        mean_intensity = intensity_list.mean()

        # norm_intensity_list = intensity_list - min_intensity
        # norm_intensity_list = norm_intensity_list / norm_intensity_list.max()
        norm_intensity_list = intensity_list

        ax.scatter(temp2_df[x_coord], temp2_df[y_coord], temp2_df[z_coord],
                   s=((norm_intensity_list ** 2) * size * 10000), c=norm_intensity_list,
                   cmap=color_map, linewidths=width)

    # Plot using intensity information
    if 'label' and 'intensity' in detections_df and time_color is False and intensity is 'df':
        color_map = 'viridis'

        temp_df = detections_df.groupby(['label']).mean()
        temp_df2 = detections_df.groupby(['label']).max()
        norm_intensity_list = temp_df2['norm_intensity']
        # norm_intensity_list = [(x / max_norm_intensity) for x in norm_intensity_list]

        # For varying size
        #ax.scatter(temp_df[x_coord], temp_df[y_coord], temp_df[z_coord],
        #           s=[(((x - min_norm_intensity) / max_norm_intensity) ** 3) * size for x in norm_intensity_list],
        #           depthshade=0,
        #           c=norm_intensity_list,
        #           cmap=color_map, linewidths=width, vmin=0, vmax=max_norm_intensity)

        # For constant size
        ax.scatter(temp_df[x_coord], temp_df[y_coord], temp_df[z_coord],
                   s=size, depthshade=0, c=norm_intensity_list, cmap=color_map, linewidths=width,
                   vmin=0, vmax=max_norm_intensity)

    # Plot with colors for labels
    if 'label' in detections_df and time_color is False and intensity is False:

        ax.scatter(detections_df[x_coord], detections_df[y_coord], detections_df[z_coord], s=size,
                   c=detections_df['label'],
                   cmap=new_cmap, vmin=0, vmax=nlabels, linewidths=width, zorder=2)

        if line_tracks:

            xvalues = detections_df[x_coord].get_values()
            yvalues = detections_df[y_coord].get_values()
            zvalues = detections_df[z_coord].get_values()
            xyz_coords = [xvalues, yvalues, zvalues]
            # nvalues = len(xvalues)
            time_cmap = matplotlib.cm.get_cmap('gist_rainbow')
            for pos in range(int(nframes)):
                x_t1 = xvalues[pos]
                x_t2 = xvalues[pos + 1]
                xpair = [x_t1, x_t2]

                y_t1 = yvalues[pos]
                y_t2 = yvalues[pos + 1]
                ypair = [y_t1, y_t2]

                z_t1 = zvalues[pos]
                z_t2 = zvalues[pos + 1]
                zpair = [z_t1, z_t2]

                ax.plot(xpair, ypair, zpair, color=time_cmap(pos / nframes), alpha=0.5, lw=1.5, zorder=1)

    # Plot using color code for time
    if time_color is True:
        color_map = 'viridis_r'
        # Plot with different colors for each frame
        if one_frame:
            cmap = matplotlib.cm.get_cmap(color_map)
            color = cmap(float(frame) / nframes_original)
            ax.scatter(detections_df[x_coord], detections_df[y_coord], detections_df[z_coord], s=size,
                       facecolor=color, edgecolor='#333333', linewidths=width)
        else:
            ax.scatter(detections_df[x_coord], detections_df[y_coord], detections_df[z_coord], s=size,
                       c=detections_df['t'], cmap=color_map, vmin=0, vmax=nframes, linewidths=0, alpha=0.8)

    ax.set_xlim3d(xstart, lim + xstart)
    ax.set_ylim3d(ystart, lim + ystart)

    if zlim:
        ax.set_zlim3d(zstart, zlim + zstart)
    else:
        ax.set_zlim3d(zstart + lim,  zstart)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set title
    if registered_coords in ['cpd', 'affine', 'registered']:
        title += ' | Registered coordinates'
    elif registered_coords == 'back':
        title += ' | Back-projected coordinates'
    else:
        title += ' | Original coordinates'

    plt.title(title)

    if save:
        fig.savefig(save, bbox_inches='tight')
        plt.close()


# Plot from pandas dataframe with interact
def plot_from_df_interact(detections_df, new_cmap):
    import memotrack.display

    if detections_df['t'].max() > 0:
        interact(memotrack.display.plot_from_df,
                 detections_df=fixed(detections_df),
                 new_cmap=fixed(new_cmap),
                 elev=(0, 90),
                 azim=(0, 180),
                 lim=(10, 2000),
                 zstart=(-500, 500),
                 ystart=(-500, 500),
                 xstart=(-500, 500),
                 size=(1, 300),
                 first_frame=False,
                 one_frame=True,
                 frame=(0, detections_df['t'].max(), 1),
                 intensity='norm',
                 title='')
    else:
        interact(memotrack.display.plot_from_df,
                 detections_df=fixed(detections_df),
                 new_cmap=fixed(new_cmap),
                 elev=(0, 90),
                 azim=(0, 180),
                 lim=(10, 2000),
                 zstart=(-500, 500),
                 ystart=(-500, 500),
                 xstart=(-500, 500),
                 size=(1, 300),
                 first_frame=False,
                 one_frame=True,
                 frame=0,
                 intensity='norm',
                 title='')


# Write frames to disk
def write_movie(detections_df, new_cmap, path='/projects/memotrack/temp/anim/', rotation=True, degrees=30,
                size=100, elev=15, azim=15, lim=600, zstart=-130, ystart=-50, xstart=-147,
                crop_data=False, one_frame=True, borders=True, time_color=False):
    """
    Writes 3D plot frames to disk
    """
    from matplotlib import pyplot as plt

    nframes = detections_df['t'].max() + 1

    # Calculate rotation step
    if rotation:
        rotation_step = degrees / nframes

    # Start to create plots
    print('Writing frames:'),
    for frame in range(int(nframes)):
        print('[' + str(frame) + ']'),
        if rotation:
            azim += rotation_step

        # Create plot
        plot_from_df(detections_df, new_cmap, size=size, elev=elev, azim=azim, lim=lim, zstart=zstart,
                     ystart=ystart, xstart=xstart, frame=frame, time_color=time_color,
                     crop_data=crop_data, one_frame=one_frame, auto_fit=False, borders=borders)

        # Write to disk
        filename = (str(frame) + '.png')
        plt.savefig(path + filename, bbox_inches='tight')
        plt.close()


# Create detection matrix
def detection_matrix(detections_df, verbose=True, color_scale=False, title=False):
    """
    Creates the detection matrix, to analyse the tracking
    :param detections_df: labeled pandas dataframe
    :param verbose: Print process
    :return:
    """
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import colors, colorbar

    nframes = detections_df['t'].nunique()
    nlabels = detections_df['label'].nunique()

    matrix = np.zeros([nlabels, nframes])

    # Add to the matrix
    for frame in range(int(nframes)):
        temp_df = detections_df[detections_df['t'] == frame].copy(deep=True)
        for label in temp_df['label']:
            if label != -1:
                matrix[int(label), int(frame)] += 1
    matrix_dim = np.shape(matrix)

    if verbose:
        print('Number of labels: ' + str(matrix_dim[0]))
        print('Number of frames: ' + str(matrix_dim[1]))

    # Setting the figure size proportional to the matrix size
    fig, ax = plt.subplots(1, 1, figsize=((float(matrix_dim[1]) / 100) * 15, matrix_dim[0] / 8))

    # Define the colormap
    cmap = plt.get_cmap('YlOrRd', 10)  # 10 discrete colors

    # Extract all colors from the map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # Force the first color entry to be grey
    cmaplist[0] = (0.3, 0.3, 0.3, 1.0)
    # Force second color to green
    cmaplist[1] = (0.3, 0.85, 0.15, 1.0)
    # Force last color to magenta
    cmaplist[9] = (1, 0, 1, 1)

    # Create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # Define the bins and normalize
    bounds = np.linspace(0, 10, 11)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Display Matrix
    ax.pcolormesh(matrix, edgecolors='#333333', linewidths=1, cmap=cmap, norm=norm)
    plt.ylim((0, matrix_dim[0]))
    plt.xlim((0, matrix_dim[1]))
    plt.gca().invert_yaxis()

    # Adjustments
    plt.yticks(np.arange(0, matrix_dim[0], 10))
    plt.xlabel('Time frames', fontsize=18)
    plt.ylabel('Labels', fontsize=18)
    # Set title
    plt.title(title)

    # Diplay Cmap
    if color_scale:
        ax2 = fig.add_axes([0.125, 0.903, 0.77, 0.00115])  # [left, bottom, width, height]
        colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds,
                              extend='max', format='%1i', orientation=u'horizontal')


def plot_error_list(error_list, plot_sum=True, lowest_error=True, lowest_error_pos=0, img_size=6.0, plot_legend=True,
                    legends=[]):
    """
    Plot erros, as from DBSCAN clustering.
    :param error_list:
    :param plot_sum:
    :param lowest_error:
    :param lowest_error_pos:
    :param img_size:
    :param plot_legend:
    :param legends:
    :return:
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import memotrack.display
    from itertools import cycle

    # Color list to cycle on plot
    color_list = cycle(['#18BF00', '#AD00FF', '#FF3600'])

    # Image size
    plt.rcParams['figure.figsize'] = img_size * 1.78, img_size

    # transpose error list
    t_errors = np.transpose(error_list)
    n_errors = len(t_errors)

    # Create fig
    fig, ax = plt.subplots()

    legend_col = n_errors

    for error in range(n_errors):
        ax.plot(t_errors[error], linestyle='--', color=color_list.next(), linewidth=2)

    if plot_sum:
        ax.plot(np.sum(error_list, axis=1), color='#00DDFF', linewidth=2)
        legend_col += 1

    if lowest_error:
        ax.scatter(lowest_error_pos, np.sum(error_list, axis=1)[lowest_error_pos], s=80, c='#FF009D', zorder=3l,
                   linewidths=2, edgecolor='#00DDFF')
        legend_col += 2

    if plot_legend:
        ax.legend(legends, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=legend_col,
                  scatterpoints=1)

    # Plot corrections
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='on')
    plt.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')
    plt.xlim([0, np.shape(error_list)[0] - 1])
    plt.ylim([0, (np.max(np.sum(error_list, axis=1)))])
    plt.yticks(np.arange(0, np.max(np.sum(error_list, axis=1)), 0.5))


def plot_1D_signals(detections_df, normalize=False, accent=False, only_responsive=False,
                    stim_frame=25, stim_duration=10, cmap=False, HighDPI=False, only_positive=False,
                    title=False, save=False, smooth=False, ymax=False, condor=False, RGBA=False):
    import matplotlib
    if condor:
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import memotrack.display
    import scipy.ndimage.filters as filters

    plt.ioff()
    original_nlabels = float(detections_df['label'].nunique())
    original_nframes = int(detections_df['t'].nunique())

    if only_responsive:
        detections_df = detections_df[detections_df['responsive'] == True].copy(deep=True)

    # From where to get the intensity values

    # Using the neighbour normalization from the df
    if normalize in ['df', 'normalized', 'norm_intensity']:
        intensity_col = 'norm_intensity'

    # Using the values from the bandpass filter
    elif normalize == 'filtered':
        intensity_col = 'filtered'
    elif normalize == 'filter':
        intensity_col = 'filtered'

    # Using the raw intensity values
    elif normalize == 'False':
        intensity_col = 'intensity'
    elif not normalize:
        intensity_col = 'intensity'
    elif normalize == 'raw':
        intensity_col = 'raw_intensity'

    # In case of doubt, go with the raw values
    else:
        print ('WARNING: Unknown parameter for normalize (' + str(normalize) + ') , using raw intensitites')
        intensity_col = 'intensity'

    # Image size
    img_size = 3
    plt.rcParams['figure.figsize'] = img_size * 5.78, img_size

    if HighDPI:
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150

    # Data info

    nlabels = int(detections_df['label'].nunique())

    nframes = int(detections_df['t'].nunique())

    average_intensity = detections_df[intensity_col].mean()

    matrix = []
    # Get label list
    labels_list = detections_df.label.unique()
    # Normalization loop
    for label in labels_list:
        df = detections_df[detections_df['label'] == label].copy(deep=True)
        temp = df.as_matrix(columns=[intensity_col])
        temp = temp.T[0]

        # Normalizing
        if normalize == 'average':
            f0 = temp[0:stim_frame[0]]
            temp = temp / np.average(f0)
        if normalize == 'percentile' or normalize == 'df':
            sorted_list = np.sort(temp)
            percent_avg = np.average(sorted_list[0:(nframes * 0.05)])
            temp = temp / percent_avg
        if normalize == 'stdev':
            stdev_intensities = np.std(temp)
            avg_intensities = np.mean(temp)
            temp = temp / (avg_intensities - stdev_intensities)

        if normalize == 'filtered' or normalize == 'filter':
            if only_positive:
                temp[temp < 0] = 0.00

        if smooth:
            temp = filters.gaussian_filter1d(temp, smooth)

        matrix.append(temp)

    # Transpose to get x axis as time
    matrix = np.asarray(matrix)
    matrix = matrix.T
    if len(matrix) > 0:  # fix for the bug in case of zero lines
        matrix = np.swapaxes(matrix, 0, 1)

    # Plotting
    fig, ax = plt.subplots()

    if cmap is False:
        new_cmap = memotrack.display.rand_cmap(nlabels, type='bright', first_color_black=False, verbose=False)
        colors = new_cmap(np.linspace(0, 1, nlabels))
    else:
        colors = []
        for label in labels_list:
            colors.append(cmap(label / original_nlabels))
            # colors = cmap(np.linspace(0, 1, nlabels))

    # get std deviations, to use with accent
    stdev_list = []
    intensity_max_list = []
    intensity_min_list = []

    for i in range(len(matrix)):
        stdev_list.append(np.std(matrix[i]))
        intensity_max_list.append(np.max(matrix[i]))
        intensity_min_list.append(np.min(matrix[i]))

    if len(matrix) > 0:  # fix for the bug in case of zero lines
        minimum_stdev = min(stdev_list)
        stdev_range = stdev_list - minimum_stdev
        stdev_range /= max(stdev_range)

    # Create X axis
    x = range(nframes)

    # Plotting every line
    for i in range(len(matrix)):
        if sum(abs(matrix[i])) > 0:
            zorder = 1
            linewidth = 1.5
            if accent:
                # alpha = ((intensity_max_list[i] - intensity_min_list[i]) / intensity_max_list[i])
                alpha = stdev_range[i]
            else:
                alpha = 0.8

            if only_responsive == 'comparison':
                if sum(detections_df[detections_df['label'] == i].super_responsive) <= 0:
                    spice = (np.random.random())
                    gray = (0.5 + spice) / 2
                    colors[i] = (gray, gray, gray, 1)
                    alpha = 0.5
                    zorder = 0
                    linewidth = 1

            if RGBA:
                plt.plot(x, matrix[i], linewidth=linewidth, color=colors[i], zorder=zorder)
            else:
                plt.plot(x, matrix[i], alpha=alpha, linewidth=linewidth, color=colors[i], zorder=zorder)


    # Plot display fixes
    plt.xlabel('Time frames')
    plt.xlim([0, original_nframes - 1])
    if ymax:
        plt.ylim([0, ymax])  # To force limits of Y axis
    plot_height = ax.get_ylim()[1]

    if only_positive:
        plt.ylim([0.0, plot_height])

    if normalize:
        plt.ylabel('Normalized intensity')
    else:
        plt.ylabel('Signal intensity')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Draw rectangle on stimulus duration
    for position in stim_frame:
        ax.add_patch(patches.Rectangle([position, 0], stim_duration, plot_height, alpha=0.5, fill=False,
                                       color='#AAAAAA', hatch='//', linewidth=0))

    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='on')
    plt.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')

    # Set title
    if normalize == 'filtered':
        title += ' | Filtered'
    if normalize == 'df':
        title += ' | Neighbour normalization'
    if only_responsive == 'all':
        title += ' | Only responsive'
    if only_responsive == 'stim':
        title += ' | Only responsive inside window'
    if smooth:
        title += (' | GaussFilter ' + str(smooth))
    title += (' | ' + str(nlabels) + ' neurons')
    plt.title(title)

    # Check if we have the data quality and plot

    if 'Q' in detections_df:
        ax2 = ax.twinx()
        temp_df = detections_df.groupby(by='t').mean()
        ax2.plot(temp_df['Q'], alpha=0.5, lw=2, linestyle=':', color='#AA1030')
        ax2.set_ylim(0, 1.005)

        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        plt.ylabel('Signal quality')

    # set ticks
    plt.xticks(np.arange(0, original_nframes + 1, 10))

    if save:
        fig.savefig(save, bbox_inches='tight')
        plt.close()


def signals(detections_df, block_size=40, stim_start=10, stim_duration=5,
            only_responsive=False, only_positive=True, save=False, condor=False,
            intensity='raw', cmap=False, ymax='auto', stim_sequence=False, empty_plot=False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import memotrack.display
    import math
    import scipy.ndimage.filters as filters

    if stim_sequence:
        stim_list = []
        for stim in stim_sequence:
            if stim == 'S':
                stim_list.append('nothing')
            elif stim == 'A':
                stim_list.append('air')
            elif stim == 'O':
                stim_list.append('oct')
            elif stim == 'M':
                stim_list.append('mch')
            else:
                stim_list.append('unknown')
                print ('WARNING: Unknown stimulus in stim_sequence!')
                print ('please use S, A, O or M')

    if condor:
        plt.ioff()
    original_nlabels = float(detections_df['label'].nunique())
    original_nframes = int(detections_df['t'].nunique())

    if only_responsive in ('only', 'exclude'):
        detections_df = detections_df[detections_df['responsive'] == True].copy(deep=True)

    # Selecting signal to use
    if intensity == 'raw':
        intensity_col = 'intensity'

    elif intensity in ('norm', 'normalized', 'norm_intensity'):
        intensity_col = 'norm_intensity'

    elif intensity in ('filter', 'filtered'):
        intensity_col = 'filtered'

    elif intensity in ('original', 'old'):
        intensity_col = 'raw_intensity'

    # In case of doubt, go with the raw values
    else:
        print ('WARNING: Unknown parameter for intensity (' + str(intensity) + ') , using raw intensitites')
        print ('(try using "raw", "normalized" or "filtered")')
        intensity_col = 'intensity'

    nblocks = int(math.floor(float(original_nframes) / block_size))

    # Image size
    img_width = 7
    img_height = 2.2 * nblocks
    plt.rcParams['figure.figsize'] = img_width, img_height

    # Start figure
    fig, ax = plt.subplots(nblocks, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.3)
    # Data info
    nlabels = int(detections_df['label'].nunique())
    nframes = int(detections_df['t'].nunique())

    if ymax == 'auto':
        ymax = 0.01 + max(detections_df.as_matrix(columns=[intensity_col]))  # Adding 0.01 because of line width

    for block in range(nblocks):

        # Create signal matrix
        matrix = []
        # Get label list
        labels_list = detections_df.label.unique()

        crop_min = (block * block_size)
        crop_max = (block + 1) * block_size

        print (crop_min, crop_max)
        block_df = detections_df[(detections_df['t'] > crop_min) & (detections_df['t'] <= crop_max)].copy(deep=True)

        for label in labels_list:
            df = block_df[block_df['label'] == label].copy(deep=True)
            temp = df.as_matrix(columns=[intensity_col])
            temp = temp.T[0]
            if intensity in ('filtered', 'filter'):
                if only_positive:
                    temp[temp < 0] = 0.00
            matrix.append(temp)

        # Transpose to get x axis as time
        matrix = np.asarray(matrix)
        matrix = matrix.T
        if len(matrix) > 0:  # fix for the bug in case of zero lines
            matrix = np.swapaxes(matrix, 0, 1)

        if cmap is False:
            new_cmap = memotrack.display.rand_cmap(nlabels, type='bright', first_color_black=False, verbose=False)
            colors = new_cmap(np.linspace(0, 1, nlabels))
        else:
            colors = []
            for label in labels_list:
                colors.append(cmap(label / original_nlabels))
        # Create X axis
        x = range(len(matrix[0]))

        # Plotting every line
        for i in range(len(matrix)):
            if sum(abs(matrix[i])) > 0:
                zorder = 1
                linewidth = 1.5
                alpha = 0.8
                if empty_plot:
                    alpha = 0.0

                if only_responsive == 'comparison':
                    if sum(detections_df[detections_df['label'] == i].responsive) <= 0:
                        spice = (np.random.random())
                        gray = (0.5 + spice) / 2
                        colors[i] = (gray, gray, gray, 1)
                        alpha = 0.3
                        if empty_plot:
                            alpha = 0
                        zorder = 0
                        linewidth = 1

                ax[block].plot(x, matrix[i], alpha=alpha, linewidth=linewidth, color=colors[i], zorder=zorder)
                # ax[block].set_ylim(0, ymax)
        # Plot display fixes
        # plt.xlabel('Time frames')
        # plt.xlim([0, block_size+1])
        if ymax:
            ax[block].set_ylim([0, ymax])  # To force limits of Y axis

        plot_height = plt.gca().get_ylim()[1]

        # Creating stim window
        if stim_sequence[block] == 'O':
            ax[block].add_patch(patches.Rectangle([stim_start, 0], stim_duration, ymax, alpha=0.1,
                                                  color='#FF5722', linewidth=0, fill=True))
        elif stim_sequence[block] == 'M':
            ax[block].add_patch(patches.Rectangle([stim_start, 0], stim_duration, ymax, alpha=0.1,
                                                  color='#03A9F4', linewidth=0, fill=True))
        else:
            ax[block].add_patch(patches.Rectangle([stim_start, 0], stim_duration, ymax, alpha=0.5,
                                                  color='#777777', linewidth=0, hatch='//', fill=False))

        ax[block].set_ylabel(stim_list[block])
        ax[block].spines['top'].set_visible(False)
        ax[block].spines['right'].set_visible(False)
        ax[block].spines['bottom'].set_visible(False)
        ax[block].tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='on')
        ax[block].tick_params(axis='y', which='both', left='on', right='off', labelleft='on')

        if 'Q' in block_df:
            ax2 = ax[block].twinx()
            temp_df = block_df.groupby(by='t').mean()
            ax2.plot(x, temp_df['Q'], alpha=0.5, lw=2, linestyle=':', color='#AA1030')
            ax2.set_ylim(0, 1.00)

            ax2.spines['top'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)

            ax2.set_ylabel('quality')

        # plt.tight_layout()

        if save:
            fig.savefig(save, bbox_inches='tight', dpi=95)
            plt.close()


def signal_matrix(detections_df, normalized=False, title=False):
    import numpy as np
    import matplotlib.pyplot as plt

    nframes = int(detections_df['t'].nunique())
    nlabels = int(detections_df['label'].nunique())

    matrix = []
    for label in range(nlabels):
        df = detections_df[detections_df['label'] == label].copy(deep=True)
        if normalized:
            temp = df.as_matrix(columns=['norm_intensity'])
        else:
            temp = df.as_matrix(columns=['intensity'])

        temp = temp.T[0]

        matrix.append(temp)

    matrix = np.asarray(matrix)
    matrix = matrix[matrix.sum(axis=1).argsort()]
    matrix[:] = matrix[::-1]

    # Display Matrix
    fig, ax = plt.subplots(1, 1, figsize=(15, (nlabels / 8)))
    ax.pcolormesh(matrix, edgecolors='k', linewidths=1, cmap='winter')
    plt.ylim(0, nlabels)
    plt.xlim(0, nframes)
    plt.gca().invert_yaxis()
    # Set title
    plt.title(title)


def plot1Dline(data, x_label='', y_label='', xticks='auto', img_size=8, linewidth=2, color='#22AAFF', title='',
               ylim=False, xlim=False, show_mean=False, alpha=0.8, annotate=False, x_spine=True, data_label=False):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams['figure.figsize'] = img_size * 1.78, img_size

    fig, ax = plt.subplots()

    counter = 0
    if type(data) == list:
        colors = iter(color)
        labels = iter(data_label)
        for data_to_plot in data:
            data_size = len(data_to_plot)
            plt.plot(data_to_plot, alpha=alpha, linewidth=linewidth, color=next(colors), label=next(labels))
            if annotate:
                data_to_annotate = data_to_plot[15:]
                x_pos = np.argmax(data_to_annotate) + (np.random.random() * 2 - 0.5)
                y_pos = np.max(data_to_annotate)
                plt.annotate(str(counter), xy=(x_pos, y_pos), size=9)
                counter += 1
        plt.legend(frameon=False)

    else:
        data_size = len(data)
        plt.plot(data, alpha=alpha, linewidth=linewidth, color=color)

    if show_mean:
        mean_data = np.mean(data, axis=0)
        plt.plot(mean_data, alpha=0.9, linestyle='--', linewidth=3, color=color)

    # Plot fixes
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # Set title
    plt.title(title)

    if xticks != 'auto':
        if type(xticks[0]) == str:
            plt.xticks(range(data_size), xticks, rotation='vertical')
        else:
            plt.xticks(range(data_size), range(xticks[0], xticks[1]))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if x_spine is not True:
        ax.spines['bottom'].set_visible(False)

    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='on')
    plt.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')

    # To fix the axis size
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)


def volume_user_interface(img, alpha_order=5, thresh=0.5):
    """
    This function is from when I've started with vtk. Later I need to integrate the interactive window
    with the global funtion and excludethis one
    :param img:
    :param alpha_order:
    :param thresh:
    :return:
    """
    import vtk
    import numpy as np

    # Check if we have 8bit data
    if img.dtype != 'uint8':
        print ('Volume visualization needs 8-bit image !')
        return

    # Get info about the image
    SizeZ = np.shape(img)[0]
    SizeY = np.shape(img)[1]
    SizeX = np.shape(img)[2]

    # Put the threshold in a 8bit range
    thresh *= 255

    # From numpy to vtk image
    dataImporter = vtk.vtkImageImport()

    # The previously created array is converted to a string of chars and imported.
    data_string = img.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))

    # Telling vtk we have uint8
    dataImporter.SetDataScalarTypeToUnsignedChar()

    # Telling vtk we have only one channel
    dataImporter.SetNumberOfScalarComponents(1)

    # Dimensions of data
    dataImporter.SetDataExtent(0, SizeX - 1, 0, SizeY - 1, 0, SizeZ - 1)
    dataImporter.SetWholeExtent(0, SizeX - 1, 0, SizeY - 1, 0, SizeZ - 1)

    # Getting list of alpha for all possible intensities
    alpha_list = vtk.vtkPiecewiseFunction()
    for i in range(0, 255):
        if i > thresh:
            alpha_list.AddPoint(i, (i / 255.0) ** alpha_order)
        else:
            alpha_list.AddPoint(i, 0)

    # Getting color for all possible intensities
    color_list = vtk.vtkColorTransferFunction()
    for i in range(0, 255):
        red = 0
        green = i
        #blue = 1 - (i / 255.0)
        blue = 0
        color_list.AddRGBPoint(i, red, green, blue)

    # Set the alpha and color
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(color_list)
    volumeProperty.SetScalarOpacity(alpha_list)

    # This class describes how the volume is rendered (through ray tracing).
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()

    # Create volume.
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    # Declare volume and its proprieties
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    # Initialize renderer and window
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)

    # Add the volume to the renderer
    renderer.AddVolume(volume)

    # Create and set camera
    camera = vtk.vtkCamera()
    renderer.SetActiveCamera(camera)

    # Background color
    renderer.SetBackground(0.15, 0.15, 0.15)
    # Window size
    renderWin.SetSize(800, 800)

    # Function to quit
    def exitCheck(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)

    # Tell the application to use the function as an exit check.
    renderWin.AddObserver("AbortCheckEvent", exitCheck)

    renderInteractor.Initialize()

    # Start renderer and interactor
    renderWin.Render()
    renderInteractor.Start()

    # print camera position
    print ('Focal point: ' + str(camera.GetFocalPoint()))
    print ('Position: ' + str(camera.GetPosition()))
    print ('View up: ' + str(camera.GetViewUp()))
    print ('Roll: ' + str(camera.GetRoll()))
    print ('View angle: ' + str(camera.GetViewAngle()))
    # After finalizing the process we still need to delete the window
    del renderWin, renderInteractor

    return


def volume(img_volume=False, img_mesh=False, final_df=False, raw_df=False, cmap=False,
           time_frame=0, camera_x=138, camera_y=250, camera_z=1500, distance=False, dolly=False,
           azimuth=-25, elevation=70, roll=0, yaw=0, pitch=0, near_clip=100, far_clip=2000,
           alpha_order=2.5, thresh=0.0, shaders=True, detection_to_focus=0, show_detection_focus=False,
           focus_on_center=True, ambient=0.5, diffuse=1.5, specular=0.4, detections_size=4.0, detections_opacity=1.0,
           antialising=2, clip_amount=1.0, parallel = False, vol_color='G',
           show_final_detections=False, show_raw_detections=False, show_volume=False,
           show_mesh=False, wireframe_mesh=False, mesh_blur=2, render=False, bbox=False, save=False):
    """
    Volume visualization with VTK.
    TO DO: Option to interactive window instead of generating figure
    """
    import vtk
    from vtk import *
    import numpy as np
    from matplotlib import pyplot as plt
    from vtk.util.numpy_support import vtk_to_numpy
    from matplotlib import cm
    import scipy.ndimage
    import sys
    import time
    import math

    # Matplotlib parameters. This should get a 1024x768 image, by adjusting the DPI accordingly
    dpi = float(300)
    render_size_w = float(1024)
    render_size_h = float(768)
    plt.rcParams['figure.figsize'] = render_size_w / dpi, render_size_h / dpi
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi

    # Render size. Using a size multiplier to get a better final resolution,
    # while keeping the 4:3 ratio. Use 1 for a fast preview, 4 for a final render.
    size_multiplier = 1
    w = render_size_w * size_multiplier
    h = render_size_h * size_multiplier

    # Gaussian blur on mesh data
    if show_mesh:
        img_mesh = scipy.ndimage.filters.gaussian_filter(img_mesh, sigma=mesh_blur)

    # Check if we have labels on the final_df
    if show_final_detections:
        if 'label' in final_df:
            pass
        else:
            print ('For final detection visualization we need a dataframe with labels !')

        # Frame to work on
        frame = final_df[final_df['t'] == time_frame]

    # Check if we have 8bit data
    '''
    if img_volume.dtype != 'uint8':
        print ('Volume visualization needs 8-bit image !')

        return
    '''
    # Get info about the image. Expecting a ZYX image (memotrack standard)
    SizeZ_volume = np.shape(img_volume)[0]
    SizeY_volume = np.shape(img_volume)[1]
    SizeX_volume = np.shape(img_volume)[2]

    # Get info about the image. Expecting a ZYX image (memotrack standard)
    SizeZ_mesh = np.shape(img_mesh)[0]
    SizeY_mesh = np.shape(img_mesh)[1]
    SizeX_mesh = np.shape(img_mesh)[2]

    # Put the threshold in a 8bit range
    thresh *= 255

    # Stop the render to change the values faster on interactive mode
    if render is not True:
        return

    # --- Loading data for volume --- #

    # From numpy to vtk image
    dataImporter_volume = vtk.vtkImageImport()
    # The previously created array is converted to a string of chars and imported.
    data_string_volume = img_volume.tostring()
    dataImporter_volume.CopyImportVoidPointer(data_string_volume, len(data_string_volume))
    # Telling vtk we have uint8
    dataImporter_volume.SetDataScalarTypeToUnsignedChar()
    # Telling vtk we have only one channel
    dataImporter_volume.SetNumberOfScalarComponents(1)
    # Dimensions of data
    dataImporter_volume.SetDataExtent(0, SizeX_volume - 1, 0, SizeY_volume - 1, 0, SizeZ_volume - 1)
    dataImporter_volume.SetWholeExtent(0, SizeX_volume - 1, 0, SizeY_volume - 1, 0, SizeZ_volume - 1)

    # --- Loading data for mesh --- #
    # From numpy to vtk image
    dataImporter_mesh = vtk.vtkImageImport()
    # The previously created array is converted to a string of chars and imported.
    data_string_mesh = img_mesh.tostring()
    dataImporter_mesh.CopyImportVoidPointer(data_string_mesh, len(data_string_mesh))
    # Telling vtk we have uint8
    dataImporter_mesh.SetDataScalarTypeToUnsignedChar()
    # Telling vtk we have only one channel
    dataImporter_mesh.SetNumberOfScalarComponents(1)
    # Dimensions of data
    dataImporter_mesh.SetDataExtent(0, SizeX_mesh - 1, 0, SizeY_mesh - 1, 0, SizeZ_mesh - 1)
    dataImporter_mesh.SetWholeExtent(0, SizeX_mesh - 1, 0, SizeY_mesh - 1, 0, SizeZ_mesh - 1)

    # This class describes how the volume is rendered (through ray tracing).
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()

    # --- Create renderer --- #

    renderer = vtk.vtkRenderer()
    # Background color
    #renderer.SetBackground(0.15, 0.15, 0.15)  # Nice gray
    renderer.SetBackground(1, 1, 1)  # Pure White
    # renderer.SetBackground(0, 0, 0)  # Pure Black

    # --- Clipping plane --- #
    planeClip = vtk.vtkPlane()
    planeClip.SetOrigin(SizeX_volume * clip_amount, SizeY_volume * clip_amount, SizeZ_volume * clip_amount)
    planeClip.SetNormal(-1.0, 0.0, 0.0)


    # --- Bounding box --- #

    if bbox:
        outline = vtkOutlineFilter()
        outline.SetInputConnection(dataImporter_volume.GetOutputPort())

        outlineMapper = vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())

        outlineActor = vtkActor()
        outlineActor.SetMapper(outlineMapper)
        outlineActor.GetProperty().SetColor(0.5, 0.5, 0.5)
        renderer.AddActor(outlineActor)

    # --- Create volume --- #
    if show_volume:
        volumeMapper = vtk.vtkVolumeRayCastMapper()
        volumeMapper.SetVolumeRayCastFunction(compositeFunction)
        volumeMapper.SetInputConnection(dataImporter_volume.GetOutputPort())
        volumeMapper.AddClippingPlane(planeClip)

        # Getting list of alpha for all possible intensities
        alpha_list = vtk.vtkPiecewiseFunction()
        for i in range(0, 255):
            if i > thresh:
                alpha_list.AddPoint(i, (i *0.6 / 255.0) ** alpha_order)
            else:
                alpha_list.AddPoint(i, 0)

        # Getting color for all possible intensities.
        # using here the colors from the matplotlib colormap
        color_list = vtk.vtkColorTransferFunction()
        # CMRmap
        for i in range(0, 255):
            #red = cm.Set3(i)[0]
            #green = cm.Set3(i)[1]
            #blue = cm.Set3(i)[2]
            if vol_color in ['G', 'green', 'Green', 'signal']:
                level = 100
                if i > level:
                    red = (i-level)/255.0
                    green = i/255.0
                    blue = 0
                else:
                    red = 0
                    green = i/255.0
                    blue = 0.0
            if vol_color in ['R', 'red', 'Red', 'nuclei']:
                    level = 100
                    if i > level:
                        red = i/255.0
                        green = (i-level)/255.0
                        blue = 0
                    else:
                        red = i/255.0
                        green = 0
                        blue = 0.0

            color_list.AddRGBPoint(i, red, green, blue)

        # Set Volume Propriety
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(color_list)
        volumeProperty.SetScalarOpacity(alpha_list)
        volumeProperty.SetInterpolationTypeToLinear()
        # Turn on shaders or no
        if shaders:
            volumeProperty.ShadeOn()
            volumeProperty.SetAmbient(ambient)
            volumeProperty.SetDiffuse(diffuse)
            volumeProperty.SetSpecular(specular)

        # Declare volume and its proprieties
        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)

        # Add the volume
        renderer.AddVolume(volume)

    # --- Mesh from volume --- #
    # Here we extract a mesh from the volume
    low_threshold = 0
    if show_mesh:
        # Threshold for binary image
        threshold = vtk.vtkImageThreshold()
        threshold.SetInputConnection(dataImporter_mesh.GetOutputPort())
        threshold.ThresholdByLower(low_threshold)  # remove all soft tissue
        threshold.ReplaceInOn()
        threshold.SetInValue(0)  # set all values below threshold to 0
        threshold.ReplaceOutOn()
        threshold.SetOutValue(1)  # set all values above threshold to 1

        # Mesh using marching cubes
        mesh = vtk.vtkDiscreteMarchingCubes()
        mesh.SetInputConnection(threshold.GetOutputPort())
        mesh.GenerateValues(1, 1, 1)

        # Smooth mesh
        smooth_mesh = vtk.vtkSmoothPolyDataFilter()
        smooth_mesh.SetInputConnection(mesh.GetOutputPort())
        smooth_mesh.SetNumberOfIterations(50)  # Original is 50
        smooth_mesh.SetRelaxationFactor(0.5)
        smooth_mesh.FeatureEdgeSmoothingOn()

        # Reducing number of vertices
        simple_mesh = vtk.vtkDecimatePro()
        simple_mesh.SetInputConnection(smooth_mesh.GetOutputPort())
        simple_mesh.SetTargetReduction(0.95)

        # Setting mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(simple_mesh.GetOutputPort())

        # Actor and proprieties
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Setting color
        actor.GetProperty().SetColor(0.0, 0.6, 0.78)
        actor.GetProperty().SetOpacity(0.5)

        # Use wireframe
        if wireframe_mesh:
            actor.GetProperty().SetRepresentationToWireframe()

        # actor.GetProperty().SetOpacity(detections_opacity)
        actor.GetProperty().SetDiffuse(diffuse)
        actor.GetProperty().SetSpecular(specular)
        actor.GetProperty().SetSpecularPower(10)

        # Assign actor to the renderer
        renderer.AddActor(actor)

    # --- Poly objects --- #
    # Here we create the spheres for the final detections
    # each sphere will have the color given by the specific label
    # on the colormap generated by memotrack
    if show_final_detections:

        for label in frame['label']:
            x = frame[frame['label'] == label]['x'].values
            y = frame[frame['label'] == label]['y'].values
            z = frame[frame['label'] == label]['z'].values

            # Create source
            source = vtk.vtkSphereSource()
            source.SetCenter(x, y, z)
            source.SetRadius(detections_size)
            source.SetThetaResolution(32)
            source.SetPhiResolution(32)
            source.LatLongTessellationOn()

            # Set Mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())

            # Actor and proprieties
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Getting colors from the colormap
            color_red = cmap(int(label))[0]
            color_green = cmap(int(label))[1]
            color_blue = cmap(int(label))[2]

            # Setting color
            actor.GetProperty().SetColor(color_red, color_green, color_blue)
            # actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetBackfaceCulling(1)

            if label == detection_to_focus:
                if show_detection_focus:
                    actor.GetProperty().SetColor(1.0, 0.0, 0.0)
            actor.GetProperty().SetOpacity(detections_opacity)
            actor.GetProperty().SetDiffuse(diffuse)
            actor.GetProperty().SetSpecular(specular)
            actor.GetProperty().SetSpecularPower(10)

            # Assign actor to the renderer
            renderer.AddActor(actor)

    # Here we're creating small spheres to symbolize the detections through time.
    # VTK cannot handle all the 200.000 objects that would result from all the detections
    # from every time frame. So, here every detection slowly fades out, disappearing after 10 frames.
    # This way we have a maximum of 20.000 objects on the scene, which can be rendered without problem.
    if show_raw_detections:
        ntrace = 10  # Number of frames that a detection will stay
        frame_opacity = float(0.0)

        # Get only the 'ntrace' time frames before the current one
        for frame_thresh in range(time_frame - ntrace, time_frame + 1):
            current_frame = raw_df[raw_df['t'] == frame_thresh]
            x_list = current_frame['x'].values
            y_list = current_frame['y'].values
            z_list = current_frame['z'].values

            # Create all the spheres for the time frame
            for i in range(len(x_list)):
                x = x_list[i]
                y = y_list[i]
                z = z_list[i]

                # Sphere
                source = vtk.vtkSphereSource()
                source.SetCenter(x, y, z)
                source.SetRadius(1.5)
                source.SetThetaResolution(6)
                source.SetPhiResolution(6)

                # Set Mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(source.GetOutputPort())

                # Actor and proprieties
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                # Setting color
                actor.GetProperty().SetColor(0.9, 0.9, 0.9)
                actor.GetProperty().SetBackfaceCulling(1)
                actor.GetProperty().SetOpacity(frame_opacity)

                actor.GetProperty().SetDiffuse(diffuse)
                actor.GetProperty().SetSpecular(specular)
                actor.GetProperty().SetSpecularPower(10)

                # Assign actor to the renderer
                renderer.AddActor(actor)

            # Increase opacity for next batch
            frame_opacity += (1 / float(ntrace))

    # --- Camera --- #

    camera = vtk.vtkCamera()
    camera.SetViewUp(0, 0, -1)
    # Setting the clip range of the camera, to include the whole volume
    camera.SetClippingRange(near_clip, far_clip)
    # Getting focal point
    if focus_on_center:
        camera.SetFocalPoint(SizeX_volume / 2.0, SizeY_volume / 2.0, SizeZ_volume / 2.0)
    else:
        x_target = int(frame[frame['label'] == detection_to_focus]['x'].values)
        y_target = int(frame[frame['label'] == detection_to_focus]['y'].values)
        z_target = int(frame[frame['label'] == detection_to_focus]['z'].values)
        camera.SetFocalPoint(x_target, y_target, z_target)

    # Camera position and angle
    camera.SetPosition(camera_x, camera_y, camera_z)
    camera.Azimuth(azimuth)
    camera.Elevation(elevation)
    camera.Roll(roll)
    camera.Yaw(yaw)
    camera.Pitch(pitch)
    if distance:
        camera.SetDistance(distance)

    if dolly:
        camera.Dolly(dolly)

    if parallel:
        d = camera.GetDistance()
        a = camera.GetViewAngle()
        camera.SetParallelScale(d*math.tan(0.5*(a*math.pi/180)))
        camera.ParallelProjectionOn()

    # Use created camera
    renderer.SetActiveCamera(camera)

    # --- Rendering --- #


    # Create window to render
    renderWindow = vtk.vtkRenderWindow()

    #renderWindow.SetDesiredUpdateRate(0)
    #renderWindow.SetOffScreenRendering(True)
    #renderWindow.OffScreenRenderingOn()

    # Set antialising
    renderWindow.SetAAFrames(antialising)

    # Size to render
    renderWindow.SetSize(w, h)
    #renderWindow.SetSize(10, 10)
    # Finally, render the scene
    renderWindow.AddRenderer(renderer)
    renderWindow.Render()

    # Convert window to image
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()

    # --- Preparing image to export --- #
    # Getting image as numpy
    vtk_image = windowToImageFilter.GetOutput()
    height, width, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    final_image = vtk_to_numpy(vtk_array).reshape(width, height, components)
    final_image = np.asarray(final_image, dtype='uint8')

    # Display using Matplotlib

    fig = plt.figure(frameon=False)
    fig.set_size_inches(render_size_w / dpi, render_size_h / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(final_image)

    if save:
        plt.savefig(save)
        plt.close(fig)

    return


def volume_interact(img_8bit_volume, img_8bit_mesh, final_df, raw_df, cmap):
    import memotrack.display
    import numpy as np

    if len(final_df) > 0:
        frame = final_df[final_df['t'] == 0]
        nframes = final_df['t'].nunique()
        ndetections = len(frame)
    else:
        frame = 0
        nframes = 0
        ndetections = 0

    interact(memotrack.display.volume,
             img_volume=fixed(img_8bit_volume),
             img_mesh=fixed(img_8bit_mesh),
             final_df=fixed(final_df),
             raw_df=fixed(raw_df),
             cmap=fixed(cmap),
             near_clip=(1, 500),
             far_clip=(1000, 2000),
             alpha_order=(0, 10, 0.01),
             thresh=(0, 1, 0.01),
             clip_amount=(0, 1, 0.01),
             shaders=True,
             ambient=(0, 5, 0.1),
             diffuse=(0, 5, 0.1),
             specular=(0, 5, 0.1),
             camera_z=(0, (len(img_8bit_volume) * 5 - 1)),
             camera_y=(0, (len(img_8bit_volume[0]) - 1)),
             camera_x=(0, (len(img_8bit_volume[0][0]) - 1)),
             azimuth=(-180, 180),
             elevation=(-180, 180),
             roll=(-180, 180),
             yaw=(-180, 180),
             pitch=(-180, 180),
             detection_to_focus=(0, ndetections),
             show_detection_focus=False,
             detections_opacity=(0, 1, 0.01),
             antialising=(0, 12),
             render=True,
             time_frame=(0, nframes),
             mesh_blur=(0, 5, 0.1))

    return


def venn(ndetections, nground, hits, jaccard):
    """
    Creates a Venn diagram to compare detections an ground truth.
    the color pf the intersection ranges from red to green, according to the Jaccard index
    :param ndetections: Total number of detections
    :param nground: Total number of points in the ground truth
    :param hits: Number of detection hits
    :param jaccard: Jaccard index
    :return:
    """
    from matplotlib_venn import venn2
    from matplotlib import pyplot as plt
    from matplotlib import pylab

    plt.figure(figsize=(12, 12))
    plt.title('Jaccard index: ' + str(jaccard))
    v = venn2(subsets=(ndetections - hits, nground - hits, hits),
              set_labels=('Detections (' + str(ndetections) + ' points)', 'Ground truth (' + str(nground) + ' points)'))

    # Props for Detections
    v.get_patch_by_id('10').set_alpha(1)
    # v.get_patch_by_id('10').set_color('#03A9F4')
    v.get_patch_by_id('10').set_color('#ACDBF2')

    # Props for Ground truth
    v.get_patch_by_id('01').set_alpha(1)
    # v.get_patch_by_id('01').set_color('#8BC34A')
    v.get_patch_by_id('01').set_color('#AFF0C2')

    # Props for Hits
    jaccard_color = pylab.cm.RdYlGn(jaccard)
    v.get_patch_by_id('11').set_alpha(1)
    v.get_patch_by_id('11').set_color(jaccard_color)

    return


def histogram(data, nbins=15, color='#006f90', title='', stats_title=True, xlog=False, ylog=False, ymax=False,
              rotate_labels=False, accent_after=False, save=False, custom_bins=False, vline=False, smart_size=False,
              normed=False, debug=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import memotrack.display
    import math

    # Custom nice colors
    color = memotrack.display.nice_colors(color)

    if stats_title:
        avg = np.mean(data)
        std = np.std(data)
        title = (title + ' (avg:' + str(avg) + ' std:' + str(std) + ')')

    # Plot
    if smart_size:
        fig = plt.figure(figsize=(nbins * 0.5, 8))
    else:
        fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    # Custom bins use a step size with only two decimal, making visualization easier
    # You may also pass a list [min,max] for pre-defined limits
    if custom_bins:
        data_min = np.min(data)
        data_max = np.max(data)

        if isinstance(custom_bins, list):
            data_min = custom_bins[0]
            data_max = custom_bins[1]

        data_diff = data_max - data_min
        step = math.ceil((float(data_diff) / nbins) * 100)  # Calculating step
        step /= 100  # Doing this because I want only 2 decimals

        if debug:
            print ('\nmin, max, diff, step')
            print (data_min, data_max, data_diff, step)

        #nbins = np.arange(math.floor(data_min), math.ceil(data_max), step)
        nbins = np.arange(data_min,data_max, step)

    n, bins, patches = plt.hist(data, nbins, normed=normed, facecolor=color,
                                edgecolor='#FFFFFF', align='mid', linewidth=2, alpha=0.4)

    if accent_after:
        position = 0
        for patch in patches:
            if bins[position] > accent_after:
                plt.setp(patch, 'alpha', 1.0)
            position += 1
    # Fixes
    plt.title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks(bins[:-1])
    plot_height = ax.get_ylim()[1]

    if rotate_labels:
        ax.set_xticklabels(bins[:-1], rotation=90)

    if xlog:
        ax.set_xscale("log", nonposx='clip')
    if ylog:
        ax.set_yscale("log", nonposx='clip')
        plt.ylim([0.5, plot_height])  # This to show values on 1 on the log scale

    # Only now get the y min value, because log scale might had changed it
    ymin = ax.get_ylim()[0]
    if ymax:
        plt.ylim([ymin, ymax])  # Setting custom ymax value

    plt.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
    plt.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')

    if vline:
        plt.axvline(x=vline, linewidth=3, color='orange')

    if save:
        plt.savefig(save)
        plt.close()
    return n, bins


def boxplot(data, xticks=False, xtitle=False, ytitle=False, save=False, colors=False, zero_bottom=False,
            dots=True, dots_labels=False, annotate=False, log_transform=False, test_type='tt'):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    import memotrack.display
    import random

    if log_transform:
        for i in range(len(data)):
            data[i] = np.log(data[i])

    # Get number of boxes for the proper size of image
    nboxes = len(data)
    fig = plt.figure(1, figsize=(5 + (0.5 * nboxes), 9))
    ax = fig.add_subplot(111)

    # Axes fixes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='on')
    plt.tick_params(axis='y', which='both', left='on', right='off', labelleft='on')

    # Statistics
    if test_type == 'tt':
        print ('t-tests:')
        for a in range(nboxes):
            for b in range(nboxes):
                if a < b:
                    t, prob = stats.ttest_ind(data[a], data[b])
                    if prob < 0.01:
                        print ('\033[92m\033[1m' + str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob) + '\033[0m')
                    elif prob < 0.05:
                        print ('\033[93m\033[1m' + str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob) + '\033[0m')

                    else:
                        print (str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob))

    if test_type == 'mw':
        print ('Mann-Whitney two-sided:')
        for a in range(nboxes):
            for b in range(nboxes):
                if a < b:
                    t, prob = stats.mannwhitneyu(data[a], data[b], alternative='two-sided')
                    if prob < 0.01:
                        print ('\033[92m\033[1m' + str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob) + '\033[0m')
                    elif prob < 0.05:
                        print ('\033[93m\033[1m' + str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob) + '\033[0m')
                    else:
                        print (str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob))

    if test_type == 'mw_greater':
        print ('Mann-Whitney greater:')
        for a in range(nboxes):
            for b in range(nboxes):
                if a < b:
                    t, prob = stats.mannwhitneyu(data[a], data[b], alternative='greater')
                    if prob < 0.01:
                        print ('\033[92m\033[1m' + str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob) + '\033[0m')
                    elif prob < 0.05:
                        print ('\033[93m\033[1m' + str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob) + '\033[0m')
                    else:
                        print (str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob))

    if test_type == 'mw_less':
        print ('Mann-Whitney lesser:')
        for a in range(nboxes):
            for b in range(nboxes):
                if a < b:
                    t, prob = stats.mannwhitneyu(data[a], data[b], alternative='less')
                    if prob < 0.01:
                        print ('\033[92m\033[1m' + str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob) + '\033[0m')
                    elif prob < 0.05:
                        print ('\033[93m\033[1m' + str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob) + '\033[0m')
                    else:
                        print (str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob))

    if test_type == 'wilcoxon':
        print ('Wilcoxon signed-rank test:')
        for a in range(nboxes):
            for b in range(nboxes):
                if a < b:
                    t, prob = stats.wilcoxon(data[a], data[b])
                    if prob < 0.01:
                        print ('\033[92m\033[1m' + str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob) + '\033[0m')
                    elif prob < 0.05:
                        print ('\033[93m\033[1m' + str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob) + '\033[0m')
                    else:
                        print (str(xticks[a]) + ' and ' + str(xticks[b])),
                        print ('\tp-value: ' + str(prob))


    # Generate plot
    if dots:
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    else:
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)

    # Insert ticks and titles
    if xtitle:
        plt.xlabel(xtitle)
    if ytitle:
        if log_transform:
            ytitle += ' (log)'
        plt.ylabel(ytitle)
    if xticks:
        ax.set_xticklabels(xticks)

    # Box fixes
    for median in bp['medians']:
        median.set(color='#C44D58', linewidth=2, alpha=0.9)
    for box in bp['boxes']:
        box.set(color='#666666', linewidth=2)
        box.set(facecolor='#EEEEEE', alpha=0.5)
    for whisker in bp['whiskers']:
        whisker.set(color='#666666', linewidth=2)
    for cap in bp['caps']:
        cap.set(color='#666666', linewidth=2)

    if dots:
        if dots_labels:
            '''
            labels_ravel = np.ravel(dots_labels)
            cmap = memotrack.display.rand_cmap(len(labels_ravel)+1, first_color_black=False,
                                               verbose=False, type='bright')
            '''
            cmap = plt.cm.get_cmap('rainbow')

        names_pos = 0
        for i in range(len(bp['boxes'])):
            y = data[i]
            # Randomly subsample the data in case of too many points
            too_many_points = 10000
            if len(y) > too_many_points:
                y = np.random.choice(y, size=too_many_points)

            x = np.random.normal(1 + i, 0.07, size=len(y))

            if annotate:
                for point_pos in range(len(x)):
                    name = annotate[names_pos][point_pos]
                    plt.annotate(str(name), xy=(x[point_pos], y[point_pos]),
                                 xytext=(x[point_pos] + 0.3, y[point_pos] + 0.005),
                                 size=9,
                                 bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec='#444444', alpha=1),
                                 arrowprops=dict(arrowstyle='->', alpha=0.5))

            names_pos += 1

            if dots_labels:
                color_list = []
                for dot in range(len(dots_labels[i])):
                    string = dots_labels[i][dot]
                    random.seed(hash(string))
                    color = cmap(random.random())
                    color_list.append(color)
                    # print (string, color)

                plt.scatter(x, y, lw=0, c=color_list, s=15, alpha=0.8, zorder=0)
            else:
                plt.scatter(x, y, lw=0, c='k', s=15, alpha=0.5, zorder=0)

    if colors == 'stim':
        blue = memotrack.display.nice_colors('blue')
        red = memotrack.display.nice_colors('red')
        bp['boxes'][0].set(facecolor='#333333', linewidth=2, alpha=0.4)
        bp['boxes'][1].set(facecolor=blue, linewidth=2, alpha=0.4)
        bp['boxes'][2].set(facecolor=red, linewidth=2, alpha=0.4)

    if colors == 'conditions':
        bp['boxes'][0].set(facecolor='#ABABAB', linewidth=2, alpha=0.4)
        bp['boxes'][1].set(facecolor='#33AA60', linewidth=2, alpha=0.4)
        bp['boxes'][2].set(facecolor='#DDAA60', linewidth=2, alpha=0.4)

    if colors == 'conditions2':
        bp['boxes'][0].set(facecolor='#ABABAB', linewidth=2, alpha=0.4)
        bp['boxes'][1].set(facecolor='#33AA60', linewidth=2, alpha=0.4)
        bp['boxes'][2].set(facecolor='#DDAA60', linewidth=2, alpha=0.4)
        bp['boxes'][3].set(facecolor='#33AA60', linewidth=2, alpha=0.4)
        bp['boxes'][4].set(facecolor='#DDAA60', linewidth=2, alpha=0.4)

    if colors == 'conditions3':
        bp['boxes'][0].set(facecolor='#00C1A6', linewidth=2, alpha=0.4)
        bp['boxes'][1].set(facecolor='#B36EC4', linewidth=2, alpha=0.4)
        bp['boxes'][2].set(facecolor='#00C1A6', linewidth=2, alpha=0.4)
        bp['boxes'][3].set(facecolor='#B36EC4', linewidth=2, alpha=0.4)

    if colors == 'conditions+naive':
        bp['boxes'][0].set(facecolor='#DADADA', linewidth=2, alpha=0.4)
        bp['boxes'][1].set(facecolor='#00C1A6', linewidth=2, alpha=0.4)
        bp['boxes'][2].set(facecolor='#B36EC4', linewidth=2, alpha=0.4)
        bp['boxes'][3].set(facecolor='#00C1A6', linewidth=2, alpha=0.4)
        bp['boxes'][4].set(facecolor='#B36EC4', linewidth=2, alpha=0.4)

    if save:
        plt.savefig(save)

    # To fix bottom to zero
    if zero_bottom:
        plot_height = ax.get_ylim()[1]
        plt.ylim([0.0, plot_height])
    plt.show()


def nice_colors(color):
    """
    Get string, gives back nice HEX color. 'rand' for random pastel color
    :param color: 'rand', 'blue', 'orange', 'lime' etc
    :return:
    """
    import numpy as np
    import matplotlib.colors

    # Nice colors
    if color == 'light_blue':
        color = '#69D2E7'
    if color == 'blue':
        color = '#00AAFF'
    if color == 'orange':
        color = '#F38630'
    if color == 'lime':
        color = '#C7F464'
    if color == 'red':
        color = '#C44D58'
    if color == 'green':
        color = '#6ACD86'
    if color == 'grey':
        color = '#333333'

    # Random pastel
    if color == 'rand' or color == 'random':
        rgb = [1, 1, 1]
        # Loop while the color is too bright
        while sum(rgb) > 2.6:
            # Get RGB values
            rgb = np.random.rand(3)
            # Make pastel
            rgb = [(c + 1) / 2 for c in rgb]

        # Convert to HEX
        color = matplotlib.colors.rgb2hex(rgb)

    return color


def delaunay(tri, verbose=False, debug=False, elev=15, azim=45,
             xstart=0, ystart=0, zstart=0, lim=500):
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt

    nsimplex = tri.nsimplex
    if debug:
        nsimplex = debug

    ndim = tri.ndim
    npoints = tri.npoints

    if verbose:
        print ('Data on ' + str(ndim) + ' dimensions.'),
        print ('Total of ' + str(npoints) + ' points, forming ' + str(nsimplex) + ' simplex.')

    # Start axes
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev, azim)

    # These are the sides of the simplex that we need to plot
    vertices_list = [[0, 1], [0, 2], [0, 3], [1, 3], [1, 2], [2, 3]]
    for simplex in range(nsimplex):
        point_list = []
        if debug:
            print ('\n Simplex #' + str(simplex))
        simplex_points = tri.simplices[simplex]

        for point in simplex_points:
            coords = tri.points[point]
            if debug:
                print (coords)
            point_list.append(coords)

        for vertice in vertices_list:
            x1 = point_list[vertice[0]][0]
            x2 = point_list[vertice[1]][0]
            y1 = point_list[vertice[0]][1]
            y2 = point_list[vertice[1]][1]
            z1 = point_list[vertice[0]][2]
            z2 = point_list[vertice[1]][2]
            ax.plot([y1, y2], [x1, x2], [z1, z2], lw=0.5, color='#666666', alpha=0.3)

        plot_coords = zip(*point_list)
        ax.scatter(plot_coords[1], plot_coords[0], plot_coords[2], color='#333333')

    # Adjusts
    ax.set_xlim3d(xstart, lim + xstart)
    ax.set_ylim3d(ystart, lim + ystart)
    ax.set_zlim3d(zstart, lim + zstart)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return point_list


def signals_interact_pre_process(df, debug=False):
    import matplotlib
    from bqplot import pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import memotrack.display

    nlabels = int(df['label'].nunique())

    if debug:
        nlabels = debug
    nframes = int(df['t'].nunique())

    intensities_list = []
    label_list = []
    # Generate plotting lists
    for label in range(nlabels):
        if df[df['label'] == label]['responsive'].any():
            temp_int_list = df[df['label'] == label].filtered.get_values()
            temp_int_list[temp_int_list < 0] = 0
            intensities_list.append(temp_int_list)
            label_list.append(label)
            print ('[' + str(label) + ']'),

    x_values = range(nframes)
    return x_values, intensities_list, label_list


def max_and_signal(path, original_df, cmap, reprocess_max=False, frame=0, verbose=True,
                   peak_threshold=0.3, plot_range=20, export_mode=False, marker='o', signal='filtered',
                   focus=False, zoom=25):
    global amount
    import memotrack
    import numpy as np
    import os
    import tifffile
    import matplotlib.pyplot as plt
    import math
    import sys
    import time

    t = time.time()

    plt.close('all')

    detections_df = original_df.copy(deep=True)

    # default path for the max projection
    max_path0 = path[:-4] + '_C1MAX.tif'
    img_cmap0 = 'magma'

    max_path1 = path[:-4] + '_C2MAX.tif'
    img_cmap1 = 'viridis'

    max_path = [max_path0, max_path1]
    img_cmap = [img_cmap0, img_cmap1]

    # Do the max projection only if needed
    for c in range(2):

        if (os.path.isfile(max_path[c]) is False) or (reprocess_max is True):
            if verbose:
                print ('Max projection not found on {}, making one now...'.format(max_path[c]))
            # Load the full image
            full_img = memotrack.io.load_full_img(path, verbose=True)
            channel_img = full_img[:, :, c, :, :]  # Assuming we have TZCYX
            nframes = np.shape(full_img)[0]
            nslices = np.shape(full_img)[1]
            max_img = np.zeros((np.shape(full_img)[0], np.shape(full_img)[3], np.shape(full_img)[4]))

            print (np.shape(max_img))
            if verbose:
                print ('\nStarting projection on {} frames'.format(len(max_img)))

            print ('Shape max_img: {}'.format(np.shape(max_img)))
            print ('Shape channel_img: {}'.format(np.shape(channel_img)))

            for mframe in range(nframes):
                if verbose:
                    print ('.'),
                for mslice in range(nslices):
                    inds = channel_img[mframe][mslice] > max_img[mframe]
                    max_img[mframe][inds] = channel_img[mframe][mslice][inds]

            if verbose:
                print ('[Done]')

            # Saving projection to disk
            print (np.shape(max_img))
            tifffile.imsave(max_path[c], max_img)

    # Load projection from disk if its already there
    else:
        max_img = [tifffile.imread(max_path[0]), tifffile.imread(max_path[1])]
        if verbose:
            print ('Image loaded from \n{}\n{}'.format(max_path[0], max_path[1]))
            print (np.shape(max_img))

    # Add padding to solve zoom problems
    pad = 20
    max_img = np.pad(max_img, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))

    # Start figure
    fig = plt.figure(figsize=(21, 9))
    ax0 = plt.subplot2grid((1, 4), (0, 0), colspan=1)
    ax1 = plt.subplot2grid((1, 4), (0, 1), colspan=1)
    ax2 = plt.subplot2grid((1, 4), (0, 2), colspan=2)
    ax = [ax0, ax1, ax2]
    pos1 = ax1.get_position()
    pos1.x0 -= 0.05
    ax1.set_position(pos1)  # pos = [left, bottom, width, height]

    # Show max projection
    for c in range(2):
        max_percentile = np.percentile(max_img[c], 99.99)
        min_percentile = np.percentile(max_img[c], 0.01)
        ax[c].imshow(max_img[c][frame], vmax=max_percentile, vmin=min_percentile, cmap=img_cmap[c])
        ax[c].axis('off')

    # Overlay detections to max projection
    nlabels = detections_df['label'].nunique()

    label_list = []
    highest_intensity = 0
    for label in range(nlabels):
        value = detections_df[detections_df['label'] == label][signal].max()
        if value > peak_threshold:
            label_list.append(label)
        if value > highest_intensity:
            highest_intensity = value
            best_label = int(label)

    if focus == 'best':
        focus = best_label
        if verbose:
            print ('Focusing on label {}'.format(best_label))

    if verbose:
        if len(label_list) <= 25:
            print ('Showing labels:\n{}'.format(label_list))
        else:
            print ('Showing {} labels'.format(len(label_list)))

    # we need to correct the coordinates for the padding

    detections_df['x'] = detections_df['xsmooth'] + pad
    detections_df['y'] = detections_df['ysmooth'] + pad

    for c in range(2):

        # If zoom is false, show whole image
        if zoom is False:
            zoom = max(np.shape(max_img))

        if zoom > min(np.shape(max_img)[2] - 2 * pad, np.shape(max_img)[3] - 2 * pad) / 2:
            zoom = min(np.shape(max_img)[2] - 2 * pad, np.shape(max_img)[3] - 2 * pad) / 2

        if focus:
            xmean = int(detections_df[detections_df['label'] == focus].x.mean())
            ymean = int(detections_df[detections_df['label'] == focus].y.mean())

        # if focus is False, zoom on center
        else:
            xmean = np.shape(max_img)[3] / 2.0
            ymean = np.shape(max_img)[2] / 2.0

        xzoom_low = xmean - zoom
        xzoom_high = xmean + zoom

        yzoom_low = (ymean - 2 * zoom)
        yzoom_high = (ymean + 2 * zoom)

        ax[c].set_xlim(int(xzoom_low), int(xzoom_high))
        ax[c].set_ylim(int(yzoom_low), int(yzoom_high))

    # Get the df only for desired frame
    time_df = detections_df[detections_df['t'] == frame]
    frame_df = time_df[time_df['label'].isin(label_list)].copy(deep=True)

    # Generate color list
    color_list = []
    for label in range(nlabels):
        color_list.append(cmap(label / float(nlabels)))

    scatter_color_list = []
    for label in label_list:
        scatter_color_list.append(color_list[label])

    # Plot detections on top of image
    for c in range(2):
        if marker in ['o', 'circle']:
            # Plot using Circles
            size = 5000 * (1.0 / (math.sqrt(zoom)))
            ax[c].scatter(frame_df['x'], frame_df['y'], edgecolor=scatter_color_list, facecolor='#00000000', lw=2.0,
                          cmap=cmap, marker='o', s=size, )
        elif marker in ['x', 'cross']:
            # Plot using X
            size = 5000 * (1.0 / zoom)
            ax[c].scatter(frame_df['x'], frame_df['y'], c=scatter_color_list, lw=2.0,
                          cmap=cmap, marker='x', s=size)

    # Plot lines
    for label, df in detections_df.groupby(by='label'):
        line = df[signal].as_matrix()
        line[line < 0] = 0
        if np.max(line) >= peak_threshold:
            ax[2].plot(line, c=color_list[int(label)], alpha=1.0, lw=2)

    if 'Q' in detections_df:
        ax2 = ax[2].twinx()
        temp_df = detections_df.groupby(by='t').mean()
        ax2.plot(temp_df['Q'], alpha=0.5, lw=2, linestyle=':', color='#AA1030')
        ax2.set_ylim(0, 1.005)

        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.set_ylabel('Detection quality')

    # Plot fixes
    ax[2].set_xlim(left=frame - plot_range, right=frame + plot_range)  # set view window
    ax[2].set_ylim(bottom=0)  # set view window
    ax[2].axvline(frame, c='#000000', lw=5, ls='-', alpha=0.25, zorder=-1)

    if signal == 'norm_intensity':
        ax[2].set_ylabel('Normalized intensity')
    elif signal == 'intensity':
        ax[2].set_ylabel('Raw intensity')
    elif signal == 'filtered':
        ax[2].set_ylabel('Filtered intensity')

    ax[2].set_xlabel('Time frame')

    plt.suptitle(os.path.basename(path)[:-4], fontsize=16)

    if export_mode:
        plt.close(fig)

    # Gather stuff for the quick version of the visualization
    preprocess = {'max_img': max_img,
                  'color_list': color_list,
                  'scatter_color_list': scatter_color_list,
                  'label_list': label_list,
                  'detections_df': detections_df,
                  'path': path,
                  'pad': pad,
                  'best_label': best_label,
                  'fig': fig,
                  'ax': ax}

    print ('Total time: {}'.format(time.time() - t))

    return fig, preprocess


def max_and_signal_quick(p, cmap, frame=92, zoom=25, signal='norm_intensity', plot_range=20, focus='best',
                         peak_threshold=0.4, export_mode=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import time

    # initial_time = time.time()
    plt.close('all')
    # Get vars from dict
    detections_df = p['detections_df']
    max_img = p['max_img']
    color_list = p['color_list']
    label_list = p['label_list']
    scatter_color_list = p['scatter_color_list']
    path = p['path']
    pad = p['pad']
    best_label = p['best_label']
    # fig = p['fig']
    # ax = p['ax']

    # t = time.time()
    # Start figure
    if export_mode:
        fig = plt.figure(figsize=(21, 9))
    else:
        fig = plt.figure(figsize=(15, 6))
    ax0 = plt.subplot2grid((1, 4), (0, 0), colspan=1)
    ax1 = plt.subplot2grid((1, 4), (0, 1), colspan=1)
    ax2 = plt.subplot2grid((1, 4), (0, 2), colspan=2)
    ax = [ax0, ax1, ax2]
    pos1 = ax1.get_position()
    pos1.x0 -= 0.05
    ax1.set_position(pos1)  # pos = [left, bottom, width, height]
    # print ('Start figure: {}'.format(time.time()-t))

    img_cmap = ['magma', 'viridis']

    # If zoom is false, show whole image
    if zoom is False:
        zoom = max(np.shape(max_img))

    min_size = min(np.shape(max_img)[2] - 2 * pad, np.shape(max_img)[3] - 2 * pad) / 2
    if zoom > min_size:
        zoom = min_size

    if focus == 'best':
        focus = best_label

    # Get center of the image
    xmean = np.shape(max_img)[3] / 2.0
    ymean = np.shape(max_img)[2] / 2.0

    if focus:
        xmean_focus = int(detections_df[detections_df['label'] == focus].x.mean())
        ymean_focus = int(detections_df[detections_df['label'] == focus].y.mean())
    else:
        xmean_focus = xmean
        ymean_focus = ymean

    # print ('Zoom: {}'.format(zoom))
    # print ('xmean: {}\tymean:{}'.format(xmean, ymean))
    # t = time.time()

    label_list = []
    highest_intensity = 0
    nlabels = detections_df['label'].nunique()

    for label in range(nlabels):
        value = detections_df[detections_df['label'] == label][signal].max()
        if value > peak_threshold:
            label_list.append(label)
        if value > highest_intensity:
            highest_intensity = value
            best_label = int(label)
    scatter_color_list = []

    for label in label_list:
        scatter_color_list.append(color_list[label])

    # Get the df only for desired frame
    time_df = detections_df[detections_df['t'] == frame]
    frame_df = time_df[time_df['label'].isin(label_list)].copy(deep=True)

    for c in range(2):
        max_percentile = np.percentile(max_img[c], 99.99)
        min_percentile = np.percentile(max_img[c], 0.01)
        ax[c].imshow(max_img[c][frame], vmax=max_percentile, vmin=min_percentile, cmap=img_cmap[c])
        ax[c].axis('off')

    # print ('Show projections: {}'.format(time.time()-t))

    # Plot detections on top of image
    for c in range(2):
        # Plot using X
        size = 5000 * (1.0 / zoom)
        ax[c].scatter(frame_df['xsmooth'] + pad, frame_df['ysmooth'] + pad, c=scatter_color_list, lw=2.0,
                      cmap=cmap, marker='x', s=size)
    # t = time.time()
    for label in label_list:
        line = detections_df[detections_df['label'] == label][signal].as_matrix()
        ax[2].plot(line, c=color_list[int(label)], alpha=1.0, lw=2)
    # print ('Plot lines: {}'.format(time.time()-t))

    # t = time.time()
    for c in range(2):
        zratio = zoom / float(min_size)

        # First, get the central position weighted by the zoom ratio
        xpos = ((xmean * zratio) + (xmean_focus * (1 - zratio)))
        ypos = ((ymean * zratio) + (ymean_focus * (1 - zratio)))

        '''
        print ('xmean: {}'.format(xmean))
        print ('ymean: {}'.format(ymean))
        print ('xmean_focus: {}'.format(xmean_focus))
        print ('ymean_focus: {}'.format(ymean_focus))
        print ('zoom: {}'.format(zoom))
        print ('min_size: {}'.format(min_size))
        print ('zratio: {}'.format(zratio))
        print ('xpos: {}'.format(xpos))
        print ('ypos: {}'.format(ypos))
        '''

        # Now we set the axis start and end based on the zoom level
        xzoom_low = xpos - zoom
        xzoom_high = xpos + zoom

        yzoom_low = (ypos - 2 * zoom)
        yzoom_high = (ypos + 2 * zoom)

        ax[c].set_xlim(int(xzoom_low), int(xzoom_high))
        ax[c].set_ylim(int(yzoom_low), int(yzoom_high))

    # print ('Make zoom: {}'.format(time.time()-t))

    # t = time.time()
    if 'Q' in detections_df:
        ax2 = ax[2].twinx()
        temp_df = detections_df.groupby(by='t').mean()
        ax2.plot(temp_df['Q'], alpha=0.5, lw=2, linestyle=':', color='#AA1030')
        ax2.set_ylim(0, 1.005)

        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.set_ylabel('Detection quality')
    # print ('Plot Quality: {}'.format(time.time()-t))
    # t = time.time()
    # Plot fixes
    ax[2].set_xlim(left=frame - plot_range, right=frame + plot_range)  # set view window
    ax[2].set_ylim(bottom=0)  # set view window
    ax[2].axvline(frame, c='#000000', lw=5, ls='-', alpha=0.25, zorder=-1)

    if signal == 'norm_intensity':
        ax[2].set_ylabel('Normalized intensity')
    elif signal == 'intensity':
        ax[2].set_ylabel('Raw intensity')
    elif signal == 'filtered':
        ax[2].set_ylabel('Filtered intensity')

    ax[2].set_xlabel('Time frame')

    if signal == 'norm_intensity':
        ax[2].set_ylabel('Normalized intensity')
    elif signal == 'intensity':
        ax[2].set_ylabel('Raw intensity')
    elif signal == 'filtered':
        ax[2].set_ylabel('Filtered intensity')

    ax[2].set_xlabel('Time frame')

    plt.suptitle(os.path.basename(path)[:-4], fontsize=16)
    # print ('Plot fixes: {}'.format(time.time() - t))
    # print ('Total time: {}'.format(time.time()-initial_time))

    if export_mode:
        plt.close(fig)

    return fig


def projections_and_signal(path, original_df, cmap, reprocess_max=False, frame=0, verbose=True,
                           peak_threshold=0.3, plot_range=20, export_mode=False, marker='o', signal='filtered',
                           focus=False, zoom=25):
    global amount
    import memotrack
    import numpy as np
    import os
    import tifffile
    import matplotlib.pyplot as plt
    import math
    import sys
    import time

    t = time.time()

    plt.close('all')

    detections_df = original_df.copy(deep=True)

    # default path for the max projection
    max_path0 = path[:-4] + '_C1MAX.tif'
    img_cmap0 = 'magma'

    max_path1 = path[:-4] + '_C2MAX.tif'
    img_cmap1 = 'viridis'

    max_path = [max_path0, max_path1]
    img_cmap = [img_cmap0, img_cmap1]

    # Do the max projection only if needed
    for c in range(2):

        if (os.path.isfile(max_path[c]) is False) or (reprocess_max is True):
            if verbose:
                print ('Max projection not found on {}, making one now...'.format(max_path[c]))
            # Load the full image
            full_img = memotrack.io.load_full_img(path, verbose=True)
            channel_img = full_img[:, :, c, :, :]  # Assuming we have TZCYX
            nframes = np.shape(full_img)[0]
            nslices = np.shape(full_img)[1]
            max_img = np.zeros((np.shape(full_img)[0], np.shape(full_img)[3], np.shape(full_img)[4]))

            print (np.shape(max_img))
            if verbose:
                print ('\nStarting projection on {} frames'.format(len(max_img)))

            print ('Shape max_img: {}'.format(np.shape(max_img)))
            print ('Shape channel_img: {}'.format(np.shape(channel_img)))

            for mframe in range(nframes):
                if verbose:
                    print ('.'),
                for mslice in range(nslices):
                    inds = channel_img[mframe][mslice] > max_img[mframe]
                    max_img[mframe][inds] = channel_img[mframe][mslice][inds]

            if verbose:
                print ('[Done]')

            # Saving projection to disk
            print (np.shape(max_img))
            tifffile.imsave(max_path[c], max_img)

    # Load projection from disk if its already there
    else:
        max_img = [tifffile.imread(max_path[0]), tifffile.imread(max_path[1])]
        if verbose:
            print ('Image loaded from \n{}\n{}'.format(max_path[0], max_path[1]))
            print (np.shape(max_img))

    # Add padding to solve zoom problems
    pad = 20
    max_img = np.pad(max_img, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))

    # Start figure
    fig = plt.figure(figsize=(21, 9))
    ax0 = plt.subplot2grid((1, 4), (0, 0), colspan=1)
    ax1 = plt.subplot2grid((1, 4), (0, 1), colspan=1)
    ax2 = plt.subplot2grid((1, 4), (0, 2), colspan=2)
    ax = [ax0, ax1, ax2]
    pos1 = ax1.get_position()
    pos1.x0 -= 0.05
    ax1.set_position(pos1)  # pos = [left, bottom, width, height]

    # Show max projection
    for c in range(2):
        max_percentile = np.percentile(max_img[c], 99.99)
        min_percentile = np.percentile(max_img[c], 0.01)
        ax[c].imshow(max_img[c][frame], vmax=max_percentile, vmin=min_percentile, cmap=img_cmap[c])
        ax[c].axis('off')

    # Overlay detections to max projection
    nlabels = detections_df['label'].nunique()

    label_list = []
    highest_intensity = 0
    for label in range(nlabels):
        value = detections_df[detections_df['label'] == label][signal].max()
        if value > peak_threshold:
            label_list.append(label)
        if value > highest_intensity:
            highest_intensity = value
            best_label = int(label)

    if focus == 'best':
        focus = best_label
        if verbose:
            print ('Focusing on label {}'.format(best_label))

    if verbose:
        if len(label_list) <= 25:
            print ('Showing labels:\n{}'.format(label_list))
        else:
            print ('Showing {} labels'.format(len(label_list)))

    # we need to correct the coordinates for the padding

    detections_df['x'] = detections_df['x'] + pad
    detections_df['y'] = detections_df['y'] + pad

    for c in range(2):

        # If zoom is false, show whole image
        if zoom is False:
            zoom = max(np.shape(max_img))

        if zoom > min(np.shape(max_img)[2] - 2 * pad, np.shape(max_img)[3] - 2 * pad) / 2:
            zoom = min(np.shape(max_img)[2] - 2 * pad, np.shape(max_img)[3] - 2 * pad) / 2

        if focus:
            xmean = int(detections_df[detections_df['label'] == focus].x.mean())
            ymean = int(detections_df[detections_df['label'] == focus].y.mean())

        # if focus is False, zoom on center
        else:
            xmean = np.shape(max_img)[3] / 2.0
            ymean = np.shape(max_img)[2] / 2.0

        xzoom_low = xmean - zoom
        xzoom_high = xmean + zoom

        yzoom_low = (ymean - 2 * zoom)
        yzoom_high = (ymean + 2 * zoom)

        ax[c].set_xlim(int(xzoom_low), int(xzoom_high))
        ax[c].set_ylim(int(yzoom_low), int(yzoom_high))

    # Get the df only for desired frame
    time_df = detections_df[detections_df['t'] == frame]
    frame_df = time_df[time_df['label'].isin(label_list)].copy(deep=True)

    # Generate color list
    color_list = []
    for label in range(nlabels):
        color_list.append(cmap(label / float(nlabels)))

    scatter_color_list = []
    for label in label_list:
        scatter_color_list.append(color_list[label])

    # Plot detections on top of image
    for c in range(2):
        if marker in ['o', 'circle']:
            # Plot using Circles
            size = 5000 * (1.0 / (math.sqrt(zoom)))
            ax[c].scatter(frame_df['x'], frame_df['y'], edgecolor=scatter_color_list, facecolor='#00000000', lw=2.0,
                          cmap=cmap, marker='o', s=size, )
        elif marker in ['x', 'cross']:
            # Plot using X
            size = 5000 * (1.0 / zoom)
            ax[c].scatter(frame_df['x'], frame_df['y'], c=scatter_color_list, lw=2.0,
                          cmap=cmap, marker='x', s=size)

    # Plot lines
    for label, df in detections_df.groupby(by='label'):
        line = df[signal].as_matrix()
        line[line < 0] = 0
        if np.max(line) >= peak_threshold:
            ax[2].plot(line, c=color_list[int(label)], alpha=1.0, lw=2)

    if 'Q' in detections_df:
        ax2 = ax[2].twinx()
        temp_df = detections_df.groupby(by='t').mean()
        ax2.plot(temp_df['Q'], alpha=0.5, lw=2, linestyle=':', color='#AA1030')
        ax2.set_ylim(0, 1.005)

        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.set_ylabel('Detection quality')

    # Plot fixes
    ax[2].set_xlim(left=frame - plot_range, right=frame + plot_range)  # set view window
    ax[2].set_ylim(bottom=0)  # set view window
    ax[2].axvline(frame, c='#000000', lw=5, ls='-', alpha=0.25, zorder=-1)

    if signal == 'norm_intensity':
        ax[2].set_ylabel('Normalized intensity')
    elif signal == 'intensity':
        ax[2].set_ylabel('Raw intensity')
    elif signal == 'filtered':
        ax[2].set_ylabel('Filtered intensity')

    ax[2].set_xlabel('Time frame')

    plt.suptitle(os.path.basename(path)[:-4], fontsize=16)

    if export_mode:
        plt.close(fig)

    # Gather stuff for the quick version of the visualization
    preprocess = {'max_img': max_img,
                  'color_list': color_list,
                  'scatter_color_list': scatter_color_list,
                  'label_list': label_list,
                  'detections_df': detections_df,
                  'path': path,
                  'pad': pad,
                  'best_label': best_label,
                  'fig': fig,
                  'ax': ax}

    print ('Total time: {}'.format(time.time() - t))

    return fig, preprocess


def XYZprojections_and_signal(xy, zy, zx, df, cmap, frame=92, path=False, peak_threshold=0.1,
                              plot_range=10, verbose=True):
    # Start figure
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import matplotlib.gridspec as gridspec

    nH = float(np.shape(xy)[2])
    nW = float(np.shape(xy)[3])
    iD = float(np.shape(zx)[3])
    size = 0.01
    fig = plt.figure(figsize=((iD + nW + nW + iD) * size, (iD + nH) * size), dpi=100)

    gs = gridspec.GridSpec(2, 4, width_ratios=[iD / nW, 1, 1, iD / nW], height_ratios=[iD / nH, 1])
    gs.update(wspace=0.03, hspace=0.03)

    ax2 = plt.subplot(gs[1, 1])
    ax0 = plt.subplot(gs[1, 0])
    ax1 = plt.subplot(gs[0, 1])
    axSignal1 = plt.subplot(gs[0, 3])

    ax5 = plt.subplot(gs[1, 2])
    ax3 = plt.subplot(gs[1, 3])
    ax4 = plt.subplot(gs[0, 2])
    axSignal2 = plt.subplot(gs[0, 0])

    # Show images for C1
    max_percentile = np.percentile(xy[:][0], 99.99)
    min_percentile = np.percentile(xy[:][0], 0.01)
    # print ('C1 percentiles: {0:.2f}, {1:.2f}'.format(min_percentile, max_percentile))
    ax0.imshow(zx[frame][0], cmap='magma', vmax=max_percentile, vmin=min_percentile)
    ax0.axis('off')
    ax1.imshow(zy[frame][0], cmap='magma', vmax=max_percentile, vmin=min_percentile)
    ax1.axis('off')
    ax2.imshow(xy[frame][0], cmap='magma', vmax=max_percentile, vmin=min_percentile)
    ax2.axis('off')

    # Show images for C2
    max_percentile = np.percentile(xy[:][1], 90)
    min_percentile = np.percentile(xy[:][1], 0)
    # print ('C2 percentiles: {0:.2f}, {1:.2f}'.format(min_percentile, max_percentile))
    ax3.imshow(zx[frame][1], cmap='viridis', vmax=max_percentile, vmin=min_percentile)
    ax3.axis('off')
    ax4.imshow(zy[frame][1], cmap='viridis', vmax=max_percentile, vmin=min_percentile)
    ax4.axis('off')
    ax5.imshow(xy[frame][1], cmap='viridis', vmax=max_percentile, vmin=min_percentile)
    ax5.axis('off')

    # Show detections on top of images

    # Overlay detections to max projection
    nlabels = df['label'].nunique()
    nframes = df['t'].nunique()

    label_list = []
    for label in range(nlabels):
        value = df[df['label'] == label]['norm_intensity'].max()
        if value > peak_threshold:
            label_list.append(label)

    if verbose:
        if len(label_list) <= 25:
            print ('Showing labels:\n{}'.format(label_list))
        else:
            print ('Showing {} labels'.format(len(label_list)))

    # Generate color list
    color_list = []
    for label in range(nlabels):
        color_list.append(cmap(label / float(nlabels)))

    scatter_color_list = []
    for label in label_list:
        scatter_color_list.append(color_list[label])

    # Get the df only for desired frame
    time_df = df[df['t'] == frame]
    frame_df = time_df[time_df['label'].isin(label_list)].copy(deep=True)

    # Star plotting
    msize = 25
    mkind = 'o'
    ax2.scatter(frame_df['xsmooth'], frame_df['ysmooth'], edgecolor=scatter_color_list,
                facecolor='#00000000', lw=1.0, cmap=cmap, marker=mkind, s=msize)
    ax5.scatter(frame_df['xsmooth'], frame_df['ysmooth'], edgecolor=scatter_color_list,
                facecolor='#00000000', lw=1.0, cmap=cmap, marker=mkind, s=msize)

    ax0.scatter(frame_df['zsmooth'], frame_df['ysmooth'], edgecolor=scatter_color_list,
                facecolor='#00000000', lw=1.0, cmap=cmap, marker=mkind, s=msize)
    ax3.scatter(frame_df['zsmooth'], frame_df['ysmooth'], edgecolor=scatter_color_list,
                facecolor='#00000000', lw=1.0, cmap=cmap, marker=mkind, s=msize)

    ax1.scatter(frame_df['xsmooth'], frame_df['zsmooth'], edgecolor=scatter_color_list,
                facecolor='#00000000', lw=1.0, cmap=cmap, marker=mkind, s=msize)
    ax4.scatter(frame_df['xsmooth'], frame_df['zsmooth'], edgecolor=scatter_color_list,
                facecolor='#00000000', lw=1.0, cmap=cmap, marker=mkind, s=msize)

    # Plot lines
    for label, df2 in df.groupby(by='label'):
        line1 = df2['norm_intensity'].as_matrix()
        line2 = df2['raw_intensity'].as_matrix()
        #line1[line1 < 0] = 0
        if np.max(line1) >= peak_threshold:
            axSignal1.plot(line1, c=color_list[int(label)], alpha=1.0, lw=1.0)
            axSignal2.plot(line2, c=color_list[int(label)], alpha=1.0, lw=1.0)

    if 'Q' in df:
        axSignal1_2 = axSignal1.twinx()
        axSignal2_2 = axSignal2.twinx()
        temp_df = df.groupby(by='t').mean()
        axSignal1_2.plot(temp_df['Q'], alpha=0.5, lw=2, linestyle=':', color='#AA1030')
        axSignal1_2.set_ylim(0, 1.005)
        axSignal1_2.spines['top'].set_visible(False)
        axSignal1_2.spines['bottom'].set_visible(False)
        # axSignal1_2.set_ylabel('Detection quality')

        axSignal2_2.plot(temp_df['Q'], alpha=0.5, lw=2, linestyle=':', color='#AA1030')
        axSignal2_2.set_ylim(0, 1.005)
        axSignal2_2.spines['top'].set_visible(False)
        axSignal2_2.spines['bottom'].set_visible(False)
        # axSignal2_2.set_ylabel('Detection quality')

    # Plot fixes
    axSignal1.set_xticks(np.arange(0, nframes + 1, 10))
    axSignal1_2.set_xticks(np.arange(0, nframes + 1, 10))

    axSignal2.set_xticks(np.arange(0, nframes + 1, 10))
    axSignal2_2.set_xticks(np.arange(0, nframes + 1, 10))

    axSignal1.set_xlim(left=frame - plot_range, right=frame + plot_range)  # set view window
    # axSignal1.set_ylim(bottom=0)  # set view window
    axSignal1.axvline(frame, c='#000000', lw=5, ls='-', alpha=0.25, zorder=-1)
    axSignal1.tick_params(axis='both', direction='in', labelleft='off', labelright='off', labelbottom='off', left='off',
                          right='off')
    axSignal1_2.tick_params(axis='both', direction='in', labelleft='off', labelright='off', labelbottom='off',
                            left='off', right='off')
    axSignal2.set_title('Raw signal')

    axSignal2.set_xlim(left=frame - plot_range, right=frame + plot_range)  # set view window
    axSignal2.set_ylim(bottom=0)  # set view window
    axSignal2.axvline(frame, c='#000000', lw=5, ls='-', alpha=0.25, zorder=-1)
    axSignal2.tick_params(axis='both', direction='in', labelleft='off', labelright='off', labelbottom='off', left='off',
                          right='off')
    axSignal2_2.tick_params(axis='both', direction='in', labelleft='off', labelright='off', labelbottom='off',
                            left='off', right='off')
    axSignal1.set_title('Normalized signal')

    # Setting titles for images
    ax1.set_title('mCherry')
    ax4.set_title('GCaMP')

    # Write marker for stim window
    stim_window = 5

    ylim = axSignal2.get_ylim()[1]
    # Air
    if 49 < frame < 49 + stim_window:
        axSignal2.text(frame - (plot_range) + 1, ylim * 0.9, 'Air', fontsize=18, color='#54c8d8', weight='bold')
    if 129 < frame < 129 + stim_window:
        axSignal2.text(frame - (plot_range) + 1, ylim * 0.9, 'Air', fontsize=18, color='#54c8d8', weight='bold')

    # Oct
    if 89 < frame < 89 + stim_window:
        axSignal2.text(frame - (plot_range) + 1, ylim * 0.9, 'Oct', fontsize=18, color='#f48338', weight='bold')
    if 169 < frame < 169 + stim_window:
        axSignal2.text(frame - (plot_range) + 1, ylim * 0.9, 'Oct', fontsize=18, color='#f48338', weight='bold')

    # Set title
    name = os.path.basename(path)
    plt.suptitle(name[:-4] + '    Frame: ' + str(frame), fontsize=12)

    return fig


def vol_viz(detections='OnlyResponsive', figure_size=20):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import memotrack.display
    import ipywidgets as widgets
    from IPython.display import display
    from ipywidgets import interact, interactive, fixed, interact_manual, Layout

    fly_name = memotrack.display.fly_name
    report_base_path = '/projects/memotrack/temp/report/'
    fig_size = (figure_size, figure_size)

    def display_image(frame):
        fig, ax = plt.subplots(1, figsize=fig_size)
        ax.axis('off')
        img = mpimg.imread(report_base_path + fly_name + '/projections/' + detections + '_{0:03d}.png'.format(frame))
        ax.imshow(img, interpolation='bilinear')

    interact(display_image, frame=widgets.IntSlider(min=0, max=249, step=1, value=90, layout=Layout(width='600px')))

    return


def image(im_path):
    import matplotlib.pyplot as plt

    dpi = 95
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray', interpolation='bilinear')

    plt.show()


def select_fly(files_path):
    from ipywidgets import interact, interactive, fixed, interact_manual, Layout
    import ipywidgets as widgets
    import memotrack
    from IPython.core.display import display, HTML
    from IPython.display import Javascript
    import ipywidgets as widgets
    global fly_name
    import numpy as np

    display(HTML("<style>.container { width:90% !important; }</style>"))
    display(HTML("<style>.widget-label {font-size:17px !important; }</style>"))

    # Get list of paths
    full_path_list = []
    for path in files_path:
        text_file = open(path, 'r')
        temp_full_path_list = text_file.read().split('\n')
        text_file.close()
        full_path_list += temp_full_path_list

    path_list = [path for path in full_path_list if len(path) > 1]

    # get names it from path list
    name_list = []
    for path in path_list:
        name_list.append(path[-17:-11])

    sorted_name_list = list(np.sort(name_list))

    from IPython.display import display
    w = widgets.Dropdown(
        description='Select fly:',
        options=sorted_name_list,
        layout=Layout(width='250px', height='35'))

    display(w)

    fly_name = sorted_name_list[0]

    def run_all(names):
        global fly_name
        fly_name = names['new']
        display(Javascript(
            'IPython.notebook.execute_cell_range(IPython.notebook.get_selected_index()+1, IPython.notebook.ncells())'))
        return fly_name

    w.observe(run_all, names='value')

    return


def fly_filter_params():
    from ipywidgets import interact, interactive, fixed, interact_manual, Layout
    import ipywidgets as widgets
    import memotrack
    from IPython.core.display import display, HTML
    from IPython.display import Javascript
    import ipywidgets as widgets
    global fly_name
    import numpy as np

    w_stim_start = widgets.IntSlider(
        description='Stim start (89 for first OCT)',
        min=0,
        max=249,
        value=89,
        layout=Layout(width='50%'),
        style={'description_width': 'initial'}
    )

    display(w_stim_start)

    w_stim_duration = widgets.IntSlider(
        description='Stim duration (default is 10)',
        min=1,
        max=20,
        value=10,
        layout=Layout(width='50%'),
        style={'description_width': 'initial'}
    )

    display(w_stim_duration)

    w_responsive_threshold = widgets.FloatText(
        description='Responsive threshold',
        value=0.01,
        style={'description_width': 'initial'},
        layout=Layout(width='200px')

    )

    display(w_responsive_threshold)

    w_peak_fold = widgets.FloatText(
        description='Peak fold     ',
        value=2.50,
        style={'description_width': 'initial'},
        layout=Layout(width='200px')

    )

    display(w_peak_fold)

    return


def volcano_plot(path, pvalue=0.01, fold=2.0, GeneID=False, GeneName=False, show_lines=False, gene_list=False,
                 fig=False, ax=False, gene_dict=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import sys

    fold_thresh = fold

    p_value_thresh = -np.log10(pvalue)

    # Load dataframe
    df = pd.read_csv(path, sep='\t', usecols=[0, 1, 2, 3, 4, 5])

    # Fix small values
    thresh = 1.0e-150
    df.loc[df['EpenDiff padj'] < thresh, 'EpenDiff padj'] = thresh

    if fig is False and ax is False:
        print ('Generating new figure...')
        fig, ax = plt.subplots(2, figsize=[9, 12])

    # Fate choice #

    # Log stuff just once:
    df['FateChoice fold change'] = np.log2(df['FateChoice fold change'])
    df['FateChoice padj'] = -np.log10(df['FateChoice padj'])

    # Set p-value color list
    df['FateChoice color'] = '#79C99E90'  # This is the pass filter color
    df['FateChoice size'] = (abs(df['FateChoice padj']) / np.percentile(df['FateChoice padj'], 99)) * (
        abs(df['FateChoice fold change']) / np.percentile(df['FateChoice fold change'], 99)) * 10.0

    df.loc[(df['FateChoice fold change'] < fold_thresh) & (
        df['FateChoice fold change'] > -fold_thresh), 'FateChoice color'] = '#45454525'
    df.loc[df['FateChoice padj'] < p_value_thresh, 'FateChoice color'] = '#45454525'

    # Starting plot
    ax[0].clear()

    sc0 = ax[0].scatter(df['FateChoice fold change'], df['FateChoice padj'],
                        s=df['FateChoice size'].tolist(),
                        c=df['FateChoice color'].tolist())

    ax[0].set_title('Fate choice')
    ax[0].set_xlabel('log2FoldChange')
    ax[0].set_ylabel('-log10padj')
    ax_xrange = max(-np.percentile(df['FateChoice fold change'], 0.01),
                    np.percentile(df['FateChoice fold change'], 99.99))
    ax[0].set_xlim(-ax_xrange - 2, ax_xrange + 2)

    if show_lines:
        ax[0].axvline(fold_thresh, linestyle='--', alpha=0.5)
        ax[0].axvline(-fold_thresh, linestyle='--', alpha=0.5)
        ax[0].axhline(p_value_thresh, linestyle='--', alpha=0.5)

    # EpenDiff #

    # Log stuff just once:
    df['EpenDiff fold change'] = np.log2(df['EpenDiff fold change'])
    df['EpenDiff padj'] = -np.log10(df['EpenDiff padj'])

    # Set p-value color list
    df['EpenDiff color'] = '#79C99E90'  # This is the pass filter color
    df['EpenDiff size'] = (abs(df['EpenDiff padj']) / np.percentile(df['EpenDiff padj'], 99)) * (
        abs(df['EpenDiff fold change']) / np.percentile(df['EpenDiff fold change'], 99)) * 10.0

    df.loc[(df['EpenDiff fold change'] < fold_thresh) & (
        df['EpenDiff fold change'] > -fold_thresh), 'EpenDiff color'] = '#45454525'
    df.loc[df['EpenDiff padj'] < p_value_thresh, 'EpenDiff color'] = '#45454525'

    # Starting plot
    ax[1].clear()

    sc1 = ax[1].scatter(df['EpenDiff fold change'], df['EpenDiff padj'],
                        s=df['EpenDiff size'].tolist(),
                        c=df['EpenDiff color'].tolist())

    ax[1].set_title('Ependimal differentiation')
    ax[1].set_xlabel('log2FoldChange')
    ax[1].set_ylabel('-log10padj')
    ax_xrange = max(-np.percentile(df['EpenDiff fold change'], 0.01), np.percentile(df['EpenDiff fold change'], 99.99))
    ax[1].set_xlim(-ax_xrange - 2, ax_xrange + 2)

    if show_lines:
        ax[1].axvline(fold_thresh, linestyle='--', alpha=0.5)
        ax[1].axvline(-fold_thresh, linestyle='--', alpha=0.5)
        ax[1].axhline(p_value_thresh, linestyle='--', alpha=0.5)

    # Tooltips
    global annot0, annot1
    annot0 = ax[0].annotate('', xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc='w'),
                            arrowprops=dict(arrowstyle="->"))

    annot1 = ax[1].annotate('', xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc='w'),
                            arrowprops=dict(arrowstyle="->"))

    annot0.set_visible(False)
    annot1.set_visible(False)

    names = df['Name']
    IDs = df['Gene ID']

    global last_color, last_ind
    last_color = False
    last_ind = False

    def update_annot(ind, accent_color='#23B5D3AA', Tooltip=True):
        global annot0, annot1, last_ind, last_color
        # ax[0].set_title(accent_color)

        # convert Hex color to RGBA
        value = accent_color
        value = value.lstrip('#')
        lv = len(value)
        accent_color = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        accent_color = tuple(np.divide(accent_color, 255.0))

        if Tooltip:

            annot0.set_visible(True)
            annot1.set_visible(True)

            for i in range(len(ind['ind'])):
                pos0 = sc0.get_offsets()[ind['ind'][i]]
                annot0.xy = pos0
                pos1 = sc1.get_offsets()[ind['ind'][i]]
                annot1.xy = pos1

                # text = '{} ({})'.format(', '.join([names[n] for n in ind['ind']]),
                #                        ', '.join([IDs[n] for n in ind['ind']]))

                text = '{} ({})'.format(names[ind['ind'][i]], IDs[ind['ind'][i]])
                annot0.set_text(text)
                annot1.set_text(text)

        last_color0 = [n for n in sc0._facecolors[ind['ind'], :]]
        last_color1 = [n for n in sc1._facecolors[ind['ind'], :]]

        sc0._facecolors[ind['ind'], :] = accent_color
        sc1._facecolors[ind['ind'], :] = accent_color

        if last_ind:
            sc0._facecolors[last_ind] = last_color0[0]
            sc1._facecolors[last_ind] = last_color1[0]

        # last_ind = ind['ind']
        if Tooltip:
            last_ind = [n for n in ind['ind']]

    def onclick(event):

        global annot0, annot1
        vis0 = annot0.get_visible()
        vis1 = annot1.get_visible()

        if event.inaxes == ax[0]:
            cont, ind = sc0.contains(event)
            if cont:
                update_annot(ind)

                fig.canvas.draw_idle()
            else:
                if vis0:
                    annot0.set_visible(False)
                    fig.canvas.draw_idle()

        if event.inaxes == ax[1]:
            cont, ind = sc1.contains(event)
            if cont:
                update_annot(ind)

                fig.canvas.draw_idle()
            else:
                if vis1:
                    annot1.set_visible(False)
                    fig.canvas.draw_idle()

    if GeneID:
        GeneID_ind = {'ind': df[df['Gene ID'] == GeneID].index.values}
        update_annot(GeneID_ind)

    if GeneName:
        GeneName_ind = {'ind': df[df['Name'] == GeneName].index.values}
        update_annot(GeneName_ind)

    if gene_list:
        # print (gene_dict[gene_list])
        gene_list_ID = {'ind': df[df['Name'].isin(gene_dict[gene_list])].index.values}
        # print (gene_list_ID)
        update_annot(gene_list_ID, accent_color='#D72483AA', Tooltip=False)

    # Connect figure
    fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show(fig)
    return


def interact_volcano_plot(path, gene_list_path=False):
    from ipywidgets import interact, interactive, fixed, interact_manual, Layout
    from IPython.core.display import display, HTML
    from IPython.display import Javascript
    import ipywidgets as widgets
    import memotrack.display
    import matplotlib.pyplot as plt
    import sys
    import numpy as np

    fig, ax = plt.subplots(2, figsize=[9, 12])

    # load Genes list
    if gene_list_path:
        gene_dict = {}
        gene_dict[''] = np.nan
        with open(gene_list_path) as f:

            for line in f:
                gene_dict[line.split('\t')[0]] = line.split('\t')[2:-1]

    interact(memotrack.display.volcano_plot,
             pvalue=widgets.FloatText(description='p-value:', value=0.01),
             fold=widgets.FloatText(description='log2fold:', value=2.0),
             GeneID=widgets.Text(description='Gene ID'),
             GeneName=widgets.Text(description='Gene Name'),
             show_lines=widgets.Checkbox(description='Display threshold lines', value=False),
             gene_list=widgets.Dropdown(description='Gene list', options=gene_dict.keys()),
             path=fixed(path),
             fig=fixed(fig),
             ax=fixed(ax),
             gene_dict=fixed(gene_dict))

    return


def PCA(features_df, normalization=False, plot_ignore=True, remove_ignore=True):
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    temp_df = features_df.copy(deep=True)

    if remove_ignore:
        temp_df = temp_df[temp_df['group'] != 'ignore'].copy(deep=True)

    colors = temp_df['color'].tolist()
    fly_names = temp_df['fly'].tolist()
    group = temp_df['group'].tolist()

    # Delete thing we don't want in the PCA
    del temp_df['fly']
    del temp_df['group']
    del temp_df['color']

    plt.figure(figsize=(14, 14))
    temp_df.fillna(value=0, inplace=True)

    if normalization == 'mean':
        temp_df = (temp_df - temp_df.mean()) / temp_df.std()

    if normalization == 'minmax':
        temp_df = (temp_df - temp_df.min()) / (temp_df.max() - temp_df.min())

    temp_df.fillna(value=0, inplace=True)
    pca = PCA(n_components=2)

    X_reduced = pca.fit_transform(temp_df)

    i = 0
    for coords in X_reduced:
        plt.scatter(coords[0], coords[1], c=colors[i], label=group[i], s=0)
        i += 1

    i = 0
    for point_pos in range(len(X_reduced)):
        if group[i] == 'ignore' and plot_ignore:

            plt.annotate(fly_names[i], xy=(X_reduced[i][0], X_reduced[i][1]), color="#FFFFFF", fontweight='bold',
                         size=10, bbox=dict(boxstyle='round, pad=0.5', fc=colors[i], ec=colors[i], alpha=0.9))

        elif group[i] != 'ignore':

            plt.annotate(fly_names[i], xy=(X_reduced[i][0], X_reduced[i][1]), color="#FFFFFF", fontweight='bold',
                         size=10, bbox=dict(boxstyle='round, pad=0.5', fc=colors[i], ec=colors[i], alpha=0.9))
        i += 1

    plt.xlabel('First component (explained variance: {:.2%})'.format(pca.explained_variance_ratio_[0]))
    plt.ylabel('Second component (explained variance: {:.2%})'.format(pca.explained_variance_ratio_[1]))

    plt.show()


def feature_vs_feature(features_df, feature1, feature2, normalization=False):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    temp_df = features_df.copy(deep=True)
    colors = temp_df['color'].tolist()
    fly_names = temp_df['fly'].tolist()
    group = temp_df['group'].tolist()

    # Delete thing we don't want in the PCA
    del temp_df['fly']
    del temp_df['group']
    del temp_df['color']

    plt.figure(figsize=(14, 14))
    temp_df.fillna(value=0, inplace=True)

    if normalization == 'mean':
        temp_df = (temp_df - temp_df.mean()) / temp_df.std()

    if normalization == 'minmax':
        temp_df = (temp_df - temp_df.min()) / (temp_df.max() - temp_df.min())

    values1 = temp_df[feature1].values
    values2 = temp_df[feature2].values

    X_reduced = np.transpose([values1, values2])

    i = 0
    for coords in X_reduced:
        plt.scatter(coords[0], coords[1], c=colors[i], label=group[i], s=0)
        i += 1

    i = 0
    for point_pos in range(len(X_reduced)):
        plt.annotate(fly_names[i], xy=(X_reduced[i][0], X_reduced[i][1]), color="#FFFFFF", fontweight='bold',
                     size=10, bbox=dict(boxstyle='round, pad=0.5', fc=colors[i], ec=colors[i], alpha=0.9))
        i += 1

    plt.xlabel('{}'.format(feature1))
    plt.ylabel('{}'.format(feature2))
    plt.title('Features comparison: {} vs {}'.format(feature1, feature2))

    plt.show()


def compare_distributions(data_group, label_group, bw='scott', log=False, high_pass=False, save=False):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    plt.figure(figsize=(14, 8))
    color_list = ['#C3C4BE', '#05D0FF', '#05BBE6', '#C06FC9', '#C06FC9']
    xmin = min(np.hstack(data_group))
    xmax = max(np.hstack(data_group))

    i = 0
    for data in data_group:
        data = data.astype('float')

        ax = plt.gca()

        if high_pass:
            data = [d for d in data if d > high_pass]
            plt.axvline(high_pass, color='#BABABA', lw=3)
            #ax.set_xlim([high_pass, xmax])

        print ('-> {}\tPeaks: {}'.format(label_group[i], len(data)))

        density = gaussian_kde(data, bw_method=bw)

        xs = np.linspace(xmin, xmax*1.2, 1000)

        if log:
            ys = density.logpdf(xs)
            ax.set_ylim([-10, 1.2*max(ys)])
        else:
            ys = density.pdf(xs)

        if label_group[i][0] == 'U':
            style = '--'
        else:
            style = '-'

        ax.plot(xs, ys, label='{}   (Peaks: {})'.format(label_group[i], len(data)),
                lw=2, alpha=0.8, color=color_list[i], linestyle=style)

        i += 1

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.legend(frameon=False)

    if save:
        plt.savefig(save)
        plt.close('all')
    else:
        plt.show()

    return
