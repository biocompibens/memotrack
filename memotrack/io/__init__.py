# Reads stack from Tiff using bioformats
def read_bioformats(path, meta, frame=0, channel=0, verbose=True):
    """
    Reads using bioformats one stack from the Tiff on a 3D Numpy array (ZYX).

    path: io path
    meta: metadata dictionary extracted with memotrack.io.meta
    frame: Time frame to extract the stack
    """
    import javabridge
    import bioformats
    import sys
    import os
    import logging
    import numpy as np

    # Disable the warnings from bioformats
    logging.disable('WARNING')

    # Get meta info in int
    SizeX = int(meta['SizeX'])
    SizeY = int(meta['SizeY'])
    SizeZ = int(meta['SizeZ'])
    SizeT = int(meta['SizeT'])
    SizeC = int(meta['SizeC'])

    # Create numpy array to fill
    img = np.zeros([SizeZ, SizeY, SizeX], 'uint16')

    if verbose:
        if frame < 10:  # Little trick here just to maintain the alignment of output
            print ('\nReading frame ' + str(frame) + '  | ZYX:' + str((np.shape(img)))),
        else:
            print ('\nReading frame ' + str(frame) + ' | ZYX:' + str((np.shape(img)))),
        sys.stdout.flush()

    # Start JVM for bioformats
    javabridge.start_vm(class_path=bioformats.JARS, run_headless=True, max_heap_size='2048M')

    # Stop system output, to avoid bioformats warnings
    _stderr = sys.stderr
    _stdout = sys.stdout
    null = open(os.devnull, 'wb')
    sys.stdout = sys.stderr = null

    for slice in range(SizeZ):
        img[slice] = bioformats.load_image(path=path, c=channel, z=slice, t=frame, series=None, index=None,
                                           rescale=False,
                                           wants_max_intensity=False, channel_names=None)

    # Return system output
    sys.stderr = _stderr
    sys.stdout = _stdout

    if verbose:
        print (' [Done]')
        sys.stdout.flush()

    # Kill java virtual machine
    # javabridge.detach()

    return img


def read(path, meta, frame=0, channel=0, verbose=True):
    """
    STILL NOT WORKING !!!!
    Reads using Tifffile one stack from the Tiff on a 3D Numpy array (ZYX).
    :param verbose:
    :param channel:
    :param frame:
    :param meta:
    :param path:
    """
    import tifffile
    import sys
    import numpy as np

    # Get meta info in int
    SizeX = int(meta['SizeX'])
    SizeY = int(meta['SizeY'])
    SizeZ = int(meta['SizeZ'])
    if meta['SizeT']:
        SizeT = int(meta['SizeT'])
    else:
        SizeT = 1

    if meta['SizeC']:
        SizeC = int(meta['SizeC'])
    else:
        SizeC = 1

    # Create numpy array to fill
    img = np.zeros([SizeZ, SizeY, SizeX], 'uint16')

    if verbose:
        if frame < 10:  # Little trick here just to maintain the alignment of output
            print ('\nReading frame ' + str(frame) + '  | ZYX:' + str((np.shape(img)))),
        else:
            print ('\nReading frame ' + str(frame) + ' | ZYX:' + str((np.shape(img)))),
        sys.stdout.flush()

    with tifffile.TiffFile(path, fastij=True) as tif:
        img_array = tif.page[0].asarray()

    img = img_array[frame, :, channel, :, :]

    if verbose:
        print (' [Done]')
        sys.stdout.flush()

    return img


def load_full_img(path, verbose=True):
    """
    Reads using Tifffile one stack from the Tiff on a 3D Numpy array (TZCYX).
    :param verbose:
    :param channel:
    :param frame:
    :param meta:
    :param path:
    """
    import tifffile
    import sys
    import numpy as np

    if verbose:
        print ('Loading the full image...'),
        sys.stdout.flush()

    with tifffile.TiffFile(path, fastij=True) as tif:
        img = tif.asarray()

    size = (sys.getsizeof(img))/float(1024**3)
    if verbose:
        print (' [Done]')
        print (np.shape(img))
        print ('Total of {:.2f} GB'.format(size))
        sys.stdout.flush()

    return img


# Read the metadata from the io
def meta_bioformats(path, verbose=True):
    """
    Function for reading io metadata. Reads a path for a Tiff io

    Allows to read the metadata from the Tiff io via the bioformats standard.
    It returns a python dictionary with all the information, and gives a warning in case a relevant one is missing
    (the dictionary is generated anyways)

    Default PhysicalSize for X and Y is 0.16125
    """

    # Imports for bioformats
    import javabridge
    import bioformats
    import xml.etree.ElementTree as ET

    # Start JVM for bioformats
    javabridge.start_vm(class_path=bioformats.JARS, run_headless=False, max_heap_size='2048M')

    # Read Metadata
    meta_xml = bioformats.get_omexml_metadata(path=path)

    # Creates the XML structure from the metadata string
    root = ET.fromstring(meta_xml.encode('utf-8'))

    # Get the dictionary with image metadata
    # meta = root[0][2].attrib    # Original line, working on local
    meta = root[0][1].attrib  # Working on remote (new version of something ?)

    # Convert some values to int
    meta['SizeT'] = int(meta['SizeT'])
    meta['SizeZ'] = int(meta['SizeZ'])
    meta['SizeY'] = int(meta['SizeY'])
    meta['SizeX'] = int(meta['SizeX'])
    meta['SizeC'] = int(meta['SizeC'])

    # Check metadata
    checklist = ['PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ', 'SizeX', 'SizeY', 'SizeZ', 'SizeT', 'SizeC', 'Type']

    if verbose:
        for item in checklist:
            if item in meta:
                print (item + ': ' + str(meta[item]))
            else:
                print('*** Metadata problem, missing ' + item)

    # Kill java virtual machine
    # javabridge.detach()

    return meta


# Read the metadata from the io
def meta(path, verbose=True):
    """
    Function for reading io metadata. Reads a path for a Tiff io

    Allows to read the metadata from the Tiff io via the bioformats standard.
    It returns a python dictionary with all the information, and gives a warning in case a relevant one is missing
    (the dictionary is generated anyways)

    Default PhysicalSize for X and Y is 0.16125
    """
    import tifffile

    img = tifffile.TiffFile(path)

    x_resolution = img.pages[0].tags['x_resolution'].value
    y_resolution = img.pages[0].tags['x_resolution'].value

    meta = {}
    meta['PhysicalSizeX'] = 1 / (float(x_resolution[0]) / float(x_resolution[1]))
    meta['PhysicalSizeY'] = 1 / (float(y_resolution[0]) / float(y_resolution[1]))
    meta['PhysicalSizeZ'] = img.pages[0].imagej_tags.get('spacing')
    meta['SizeX'] = img.pages[0].tags['image_width'].value
    meta['SizeY'] = img.pages[0].tags['image_length'].value
    meta['SizeZ'] = img.pages[0].imagej_tags.get('slices')
    meta['SizeT'] = img.pages[0].imagej_tags.get('frames')
    meta['SizeC'] = img.pages[0].imagej_tags.get('channels')
    meta['Type'] = img.pages[0].tags['bits_per_sample'].value

    # Check metadata
    checklist = ['PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ', 'SizeX', 'SizeY', 'SizeZ', 'SizeT', 'SizeC', 'Type']

    if verbose:
        for item in checklist:
            if item in meta:
                print (item + ': ' + str(meta[item]))
            else:
                print('*** Metadata problem, missing ' + item)

    return meta


# Write image to disk
def write(img, path, dtype='uint16', verbose=True):
    """
    Write img as TIF on disk
    :param img: Numpy ZYX stack
    :param path: Full path (with filename) to save
    :param type: byte type to save
    :param verbose: Print process
    :return: Nothing, saves file to disk
    """

    import bioformats.omexml as ome
    import javabridge as jutil
    import bioformats
    import numpy as np
    import os
    import sys

    # Memotrack works on TZYX, swaps to XYZT, for the standard.
    # memotrack numpy sequence T[0]Z[1]Y[2]X[3]
    img_swap = np.swapaxes(img, 0, 3)  # Here swapping T and X: TZYX -> XZYT
    img_swap = np.swapaxes(img_swap, 1, 2)  # Now on Z and Y: XZYT -> XYZT

    # Inserting the C dimension (to get the XYCZT)
    img_XYCZT = np.expand_dims(img_swap, 2)

    if verbose:
        print('Dimensions (XYCZT): ' + str(np.shape(img_XYCZT)))
        sys.stdout.flush()

    # Get the new dimensions
    SizeX = np.shape(img_XYCZT)[0]
    SizeY = np.shape(img_XYCZT)[1]
    SizeC = np.shape(img_XYCZT)[2]
    SizeZ = np.shape(img_XYCZT)[3]
    SizeT = np.shape(img_XYCZT)[4]

    # Start JVM for bioformats
    jutil.start_vm(class_path=bioformats.JARS)

    # Getting metadata info
    omexml = ome.OMEXML()
    omexml.image(0).Name = os.path.split(path)[1]
    p = omexml.image(0).Pixels
    assert isinstance(p, ome.OMEXML.Pixels)
    p.SizeX = SizeX
    p.SizeY = SizeY
    p.SizeC = SizeC
    p.SizeT = SizeT
    p.SizeZ = SizeZ
    p.DimensionOrder = ome.DO_XYCZT
    p.PixelType = dtype
    p.channel_count = SizeC
    p.plane_count = SizeZ
    p.Channel(0).SamplesPerPixel = SizeC
    omexml.structured_annotations.add_original_metadata(ome.OM_SAMPLES_PER_PIXEL, str(SizeC))

    # Converting to omexml
    xml = omexml.to_xml()

    # Write file using Bioformats
    if verbose:
        print ('Writing frames:'),
        sys.stdout.flush()

    for frame in range(SizeT):
        if verbose:
            print('[' + str(frame + 1) + ']'),
            sys.stdout.flush()

        index = frame

        pixel_buffer = bioformats.formatwriter.convert_pixels_to_buffer(img_XYCZT[:, :, :, :, frame], dtype)

        script = """
        importClass(Packages.loci.formats.services.OMEXMLService,
                    Packages.loci.common.services.ServiceFactory,
                    Packages.loci.formats.out.TiffWriter);

        var service = new ServiceFactory().getInstance(OMEXMLService);
        var metadata = service.createOMEXMLMetadata(xml);
        var writer = new TiffWriter();
        writer.setBigTiff(true);
        writer.setMetadataRetrieve(metadata);
        writer.setId(path);
        writer.setInterleaved(true);
        writer.saveBytes(index, buffer);
        writer.close();
        """
        jutil.run_script(script, dict(path=path, xml=xml, index=index, buffer=pixel_buffer))

    if verbose:
        print ('[Done]')
        sys.stdout.flush()

    if verbose:
        print('File saved on ' + str(path))
        sys.stdout.flush()

        # Kill java virtual machine
        # jutil.kill_vm()


# Write detections as pandas dataframe
def write_df(detections_df, path):
    """
    Write the pandas dataframe to disk as a csv file
    :param detections_df: pandas dataframe to write
    :param path: full path, ending with .csv
    :return:
    """
    detections_df.to_csv(path, index_label=False)


# Read data saved as pandas data frame
def read_df(path, no_index=False):
    """
    Reads the csv file as a pandas dataframe
    :param path: file path
    :return: pandas dataframe
    """
    import pandas as pd
    if no_index:
        new_df = pd.read_csv(path, index_col=False)
    else:
        new_df = pd.read_csv(path)
    return new_df
