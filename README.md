# Memotrack
Detection and tracking of neurons for in vivo analysis of memory traces

## Instalation
1) Set the folder "memotrack" within your path for Python packages
2) Check if you have all the dependencies by ``pip install -r requirements.txt``
3) The package can be loaded via `import memotrack`

## Basic usage
The image must be a 2 channel Tiff file, with the first channel being the nuclei and the second channel the neuron signal. FWHMxy is the Full Width at Half Maximum of the desired object to be detected (on the first channel). 

The following command starts the analysis:

```
import memotrack
memotrack.run(image_path, verbose=True, FWHMxy=1.21)
```

## Sample images
For testing purposes, sample images can be found [here](https://cloud.biologie.ens.fr/index.php/s/tzbIwlpW5FM5qaR).
