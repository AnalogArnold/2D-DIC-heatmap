# Overview
This repository provides Python code for processing and analyzing 2D Digital Image Correlation (DIC) displacement data exported from DaVis 10 (by LaVision GmbH). The primary goal is to quantify the noise floor in static DIC experiments by computing statistical measures and visualizing the data.

**Note:** This code was originally developed for a specific PhD experiment but it can be adapted for other DIC datasets with a similar structure. Depending on the output file, it may also work with files from other DIC-processing software. See below.

An example of a DIC shot:
![example DIC frame](/images/test_view.PNG)
If you want to learn more about this technique, you can start with [Photo-Sonics' introduction to DIC](https://photo-sonics.co.uk/technical-library/digital-image-correlation-dic/).

# Features
* **Statistical analysis:** Uses pandas to generate descriptive statistics (mean, min, max, etc.) to show the central tendencies of each tested case.
* **Visualization:** 
  * Line plots of mean displacement and standard deviation of the x-displacement over time or frames.
  * 2D heatmaps:
    * Mean x-displacement (average noise)
    * Standard deviation of x-displacement (temporal noise floor)
    * Mean y-displacement (optional; not of interest in my case but I added it anyway)
* **Batch processing:** Assuming a consistent naming convention, it can handle all CSV files in a selected directory and infer the framerate from its name.
* **Easy for an inexperienced user:** Unless major changes are needed, all input variables (e.g., the pixel/mm factor from the DIC calibration) are at the top of the code.
* **Measures to handle bad input data:** The code is written to handle cases such as incorrectly named files, duplicates, exporting more quantities than needed... If I make any more mistakes while saving data tired after hours of testing, I will add measures to handle these, too.
## Outputs
1. 2D heatmaps
  ![example of 2D heatmaps](/images/outputs1.png)
2. Line plots of statistical data
![example of statistical data plots](/images/outputs2.png)
*Please note that I used random data from failed tests to show the outputs, so if these plots don't look right, it's not because of the code (sadly).*
3. Statistical summary CSV in the parent directory
![example of output CSV file](/images/outputs3.png)
# Requirements
  (Or at least the versions I used while coding it and I cannot guarantee compatibility otherwise)
  * Python 3.12
  * pandas 2.3.0
  * seaborn 0.13.2
  * easygui 0.98.3
  * numpy 1.26.4
  * [smplotlib 1.0.0](https://github.com/AstroJacobLi/smplotlib)
    * Used to make the graphs look "old-school" (see the linked repository), so if this is not your cup of tea, feel free to remove it. No other changes in code are needed.
  * $${\color{red}IMPORTANT!}$$ matplotlib **3.7.3**
    * This is **critical for the heatmaps**. Newer versions change subplot sharing behavior and will break heatmap plots as implemented. You can see this in effect below:
    ![comparison of heatmaps plotted using matplotlib 3.7.3 and newer](/images/matplotlib_version_info.png)
To install dependencies, you can use:
```
pip install matplotlib==3.7.3 pandas numpy seaborn easygui
```
# Usage
## 1. Prepare your data
Export DIC displacement data from DaVis 10 as CSV files. Each frame will automatically be saved as a separate CSV file in the parent folder.
$${\color{red}IMPORTANT!}$$ While naming your files, keep the naming convention:
```
sampleCode_testCode_framerate_attemptCode
```
to get the most out of the automation features. For example, my folder would be called S1_static01_60fps_a.

The output folder (which is the input folder for this code) will look like this:

 ![example of the output folder form DaVis](/images/input_folder_structure.png)
 
 And the files inside should look like this:
 
  ![example of the output file form DaVis](/images/input_file_example.png)
  
However, if you accidentally chose more quantities (auto-selecting all of them is too easy to do accidentally), this is not a problem.
## 2. Run the script
1. A GUI will prompt you to select the directory containing your CSV files.
   * Processing one test at a time is purposeful to look at the plots as the data is processed and potentially identify any patterns, issues, etc. However, it is possible to automate this code to select all of them.
2. Set `PIXEL_PER_MM` at the top of `NoiseFloorAnalyser` using the px/mm scale factor from your DIC calibration.
3. Adjust `COLUMN_NAMES` if your CSV export uses different headers/
4. Plot options: toggle (set to True/False) `PLOT_HEATMAPS`, `PLOT_LINE_PLOTS`, `LINE_PLOTS_IN_PX` to control the output graphs.
If your filename contains the framerate in it, the line plots will automatically display time as the temporal value on the *y* axis as this is normally preferred.

# Limitations and notes
* **Error handling:** Currently limited to things that *I*, as an author and the primary user, have done wrong while handling the data (or expect I would). This script by no means covers all bases that would have to be considered with a wider user base. The script skips files with errors but will notify you in the console.
* **File consistency:** All files in the selected directory must follow the same naming and structure. A manual bypass is included and commented out, but this is a workaround for a single case (e.g., a typo) rather than something that should be used for multiple tests.
* **Matplotlib version:** Only 3.7.3 is supported due to the subplot axis sharing requirements in heatmaps (`sharex` and `sharey` do not do what is needed there).
* **Many (perhaps *too* many) comments:** This is for two reasons:
  * This code was written for me and my labmates who, for the most part, do not have much coding experience and I wanted this to be as understandable for them as possible.
  * Plotting heatmaps and using classes (version 2.0) was something new to me at the time, so I kept these comments for future reference.
# License
MIT License
Copyright (C) 2025 AnalogArnold (Michael Darcy)

If you use or adapt this code, please cite or acknowledge the author.
