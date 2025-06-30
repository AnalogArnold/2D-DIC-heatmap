##################################################################################################################
#
# Code written to process 2D DIC displacement data from DaVis 10 (by LaVision GMBH) to quantify the noise floor.
#
# Functions:
# - Find statistics of each attempt;
# - Plot:
#   - Standard deviation of x-displacement (temporal noise floor) over time/images;
#   - Mean of x-displacement over time/images, including the standard deviation as the shadowed area;
#   - 2D heatmaps of the mean x-displacement and standard deviation of x-displacement across all images taken, which
#     represent the average and temporal noise floor.
#
# Note that this code is written for the specific experiment during my PhD studies, but it should be easy to adapt it to
# other experiments.
# Naming conventions: the data from each attempt is stored under sample code_attempt code_frames per second_a/b.
# For example, Processed -> S1_static01_60fps_a -> from S1_static01_60fps_a 0001.csv to C1static04a0250.csv for 250
# images taken in an attempt.
# In this case it made sense to process each attempt separately rather than automating the entire process so I could
# notice any abnormalities as I went.
#
# NOTE: Requires matplotlib 3.7.3. Newer versions require the use of sharex and sharey in the heatmap subplots,
# which simply do not work the same way as get_shared_y_axes().join() and the plots do not look as intended.
#
# Version: 2.0 (June 2025)
# Author: Michael Darcy
# License: MIT
# Copyright (C) 2025 AnalogArnold
#
##################################################################################################################

import os
import pandas as pd
import easygui
import matplotlib
if matplotlib.__version__ != '3.7.3':
    print('WARNING: Matplotlib version is not 3.7.3. This will cause issues with the heatmap plots.')
else:
    from matplotlib import pyplot as plt
import smplotlib
import numpy as np
import seaborn as sns

#######################################################################################################################

class NoiseFloorAnalyser:
    # Constants - REQUIRE USER INPUT
    PIXEL_PER_MM = 44.39 # px/mm for conversion of units; it should be taken from the DIC calibration
    # Toggles to select what we want to plot
    PLOT_HEATMAPS = True
    PLOT_LINE_PLOTS = True
    LINE_PLOTS_IN_PX = False  # Plot the line plots in pixels if True, otherwise in mm
    HEATMAP_TICK_JUMP = 4 # Integer. How many ticks to skip between each tick on the heatmap.
    # Define the columns we will read into the software. This is not necessary but helpful if you accidentally export
    # quantities you're not really interested in (very easy to auto-select in DaVis)
    COLUMN_NAMES = ['x [mm]', 'y [mm]', 'x-displacement [mm]', 'y-displacement [mm]', 'Displacement [mm]']

    # Change the below only if interested in something other than the x-displacement
    def __init__(self):
        self.csv_attempt_data = [] # BEFORE CALLED attempt_data
        self.x_displacement_mean = [] # Mean value of x-displacement for every image (so mean of each CSV)
        self.x_std = [] # Standard deviation of x-displacement for every image
        self.directory_path = None
        self.directory_name = None
        self.top_directory = None
        self.time_per_image = 0

    def select_directory(self):
        """Select and setup working directory, the directory to save the output CSV, and the base for the CSV
        filenames as these use the same naming convention.
        For example, we would have:
        Processed -> S1_static01_60fps_a -> from S1_static01_60fps_a 0001.csv to C1static04a0250.csv.
        where S1 - sample code, static01 - attempt code, 60fps - frame rate, a/b - attempt code, 0001 - image number."""
        # Open GUI to allow user to select the directory with the attempt (e.g., S1_static01_60fps_a)
        try:
            self.directory_path = easygui.diropenbox()
            self.directory_name = os.path.basename(self.directory_path) # Directory name is the base for the CSV filenames
            self.top_directory = os.path.dirname(self.directory_path) # Directory containing the folder to save the output CSV
            self._set_time_per_image()
            # temp_name = 'S1_static01_1000fps_ah' # Use if you make a mistake naming of the files at some point and replace with whatever the actual name is
        except:
            print('An exception occured while opening the directory. It is most likely due to an issue in the naming or having duplicates - please check and use the temp_file route if needed.')
            return

    def _set_time_per_image(self):
        """Set time per image based on fps in directory name. This function has been rewritten using regular expressions
        to avoid having to hard-code the framerate used in the test."""
        # Get the list of strings comprising the filename. E.g., ['S1', 'static01', '800fps', 'a']
        split_directory_name = self.directory_name.split(sep='_')
        # Find the first item in the list that contains the string 'fps'. None is the default value.
        fps_item = next((item for item in split_directory_name if 'fps' in item), None)
        # Extract the number preceding 'fps' to get the framerate if encoded in the filename
        if fps_item:
            match = re.search(r'\d+', fps_item) # Find first sequence of digits in the string
            if match:
                fps_value = int(match.group())
                fps_to_time = 1/fps_value # Convert knowing that X fps corresponds to images taken every 1/X second
                self.time_per_image = fps_to_time

    def process_csv_files(self):
        """Process all CSV files in directory"""
        # Count the number of files in the directory, so the number of images taken per test doesn't need to be hard-coded
        file_count = len(os.listdir(self.directory_path)) # Note: This will also count subdirectories, so be careful!
        # Open all CSV files in the directory (one per image taken) and store the data
        for i in range(1, file_count + 1):
            # DaVis encodes filenames in format 0000, so e.g., 0001, 0099. We want to make sure we add a sufficient
            # number of 0s to the front (left) to get the appropriate filename:
            file_number = str(i).zfill(4)
            try:
                # filepath = directory_path + '\\' + directory_name + file_number + '.csv' # Uglier but easier to
                # comprehend way of doing the line below. Left here for the sake of learing
                filepath = os.path.join(self.directory_path, f"{self.directory_name}{file_number}.csv")
                # filepath = directory_path + '\\' + temp_name + number_appendix + '.csv' # Use if you make a mistake naming of the files at some point
                # Encoding is mbcs because with some quantities, DaVis uses ANSI instead of UTF-8 and Python returns an error
                data = pd.read_csv(filepath, usecols=self.COLUMN_NAMES, encoding='mbcs')
                # Add the image data to a list
                self.csv_attempt_data.append(data)
            except Exception as e:
                print(f"Error processing file {filepath}: {str(e)}. Make sure that your folder contains only .csv files with a consistent naming convention.")
                print("Skipping to next file...")
                continue

    def calculate_statistics(self):
        """Take a list of dataframes (one per image taken) and returns a dataframe with statistical analysis of
        the data, which is saved to a CSV file. Also converts the values from px to mm using the scaling factor."""
        # Concatenate data frames from all images in this attempt into one, so we can run statistical analysis on it easily
        df_all_images = pd.concat(self.csv_attempt_data)
        # Convert the values from px to mm using the scaling factor
        df_all_images['x-displacement [px]'] = df_all_images[self.COLUMN_NAMES[2]] * self.PIXEL_PER_MM
        df_all_images['y-displacement [px]'] = df_all_images[self.COLUMN_NAMES[3]] * self.PIXEL_PER_MM
        columns_to_drop = ['x [mm]', 'y [mm]'] # We do not need statistics of the positions on the grid
        # Analysis including mean, min, max, std, etc. Percentiles are not relevant so drop them.
        stat_analysis = df_all_images.describe(percentiles=[]).drop(columns=columns_to_drop)
        # Save to CSV file
        output_path = os.path.join(self.top_directory, f"{self.directory_name} stat analysis.csv")
        stat_analysis.to_csv(output_path, mode='w', index=True, header=True)
        return df_all_images

    def create_displacement_map(self, df_all_images):
        """Create a 2D displacement map:
        Get all x and y positions for the grid from the first image, assuming that these will remain constant in the
        other images (they should for static noise floor tests).
        Use the fact that sets, unlike lists, do not allow duplicate members so we remove the repeats stemming from
        the instances where the same x value appears again next to a different y-value in our dataset. Then we turn
        it back into a list and sort it since sets are not ordered."""
        if self.PLOT_HEATMAPS:
            x_coords = sorted(list(set(self.csv_attempt_data[0][self.COLUMN_NAMES[0]])))
            y_coords = sorted(list(set(self.csv_attempt_data[0][self.COLUMN_NAMES[1]])))

            grid_data = self._find_grid_data(df_all_images, x_coords, y_coords)
            self._plot_displacement_maps(grid_data)

    def _find_grid_data(self, df_all_images, x_coords, y_coords):
        """Calculate grid data for displacement maps"""
        positions = [(x, y) for x in x_coords for y in y_coords]
        grid_data = {
            'x position [mm]': [], 'y position [mm]': [],
            'mean x displacement [px]': [], 'mean y displacement [px]': [],
            'U_x std [px]': []
        }
        # Iterate through all x and y positions and find the mean displacement for each of them.
        # Find the rows for given x and y coordinates in ALL images taken, so we can use it for finding the mean
        # displacement for this spot. The number of results should be equal to the number of images taken if
        # everything works correctly.
        for x, y in positions:
            result = df_all_images.loc[
                (df_all_images[self.COLUMN_NAMES[0]] == x) &
                (df_all_images[self.COLUMN_NAMES[1]] == y)
                ]
            grid_data['x position [mm]'].append(round(x, 2))
            grid_data['y position [mm]'].append(round(y, 2))
            # Find the mean of the x/y displacements at the given x and y position in all images taken
            grid_data['mean x displacement [px]'].append(result['x-displacement [px]'].mean())
            grid_data['mean y displacement [px]'].append(result['y-displacement [px]'].mean())
            # Standard deviation of x-displacement at (x,y) in all images taken
            grid_data['U_x std [px]'].append(result['x-displacement [px]'].std())
        return grid_data

    def _plot_displacement_maps(self, grid_data):
        """Plot displacement maps for x and y, and standard deviation in x"""
        grid_x_df = pd.DataFrame({
            'x position [mm]': grid_data['x position [mm]'],
            'y position [mm]': grid_data['y position [mm]'],
            'mean x displacement [px]': grid_data['mean x displacement [px]']
        })
        grid_y_df = pd.DataFrame({
            'x position [mm]': grid_data['x position [mm]'],
            'y position [mm]': grid_data['y position [mm]'],
            'mean y displacement [px]': grid_data['mean y displacement [px]']
        })
        grid_x_stds_df = pd.DataFrame({
            'x position [mm]': grid_data['x position [mm]'],
            'y position [mm]': grid_data['y position [mm]'],
            'U_x std [px]': grid_data['U_x std [px]']
        })
        #  Reshape data based on column values so it can be sent to seaborn for plotting.
        #  Index goes on y-axis, columns go on the x-axis.
        mean_x_map = grid_x_df.pivot(
            index='y position [mm]',
            columns='x position [mm]',
            values='mean x displacement [px]'
        )
        mean_y_map = grid_y_df.pivot(
            index='y position [mm]',
            columns='x position [mm]',
            values='mean y displacement [px]'
        )
        x_stds_map = grid_x_stds_df.pivot(
            index='y position [mm]',
            columns='x position [mm]',
            values='U_x std [px]')
        self._create_and_save_heatmap_mean_and_std(mean_x_map, x_stds_map, grid_x_df['mean x displacement [px]'])
        self._create_and_save_heatmap_mean_xy(mean_x_map, mean_y_map, grid_x_df['mean x displacement [px]'],
                                             grid_y_df['mean y displacement [px]'])

    def _create_and_save_heatmap_mean_and_std(self, mean_x_map, x_stds_map, x_dis):
        """Create and save 2D heatmap visualization of mean x-displacements on the left, and standard deviation of x
        on the right."""
        # Set-up limits for the limits of the colorbars. Pick your favorite options.
        # 1. Select hard values based on the results - recommended only if they're rather consistent and the displacements
        # and standard deviation are very close (rarely the case).
        # cbar_min = -0.15
        # cbar_max = 0.15
        # 2. Select limits based on the displacements +/- standard deviation - I found these limits to be too generous
        # and my heatmaps hardly showed anything as the values oscillated too close to zero
        # cbar_min = min(mean_x_displacements + x_stds)
        # cbar_max = max(mean_x_displacements + x_stds)
        # 3. Default in this code - simply go off the displacement limits
        cbar_min = min(x_dis)
        cbar_max = max(x_dis)
        # Set-up the figure that displays the x-displacements and standard deviation of x next to each other
        # Basically fig,axs = something. hxx contains our axes. So hxx[0] is axes for left figure, hxx[1] for the right
        map_2D, hxx = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'hspace': 0.2},
                                    figsize=(12, 8), layout='constrained')
        map_2D.suptitle(f"Average and temporal noise floor in $x$ \n Attempt: {self.directory_name}", fontsize=30,
                        y=0.95)  # Overall title of the figure at the very top
        # Share the axes between the two plots so they're easily comparable
        hxx[1].get_shared_y_axes().join(hxx[0])
        hxx[1].get_shared_x_axes().join(hxx[0])
        # Plot mean x-displacements (x1) and standard deviation of x (y1) on the two subplots.
        # To change the colour: cmap=sns.cubehelix_palette(as_cmap=True); palettes can be found on:
        # https://seaborn.pydata.org/tutorial/color_palettes.html. If you add _r to the name of the palette, it will be
        # reversed.
        x1 = sns.heatmap(mean_x_map, center=0.0, cmap='coolwarm', square=True, ax=hxx[0], cbar_kws={'shrink': 0.5})
        y1 = sns.heatmap(x_stds_map, cmap='viridis', square=True, ax=hxx[1], cbar_kws={'shrink': 0.5})
        # Individual subplot titles
        # Formatting: r'$\overline{U_{x}} \ [px]$' is bar U_x so mean x-displacement in px. \bar would go over one
        # character, \overline goes over multiple characters
        hxx[0].set_title(r'$\overline{U_{x}} \ [px]$', fontsize=25)
        hxx[1].set_title(r'$SD(U_{x}) \ [px]$', fontsize=25)  # SD(U_x) is the standard deviation of the x-displacement, so temporal noise floor.
        # Flip the y-axis so it goes in the correct direction
        hxx[0].invert_yaxis()
        hxx[1].invert_yaxis()
        # Replace the ticks to make the plots look cleaner and easier to read
        new_x_ticks, new_x_labels, new_y_ticks, new_y_labels = self._replace_plot_ticks(hxx[0])
        # Overwrite default ticks
        x1.set_xticks(new_x_ticks, new_x_labels)
        x1.set_xlabel('$x$-position [mm]', fontsize=20)
        x1.set_yticks(new_y_ticks, new_y_labels)
        x1.set_ylabel('$y$-position [mm]', fontsize=20)
        y1.set_xticks(new_x_ticks, new_x_labels)
        y1.set_xlabel('$x$-position [mm]', fontsize=20)
        # y-axis is shared with the x-displacement figure and it looks clear, so make it empty for the 2nd plot
        y1.set_yticks([])
        y1.set_ylabel('')
        output_path = os.path.join(self.top_directory, f"{self.directory_name} x 2D_map.png")
        plt.show()
        map_2D.savefig(output_path)
        #plt.close() # Optional, I like to see the plots as I process the data to quickly identify abnormalities

    def _create_and_save_heatmap_mean_xy(self, mean_x_map, mean_y_map, x_dis, y_dis):
        """Create and save 2D heatmap visualization of mean x-displacements on the left and mean y-displacements on
        the right. They share the same colorbar.
        A lot of this code is the same as _create_and_save_heatmap_mean_and_std, so I'm not going to repeat the comments.
        The only major change is adding the third axis (hxx[2]) for the colorbar."""
        # Colorbar limits - based on both x and y displacement values
        cbar_min = min(x_dis + y_dis) # Only 2 small lists so + is should be fine performance-wise
        cbar_max = max(x_dis + y_dis)
        # Set-up the figure that displays the x-displacements, y-displacements, and colorbar
        # hxx[2] (so 3rd subplot) is for the colorbar so the widths of the figures are the same.
        # See: https://stackoverflow.com/questions/42712304/seaborn-heatmap-subplots-keep-axis-ratio-consistent
        maps_2D, hxx = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.1], 'hspace': 0.2},
                                    figsize=(12, 8),layout='constrained')
        maps_2D.suptitle(f"Mean displacements across all frames \n Attempt: {self.directory_name}", fontsize=30,
                        y=0.95)
        # Share the axes between the two plots
        hxx[1].get_shared_y_axes().join(hxx[0])
        hxx[1].get_shared_x_axes().join(hxx[0])
        # Plot mean x-displacements (x1) and standard deviation of x (y1) on the two subplots.
        # To change the colour: cmap=sns.cubehelix_palette(as_cmap=True); palettes can be found on:
        # https://seaborn.pydata.org/tutorial/color_palettes.html. If you add _r to the name of the palette, it will be
        # reversed.
        x1 = sns.heatmap(mean_x_map, vmin=cbar_min, vmax=cbar_max, center=0.0, cmap='coolwarm', square=True, ax=hxx[0],
                         cbar=False) # Turn off cbar as it will be displayed separately
        y1 = sns.heatmap(mean_y_map, vmin=cbar_min, vmax=cbar_max, center=0.0, cmap='coolwarm', square=True, ax=hxx[1],
                         cbar_ax=hxx[2], cbar_kws={'shrink': 0.5})  # cbar_ax - Puts the colorbar at the location for the 3rd subplot
        # Individual subplot titles
        hxx[0].set_title(r'$\overline{U_{x}} \ [px]$', fontsize=25)
        hxx[1].set_title(r'$\overline{U_{y}} \ [px]$', fontsize=25)
        # Flip the y-axis so it goes in the correct direction.
        hxx[0].invert_yaxis()
        hxx[1].invert_yaxis()
        # Replace the ticks to make the plots look cleaner and easier to read
        new_x_ticks, new_x_labels, new_y_ticks, new_y_labels = self._replace_plot_ticks(hxx[0])
        # Overwrite default ticks
        x1.set_xticks(new_x_ticks, new_x_labels)
        x1.set_xlabel('x position [mm]', fontsize=20)
        x1.set_yticks(new_y_ticks, new_y_labels)
        x1.set_ylabel('y position [mm]', fontsize=20)
        y1.set_xticks(new_x_ticks, new_x_labels)
        y1.set_xlabel('x position [mm]', fontsize=20)
        # y-axis is shared with the x-displacement figure and it looks clear, so make it empty for the 2nd plot
        y1.set_yticks([])
        y1.set_ylabel('')
        # Set the 'title' of the colorbar axis - the only way I managed to scale it down
        hxx[2].set_title('\n\n\n\n\n\n\n\n')
        output_path = os.path.join(self.top_directory, f"{self.directory_name} 2D_maps.png")
        plt.show()
        maps_2D.savefig(output_path)
        #plt.close() # Optional, I like to see the plots as I process the data to quickly identify abnormalities

    def _replace_plot_ticks(self, axis_entity):
        """Helper function to replace the ticks and labels on the heatmap subplots to reduce their number and make the
        plots look neater. We have min, max, and a few intermediates. This is especially important for small DIC subsets
        where the grid is dense."""
        # Get the tick locations and labels in data coordinates.
        x_ticks = axis_entity.get_xticks()
        x_labels = axis_entity.get_xticklabels()
        y_ticks = axis_entity.get_yticks()
        y_labels = axis_entity.get_yticklabels()
        n = self.HEATMAP_TICK_JUMP # Only defining this to make the code below more legible
        # Create new ticks with a reduced number of labels
        new_x_ticks = np.append(x_ticks[0::n], x_ticks[len(x_ticks) - 1])
        new_x_labels = np.append(x_labels[0::n], x_labels[len(x_labels) - 1])
        new_y_ticks = np.append(y_ticks[0::n], y_ticks[len(x_ticks) - 1])
        new_y_labels = np.append(y_labels[0::n], y_labels[len(y_labels) - 1])
        return new_x_ticks, new_x_labels, new_y_ticks, new_y_labels

    def plot_mean_and_std(self):
        """Processes the data for the line plots - converts to pixels (if desired) and sets the temporal values based on
        the data available - using either frames recorded or seconds.
        """
        if self.PLOT_LINE_PLOTS:
            for index, image in enumerate(self.csv_attempt_data):
                self.x_displacement_mean.append(image[self.COLUMN_NAMES[2]].mean())
                self.x_std.append(image[self.COLUMN_NAMES[2]].std())
            # Convert the values to pixels if this option is set to true
            if self.LINE_PLOTS_IN_PX:
                self.x_displacement_mean = [i * self.PIXEL_PER_MM for i in self.x_displacement_mean]
                self.x_std = [j * self.PIXEL_PER_MM for j in self.x_std]
            frames = np.linspace(1, len(self.x_displacement_mean) + 1, len(self.x_displacement_mean))
            # Use values for time for the y-axis if we can convert; otherwise, just use the frame number
            if self.time_per_image != 0:
                frames_to_time = frames * self.time_per_image
                temporal_values = frames_to_time
                temporal_label = 'Time [s]'
            else:
                temporal_values = frames
                temporal_label = 'Image'
            self._create_displacement_and_std_line_plot(temporal_values, temporal_label)

    def _create_displacement_and_std_line_plot(self, temporal_values, temporal_label):
        """Plots the mean x-displacement (top) and standard deviation (bottom) over time - either per
        frame or per time. It also adds shadowed area to the mean x-displacement to account for the std."""
        fig1, axs1 = plt.subplots(2, 1, sharex=True, figsize=(12, 8), layout='constrained')
        fig1.suptitle(f"x-displacement statistics \n Attempt: {self.directory_name}", fontsize=30)
        # Individual subplot titles
        axs1[0].set_title('Mean $x$-displacement', fontsize=25)
        axs1[1].set_title('Standard deviation', fontsize=25)
        # Plot
        axs1[0].plot(temporal_values, self.x_displacement_mean, color='#DDA3B2')
        axs1[1].plot(temporal_values, self.x_std, color='#7a6563')
        # Ticks and labels
        axs1[0].set_ylabel("Mean $x$-displacement [px]", fontsize=20)
        axs1[0].tick_params(axis='y', labelcolor='#000000')
        axs1[1].set_ylabel('Standard deviation [px]', fontsize=20)
        axs1[1].tick_params(axis='y', labelcolor='#000000')
        axs1[1].set_xlabel(temporal_label, fontsize=20)
        # Add a shadowed area to account for the standard deviation in the mean displacement values
        plusStd = [(x + y) for x, y in zip(self.x_displacement_mean, self.x_std)]
        minusStd = [(x - y) for x, y in zip(self.x_displacement_mean, self.x_std)]
        axs1[0].fill_between(temporal_values, plusStd, minusStd, facecolor='#7a6563', alpha=0.2)
        output_path = os.path.join(self.top_directory, f"{self.directory_name} x_displacement statistics.png")
        plt.show()
        fig1.savefig(output_path)
        #plt.close() # Optional, I like to see the plots as I process the data to quickly identify abnormalities

#######################################################################################################################

def main():
    analyser = NoiseFloorAnalyser()
    analyser.select_directory()
    analyser.process_csv_files()
    df_all_images = analyser.calculate_statistics()
    analyser.create_displacement_map(df_all_images)
    analyser.plot_mean_and_std()
    print("Data has been processed!")

if __name__ == "__main__":
    main()
