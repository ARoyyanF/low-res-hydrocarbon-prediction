import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker  # Untuk kustomisasi ticks axis
import seaborn as sns  # Untuk visualisasi statistik yang lebih baik
import altair as alt  # Visualisasi interaktifer
from scipy.signal import savgol_filter  # Add missing import

def combo_plot(data, tracks_config, depth_column, top_depth, bottom_depth, figure_height, 
                    subplot_adjust_top, major_ticks_interval, minor_ticks_interval, 
                    num_vertical_gridlines=5, formation=None):
    """
    Function to create a flexible well log visualization with customizable tracks and uniform gridlines.

    This function creates a well log visualization with a configurable number of tracks,
    each containing multiple log curves. It ensures that each curve has a fixed number
    of vertical gridlines, providing a consistent look across different scales.

    Parameters:
    ----------
    data : pandas.DataFrame
        DataFrame containing well log data with depth column and all log curve columns.
    tracks_config : list of dict
        List of track configurations. Each track dict should contain:
        - 'title': str - Title for the track.
        - 'show_grid': bool - Whether to show grid for this track (optional).
        - 'smoothing': str - "yes"/"no" for smoothing curves in this track (optional).
        - 'curves': list of dict - List of curve configurations for this track.
          Each curve dict should contain:
          - 'column': str - Column name in DataFrame.
          - 'label': str - Label for x-axis.
          - 'color': str - Color for the curve.
          - 'linestyle': str - Line style ('-', '--', ':').
          - 'min_val': float - Minimum scale value.
          - 'max_val': float - Maximum scale value.
          - 'log_scale': bool - Whether to use log scale (default False).
          - 'invert_axis': bool - Whether to invert x-axis (default False).
          - 'position_offset': int - Spine position offset (0, 40, 80, etc.).
          - 'show_grid': bool - Whether to show grid for this specific curve (optional, overrides track setting).
    depth_column : str
        Name of the depth column in DataFrame.
    top_depth : float
        Top depth for log display (upper limit).
    bottom_depth : float
        Bottom depth for log display (lower limit).
    figure_height : float
        Height of the figure in inches.
    subplot_adjust_top : float
        Adjustment parameter for the top of the subplots.
    major_ticks_interval : float
        Interval between major depth ticks in the y-axis.
    minor_ticks_interval : float
        Interval between minor depth ticks in the y-axis.
    num_vertical_gridlines : int, optional
        The fixed number of vertical gridlines to display for each curve. Default is 5.
    formation : dict, optional
        Dictionary containing formation data with formation names as keys and depth values.
        Example: {'LL-35-TOP': 3700, 'LL-35-BOTTOM': 4000}

    Returns:
    -------
    matplotlib.figure.Figure
        The figure object containing the well log visualization that can be displayed or saved.
    """
    # Filter data to the specified depth range
    logs = data[(data[depth_column] >= top_depth) & (data[depth_column] <= bottom_depth)].copy()
    
    num_tracks = len(tracks_config)
    
    # Create figure with specified number of tracks sharing the y-axis (depth)
    fig, axes = plt.subplots(nrows=1, ncols=num_tracks, figsize=(4 * num_tracks, figure_height), sharey=True)
    
    # Ensure axes is always a list for consistent indexing
    if num_tracks == 1:
        axes = [axes]
    
    fig.subplots_adjust(top=subplot_adjust_top, wspace=0.15)

    # General setting for all axes - apply consistent formatting to all tracks
    for ax in axes:
        ax.set_ylim(top_depth, bottom_depth)
        ax.invert_yaxis()  # Invert y-axis to show increasing depth downward
        ax.yaxis.grid(True, which='minor', linestyle=':', linewidth=0.5)  # Dotted minor grid
        ax.yaxis.grid(True, which='major', linestyle='-', linewidth='1')  # Solid major grid
        ax.yaxis.set_major_locator(ticker.MultipleLocator(major_ticks_interval))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_ticks_interval))
        ax.get_xaxis().set_visible(False)  # Hide the original x-axis

    # Set depth label only on the first track
    axes[0].set_ylabel("Depth", fontsize=12)

    # Apply Savitzky-Golay filter to smooth noisy log data if requested
    for track_idx, track_info in enumerate(tracks_config):
        # Check for track-level smoothing first, then fall back to smoothing_config
        should_smooth = track_info.get('smoothing', 'no') == "yes"
        
        if should_smooth:
            for curve_info in track_info['curves']:
                col = curve_info['column']
                if col in logs.columns:
                    valid_data = logs[col].dropna()
                    if len(valid_data) > 5:
                        smoothed = savgol_filter(valid_data.values, window_length=5, polyorder=3, mode='nearest')
                        logs.loc[valid_data.index, col] = smoothed

    # Plot each track
    for track_idx, track_info in enumerate(tracks_config):
        ax = axes[track_idx]
        curves = track_info.get('curves', [])
        
        # Set track title
        max_offset = max([curve.get('position_offset', 0) for curve in curves]) if curves else 0
        ax.set_title(track_info.get('title', f'Track {track_idx + 1}'), 
                    pad=max_offset + 60, fontsize=14, fontweight='bold')
        
        # Get track-level grid setting
        track_show_grid = track_info.get('show_grid', True)
        
        # Plot curves in this track
        for curve_info in curves:
            col = curve_info['column']
            
            # Skip if column doesn't exist in data
            if col not in logs.columns:
                print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
                continue
            
            # Create twin axis for this curve
            twin_ax = ax.twiny()
            
            # Set scale limits
            if 'min_val' in curve_info and 'max_val' in curve_info:
                twin_ax.set_xlim(curve_info['min_val'], curve_info['max_val'])
            
            # Check if log scale is requested
            is_log_scale = curve_info.get('log_scale', False)
            if is_log_scale:
                twin_ax.set_xscale('log')
            
            # Invert axis if specified
            if curve_info.get('invert_axis', False):
                twin_ax.invert_xaxis()
            
            # Position the spine
            position_offset = curve_info.get('position_offset', 0)
            twin_ax.spines['top'].set_position(('outward', position_offset))
            
            # Set labels and colors
            twin_ax.set_xlabel(curve_info.get('label', col), color=curve_info.get('color', 'black'))
            twin_ax.tick_params(axis='x', colors=curve_info.get('color', 'black'))
            
            # Plot the curve
            twin_ax.plot(logs[col], logs[depth_column], 
                        linestyle=curve_info.get('linestyle', '-'),
                        color=curve_info.get('color', 'black'),
                        linewidth=1.0)
            
            # Add grid - check curve-level first, then track-level
            show_grid = curve_info.get('show_grid', track_show_grid)
            if show_grid:
                # Set a fixed number of gridlines for the x-axis for universal appearance.
                if is_log_scale:
                    # For log scales, LogLocator is more appropriate.
                    locator = ticker.LogLocator(numticks=num_vertical_gridlines)
                else:
                    # For linear scales, LinearLocator works well.
                    locator = ticker.LinearLocator(numticks=num_vertical_gridlines)
                
                twin_ax.xaxis.set_major_locator(locator)
                twin_ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))
                
                twin_ax.grid(True, which='major', axis='x', linestyle='--', linewidth='0.5', color='gray')
                twin_ax.set_axisbelow(True)
    
    # Add formation lines and labels if formation data is provided
    if formation is not None:
        # Apply formation lines to all tracks
        for ax in axes:
            for formation_name, depth in formation.items():
                if (depth >= top_depth) and (depth <= bottom_depth):
                    ax.axhline(y=depth, linewidth=0.5, color="black")
                    ax.text(
                        0.1, depth, formation_name, horizontalalignment="center", verticalalignment="center",
                        transform=ax.get_yaxis_transform(), fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
                    )

    return fig

def add_productive_zones(data, productive_zones):
    """
    Add formation classification (0 or 1) based on formation depths.
    1: Within productive formation depths
    0: Outside productive formation depths
    
    Args:
        data (pd.DataFrame): Input dataframe containing DEPT column
        
    Returns:
        pd.DataFrame: DataFrame with added formation_class column
    """
    # Create a copy to avoid modifying original data
    result = data.copy()
    
    # Initialize classification column with zeros
    result['hydrocarbon_formation_class'] = 0
    
    # Define productive zones from formation depths
    
    
    # Mark productive zones as 1
    for top, base in productive_zones:
        mask = (result.DEPT >= top) & (result.DEPT <= base)
        result.loc[mask, 'hydrocarbon_formation_class'] = 1
    
    return result