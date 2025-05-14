import os
import re
from typing import Optional
import numpy as np
import pandas as pd
from scipy.ndimage import label


def load_data_into_dataframe(data_path):
    """
    Loads all CSV files in a given directory into a single Pandas DataFrame and performs some basic preprocessing steps.

    Parameters
    ----------
    data_path : str
        The path to the directory containing the CSV files.

    Returns
    -------
    combined_df : pandas.DataFrame
        A DataFrame containing all the data from the CSV files in the given directory.

    Notes
    -----
    The returned DataFrame has the 'filename' column updated to remove the '.zip' extension, and the 'frame' column incremented by 1.
    """
    # Initialize an empty DataFrame to hold the combined data
    combined_df = pd.DataFrame()
    # Expand the path to the directory
    downloads_path = os.path.expanduser(data_path)
    # Iterate over each file in the directory
    for file_name in os.listdir(downloads_path):
        if file_name.endswith('.csv'):
            # Construct the full path to the file
            full_path = os.path.join(downloads_path, file_name)
            # Load the CSV file into a DataFrame
            df = pd.read_csv(full_path)
            # Concatenate the loaded DataFrame with the combined DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Remove '.zip' from all rows in the 'filename' column if the column exists
    if 'filename' in combined_df.columns:
        combined_df['filename'] = combined_df['filename'].str.replace('.zip', '', regex=False)
    
    combined_df['frame'] = combined_df['frame'] + 1

    return combined_df

def analyze_cells(frame, conversion_factor=746/599):
    """
    Analyzes cells in a given frame to compute cell-specific metrics such as area, perimeter, and boundary location.

    Parameters
    ----------
    frame : np.ndarray
        A 2D numpy array representing a frame of cell data, where each unique non-zero integer represents a distinct cell.
    conversion_factor : float, optional
        A factor to convert pixel measurements to microns. Default is 746/599.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the following columns:
        - 'cell_id': Unique identifier for each cell.
        - 'cell_area': Area of the cell in microns squared.
        - 'perimeter_cell': Boolean indicating if the cell is located on the perimeter of the frame.
        - 'cell_perimeter': Perimeter of the cell in microns.
    """

    unique_cells = np.unique(frame)
    unique_cells = unique_cells[unique_cells != 0]
    
    area_conversion_factor = conversion_factor ** 2  # Converting area to microns squared
    
    cell_data = {'cell_id': [], 'cell_area': [], 'perimeter_cell': [], 'cell_perimeter': []}
    for cell_id in unique_cells:
        cell_area_pixels = np.sum(frame == cell_id)
        cell_area = cell_area_pixels * area_conversion_factor  # Convert area from pixels^2 to microns^2
        
        on_perimeter = np.any(frame[0, :] == cell_id) or np.any(frame[:, 0] == cell_id) or \
                       np.any(frame[-1, :] == cell_id) or np.any(frame[:, -1] == cell_id)
        
        cell_perimeter_pixels = calculate_perimeter(frame, cell_id)
        cell_perimeter = cell_perimeter_pixels * conversion_factor  # Convert perimeter from pixels to microns
        
        cell_data['cell_id'].append(cell_id)
        cell_data['cell_area'].append(cell_area)
        cell_data['perimeter_cell'].append(on_perimeter)
        cell_data['cell_perimeter'].append(cell_perimeter)
    
    cell_data_df = pd.DataFrame(cell_data)
    return cell_data_df

def calculate_perimeter(frame, cell_id):
    """
    Calculates the perimeter of a specified cell within a frame.

    Parameters
    ----------
    frame : np.ndarray
        A 2D numpy array where each unique non-zero integer represents a distinct cell.
    cell_id : int
        The unique identifier for the cell whose perimeter is to be calculated.

    Returns
    -------
    int
        The perimeter of the cell in pixels, computed as the number of transitions from cell to non-cell pixels
        in the binary mask of the cell.

    Notes
    -----
    The function creates a binary mask of the specified cell, pads it to account for edge cells,
    and calculates the perimeter by counting transitions from 1 to 0 in the mask.
    """
    # Create a binary mask where the current cell_id is 1, and others are 0
    cell_mask = frame == cell_id

    # Pad the mask with zeros on all sides to handle edge cells correctly
    padded_mask = np.pad(cell_mask, pad_width=1, mode='constant', constant_values=0)

    # Count transitions from 1 to 0 (cell to non-cell) at each pixel
    perimeter = (
        np.sum(padded_mask[:-2, 1:-1] & ~padded_mask[1:-1, 1:-1]) +  # up
        np.sum(padded_mask[2:, 1:-1] & ~padded_mask[1:-1, 1:-1]) +   # down
        np.sum(padded_mask[1:-1, :-2] & ~padded_mask[1:-1, 1:-1]) +  # left
        np.sum(padded_mask[1:-1, 2:] & ~padded_mask[1:-1, 1:-1])     # right
    )

    return perimeter

def process_all_frames(
        input_array, 
        filename,
        use_connected_component_labeling: Optional[bool] = False
):
    """
    Processes a series of frames from an input array, extracting cell data and optionally labeling connected components.

    Parameters
    ----------
    input_array : np.ndarray
        A 3D numpy array representing a stack of 2D frames to be processed.
    filename : str
        The filename containing the frame data, used to extract start frame number and assign to each processed frame.
    use_connected_component_labeling : bool, optional
        Whether to use connected component labeling on each frame. Default is False.

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame containing cell data from all processed frames, with columns including 'cell_id', 'cell_area',
        'perimeter_cell', 'cell_perimeter', 'frame', and 'filename'.

    Notes
    -----
    The function uses a regular expression to extract the start frame number from the filename and increments this number
    for each processed frame. If connected component labeling is enabled, the number of connected components in each frame
    is printed.
    """

    all_frames_data = []
    
    # Extract the start frame number from the filename using a regular expression
    match = re.search(r'start_(\d+)_end_(\d+)', filename)
    if match:
        start_frame = int(match.group(1))

    # Iterate through each frame in all_tcell_mask
    for frame_idx in range(input_array.shape[0]):
        frame = input_array[frame_idx, :, :]

        if use_connected_component_labeling:
            # Label the connected components in the frame
            frame, num_features = label(frame)
            print(f"Frame {frame_idx + 1} has {num_features} connected components")

        cell_data_df = analyze_cells(frame)  # Ensure analyze_cells accepts filename
        # Calculate the correct frame number based on the start frame
        cell_data_df['frame'] = start_frame + frame_idx + 1
        cell_data_df['filename'] = filename.replace('.zip', '')
        all_frames_data.append(cell_data_df)

    # Concatenate all DataFrames in the list into one large pd DataFrame
    all_cell_data_df = pd.concat(all_frames_data, ignore_index=True)

    return all_cell_data_df

def identify_consecutive_frames(interaction_id_grouped):
    """
    Processes a DataFrame to identify and label consecutive frames for each interaction ID group.

    Parameters
    ----------
    interaction_id_grouped : pd.DataFrame
        A DataFrame grouped by 'interaction_id', containing columns 'frame' and 'filename'.

    Returns
    -------
    pd.DataFrame
        An updated DataFrame with additional columns:
        - 'frame_diff': Difference between the current and previous frame numbers.
        - 'is_consecutive': Boolean indicating if frames are consecutive.
        - 'consec_group': Unique identifier for each sequence of consecutive frames.
        - 'file_segment': Extracted part of the filename used in the unique identifier.
        - 'unique_consec_group': Unique identifier for each group combining file segment, interaction ID, and consecutive group.
        - 'interaction_id_consec_frame': Frame enumeration within each consecutive group.
    """
    # Calculate the difference between current and previous frames
    interaction_id_grouped['frame_diff'] = interaction_id_grouped['frame'] - interaction_id_grouped['frame'].shift(1, fill_value=interaction_id_grouped['frame'].iloc[0])
    
    # Identify where frames are consecutive
    interaction_id_grouped['is_consecutive'] = interaction_id_grouped['frame_diff'] == 1
    
    # Create unique identifiers for consecutive frame sequences
    interaction_id_grouped['consec_group'] = (~interaction_id_grouped['is_consecutive']).cumsum()

    # Extract the part of the filename between the first and second underscore
    interaction_id_grouped['file_segment'] = interaction_id_grouped['filename'].apply(lambda x: x.split('_')[1])
    # Form a unique identifier for each group combining 'file_segment', 'interaction_id', and 'consec_group'
    interaction_id_grouped['unique_consec_group'] = interaction_id_grouped['file_segment'] + '_' + interaction_id_grouped['interaction_id'].astype(str) + '_group' + interaction_id_grouped['consec_group'].astype(str)
    
    # Now, enumerate the frames within each consecutive group
    interaction_id_grouped['interaction_id_consec_frame'] = interaction_id_grouped.groupby('consec_group').cumcount() + 1
    
    return interaction_id_grouped

def calculate_consecutive_frames(df_interactions_file):
    """
    Identifies and labels consecutive frames for each interaction ID group in a DataFrame.

    Parameters
    ----------
    df_interactions_file : pd.DataFrame
        A DataFrame containing columns 'interaction_id', 'frame', and 'filename'.

    Returns
    -------
    pd.DataFrame
        An updated DataFrame with additional columns:
        - 'unique_consec_group': Unique identifier for each group combining file segment, interaction ID, and consecutive group.
        - 'interaction_id_consec_frame': Frame enumeration within each consecutive group.
    """
    # Sort the DataFrame by 'interaction_id' and 'frame'
    df_sorted = df_interactions_file.sort_values(by=['interaction_id', 'frame'])

    # Group by 'interaction_id'
    interaction_id_grouped = df_sorted.groupby('interaction_id')

    df_with_consec = interaction_id_grouped.apply(identify_consecutive_frames)
    df_with_consec.reset_index(drop=True, inplace=True)
    
    # Drop intermediate columns used for computation
    df_with_consec.drop(columns=['frame_diff', 'is_consecutive', 'consec_group', 'file_segment'], inplace=True)
    
    return df_with_consec

def find_cell_interactions_with_counts(
        t_cells,
        cancer_cells,
        filepath
):
    """
    Finds cell interactions and counts the number of pixels in contact in each frame.

    Parameters
    ----------
    t_cells : np.ndarray
        A 3D numpy array representing the T cell masks. Dimensions are (frames, height, width).
    cancer_cells : np.ndarray
        A 3D numpy array representing the cancer cell masks. Dimensions are (frames, height, width).
    filepath : str
        The path to the file containing the frame data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing columns 't_cell_id', 'cancer_cell_id', 'contact_pixels', 'filename', and 'frame'.
    """
    # Initialize an empty list to store DataFrames from each frame
    all_interactions_data = []
    
    # Extract the start frame number from the filename, assuming a similar naming convention
    start_frame = 1  # Default start frame
    match = re.search(r'start_(\d+)_end_(\d+)', filepath)
    if match:
        start_frame = int(match.group(1))

    for frame_idx in range(t_cells.shape[0]):
        t_cell_frame = t_cells[frame_idx, :, :]
        cancer_cell_frame = cancer_cells[frame_idx, :, :]
        unique_t_cells = np.unique(t_cell_frame[t_cell_frame > 0])
        
        interaction_pairs = []
        for t_cell_id in unique_t_cells:
            t_cell_coords = np.argwhere(t_cell_frame == t_cell_id)
            
            for coord in t_cell_coords:
                x, y = coord
                neighbors = [(i, j) for i in range(max(0, x-1), min(t_cell_frame.shape[0], x+2))
                            for j in range(max(0, y-1), min(t_cell_frame.shape[1], y+2))
                            if (i, j) != (x, y)]
                
                for nx, ny in neighbors:
                    if cancer_cell_frame[nx, ny] > 0:
                        interaction_pairs.append((t_cell_id, cancer_cell_frame[nx, ny]))
        
        # Convert interactions to DataFrame and count duplicates
        df_interactions = pd.DataFrame(interaction_pairs, columns=['t_cell_id', 'cancer_cell_id'])
        df_interactions['contact_pixels'] = 1
        df_interactions = df_interactions.groupby(['t_cell_id', 'cancer_cell_id']).count().reset_index()
        df_interactions['filename'] = os.path.basename(filepath).replace('.zip', '')
        
        # Add the frame number to the DataFrame
        df_interactions['frame'] = start_frame + frame_idx + 1
        
        all_interactions_data.append(df_interactions)
    
    # Concatenate all frame DataFrames into one
    all_interactions_df = pd.concat(all_interactions_data, ignore_index=True)
    
    return all_interactions_df

def get_group_from_filename(
        filename, 
        group_logic
):
    """
    Determines the group associated with a given filename based on predefined prefixes.

    Parameters
    ----------
    filename : str
        The name of the file to be checked against group prefixes.
    group_logic : dict
        A dictionary where keys are group names and values are lists of prefixes
        associated with each group.

    Returns
    -------
    str or None
        The group name if a matching prefix is found in the filename; otherwise, None.
    """

    for group, prefixes in group_logic.items():
        if any(prefix in filename for prefix in prefixes):
            return group
    return None

def filter_and_label(
        group,
        num_frames=10
):
    """
    Filters and labels a DataFrame of frames based on a reference frame.

    Parameters
    ----------
    group : pd.DataFrame
        A DataFrame containing columns 'frame_x' and 'frame_y', representing frame references.
    num_frames : int, optional
        The number of frames to include before the reference frame. Default is 10.

    Returns
    -------
    pd.DataFrame
        A filtered and labeled DataFrame containing only frames from the last 10 before the 
        reference frame, including the reference frame itself, with an additional column 
        'interaction_id_consec_frame' indicating the order of frames.
    """
    # Get the reference frame from frame_x
    ref_frame = group['frame_x'].iloc[0]
    
    # Filter to include only frames within last num_frames before frame_x plus the reference frame (num_frames + 1)
    condition = (group['frame_y'] <= ref_frame) & (group['frame_y'] > ref_frame - (num_frames + 1))
    filtered_group = group[condition].copy()
    
    # Sort and label the last 10 frames plus the reference frame
    if not filtered_group.empty:
        filtered_group = filtered_group.sort_values(by='frame_y', ascending=False)
        # Ensure the length of labels matches the number of rows
        filtered_group['interaction_id_consec_frame'] = list(range(0, -len(filtered_group), -1))
    
    return filtered_group