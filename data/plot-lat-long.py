import re
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def create_dummy(file_path):
    # Create a dummy file for demonstration
    dummy_content = """#-------------------------------------- OrbGen run: Fri Apr 11 14:11:53 2025 ------------------------------------------------- #
    # yy-mm-ddThh:mm:ss.sss dt(min) IFA F10.7 FB10.7 Ap IDW Rho(kg/m3) Vn(km/s) Ve(km/s) MMWT Tloc(K) Texo(K) LST(h) GLat(d) GLon(d) GAlt(km) Va(km/s) gam(d) gload(-) qdot(W/m2) SRat(-) TRat(-) KnInf(-) MaInf(-) CD(-) CD/CD0 Orb(-) ULat(d) dS(km) Hpe(km) Hap(km) H(km) Torb(min) mjd1950.0(d) #
    # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 #
    #------------------------------------------------------------------------------------------------------------------------------
    2025-02-19T02:13:45.786 0.00000D+00 5. 136.00 132.22 13.56 7. 4.5608D-09 0.0000D+00 0.0000D+00 25.418 719.06 1077.62 1.955 50.215 -0.570 143.838 7.549260 0.144 1.002 7.5197D+03 11.007 35.616 2.13D+01 1.32D+01 2.3915 1.0000 0.204 73.586 0.000000D+00 122.292 159.026 131.278 87.409 27443.092891050
    2025-02-19T02:14:45.786 1.00000D+00 5. 136.00 132.22 13.56 7. 4.2098D-09 0.0000D+00 0.0000D+00 25.333 733.08 1075.06 2.368 51.528 5.369 145.500 7.547003 0.155 1.002 7.2180D+03 11.007 35.616 2.13D+01 1.32D+01 2.3915 1.0000 0.216 77.725 4.420675D+02 121.382 158.457 132.459 87.403 27443.093585494
    2025-02-19T02:15:45.786 2.00000D+00 5. 136.00 132.22 13.57 7. 3.8791D-09 0.0000D+00 0.0000D+00 25.217 744.18 1070.41 2.801 52.497 11.622 147.115 7.544825 0.165 1.002 6.9228D+03 11.007 35.616 2.13D+01 1.32D+01 2.3915 1.0000 0.227 81.862 8.838052D+02 120.417 158.228 133.722 87.398 27443.094279939
    """
    with open(file_path, 'w') as f:
        f.write(dummy_content)
    

def read_orbgen_file(file_path):
    """
    Reads an OrbGen output file, extracts the header and data,
    and returns them as a tuple of lists and a Pandas DataFrame.

    Args:
        file_path (str): The path to the OrbGen output file.

    Returns:
        tuple: A tuple containing:
            - header_comments (list): A list of comment lines from the header.
            - column_names (list): A list of column names extracted from the file.
            - data_df (pandas.DataFrame): A DataFrame containing the data.
              Returns None if no data is found.
    """
    header_comments = []
    column_names = []
    data_lines = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                header_comments.append(line)
                if not column_names and not line.startswith('#--'):
                    print(line)
                    # Extract column numbers and names
                    for c in line.split():
                        if c != "#":
                            column_names.append(c)
            elif line and not line.startswith('#'):
                data_lines.append(line)

    if data_lines:
        data = StringIO('\n'.join(data_lines))
        data_df = pd.read_csv(data, sep='\s+', names=column_names, comment='#')
        return header_comments, column_names, data_df
    else:
        return header_comments, column_names, None


import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_altitude_on_map(data_df, lat_col='GLat(d)', lon_col='GLon(d)', alt_col='GAlt(km)',
                         title='Altitude on Map', cmap='viridis', marker_size=100,
                         figsize=(10, 8), extent=None):
    """
    Plots altitude data on a geographical map using Cartopy.

    Args:
        data_df (pd.DataFrame): DataFrame containing latitude, longitude, and altitude data.
        lat_col (str, optional): Name of the latitude column. Defaults to 'GLat(d)'.
        lon_col (str, optional): Name of the longitude column. Defaults to 'GLon(d)'.
        alt_col (str, optional): Name of the altitude column. Defaults to 'GAlt(km)'.
        title (str, optional): Title of the plot. Defaults to 'Altitude on Map'.
        cmap (str, optional): Colormap to use for altitude. Defaults to 'viridis'.
        marker_size (int, optional): Size of the markers on the map. Defaults to 10.
        figsize (tuple, optional): Size of the figure (width, height). Defaults to (10, 8).
        extent (list or tuple, optional): Bounding box for the map (lon_min, lon_max, lat_min, lat_max).
                                          If None, the map will try to fit the data.
    """
    if not all(col in data_df.columns for col in [lat_col, lon_col, alt_col]):
        print(f"Error: DataFrame must contain columns '{lat_col}', '{lon_col}', and '{alt_col}'.")
        return

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, facecolor='lightcyan')
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    # Scatter plot of altitude data
    ax.plot(data_df[lon_col], data_df[lat_col], transform=ccrs.PlateCarree())
    scatter = ax.scatter(data_df[lon_col], data_df[lat_col], c=data_df[alt_col],
                       cmap=cmap, s=marker_size, transform=ccrs.PlateCarree())

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Altitude (km)', shrink=0.6)

    # Set title
    ax.set_title(title)

    # Set extent if provided
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        # Try to set extent based on data
        try:
            lon_min = data_df[lon_col].min() - 5
            lon_max = data_df[lon_col].max() + 5
            lat_min = data_df[lat_col].min() - 5
            lat_max = data_df[lat_col].max() + 5
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        except ValueError:
            print("Warning: Could not automatically determine map extent.")

    plt.tight_layout()
    plt.savefig(file_path.replace(".dat","-map.png"), dpi=150)
    plt.show()

if __name__ == "__main__":
    # Example usage:
    file_path = 'orbgen#12-cut.dat'  # Replace with the actual path to your file
    
    
    header, columns, df = read_orbgen_file(file_path)
    
    
    if df is not None:
        # Assuming column 1 is the first column (index 0) and holds the date and time
        if columns and len(columns) >= 17:
            time_col = columns[0]
    
            # Convert the time column to datetime objects
            try:
                df['Time'] = pd.to_datetime(df[time_col], format='%Y-%m-%dT%H:%M:%S.%f')
            except ValueError:
                try:
                    df['Time'] = pd.to_datetime(df[time_col], format='%Y-%m-%dT%H:%M:%S')
                except ValueError as e:
                    print(f"Error converting time column: {e}")
                    df = None # Set df to None to prevent plotting
            if df is not None and 'Time' in df.columns:
                # Create the figure and subplots
                fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
                # Plot Latitude vs Time
                axs[0].plot(df['Time'], df["GLat(d)"])
                axs[0].set_ylabel('Latitude [°]')
                axs[0].set_title('Latitude, Longitude, and Altitude vs. Time')
                axs[0].grid(True)
    
                # Plot Longitude vs Time
                axs[1].plot(df['Time'], df["GLon(d)"])
                axs[1].set_ylabel('Longitude [°]')
                axs[1].grid(True)
    
                # Plot Altitude vs Time
                axs[2].plot(df['Time'], df["GAlt(km)"])
                axs[2].set_ylabel('Altitude [km]')
                axs[2].set_xlabel('Time')
                axs[2].grid(True)
    
                # Plot Speed vs Time
                axs[3].plot(df['Time'], df["Va(km/s)"])
                axs[3].set_ylabel('Va [km/s]')
                axs[3].grid(True)
    
                # Format the x-axis to display dates nicely
                xfmt = mdates.DateFormatter('%H:%M:%S')
                axs[-1].xaxis.set_major_formatter(xfmt)
                axs[-1].set_xlabel('Time [UT] on 2025-03-19')
                fig.autofmt_xdate()
    
                plt.tight_layout()
                plt.savefig(file_path.replace(".dat",".png"), dpi=150)
                plt.show()
            else:
                print("Could not process time data for plotting.")
            plot_altitude_on_map(df)        
        else:
            print("Not enough columns in the data to plot Latitude, Longitude, and Altitude.")
    else:
        print("No data found in the file.")
