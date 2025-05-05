# 🌍 Land Surface Temperature Estimation from Satellite Data (AMSR-E & MODIS)

## Overview

This repository documents the work conducted during my internship on estimating **Land Surface Temperature (LST)** using satellite data. The project is part of the R&D efforts related to the future **Copernicus Imaging Microwave Radiometer (CIMR)** mission. Since CIMR is not yet in orbit, the study uses data from heritage sensors with similar characteristics, namely **AMSR-E**, **WindSat**, and **MODIS**. The ultimate goal is to develop and validate an algorithm capable of retrieving LST from CIMR-like data.

## 🚀 Objective

To retrieve **Land Surface Temperature (LST)** using:

- **Microwave brightness temperature** data from the **AMSR-E** satellite (specifically the 37 GHz channel)
- **Thermal infrared LST** from **MODIS** as a reference (which operates in the Near-Infrared range)
- **In-situ station data** from **FLUXNET** to train and validate the regression model

This project contributes to the development of L2 algorithms for the CIMR mission by exploring regression-based retrieval techniques and data integration methods.

## 🔍 Methodology

1. **Data Acquisition**
   - Downloading AMSR-E brightness temperature (TB) data for the year **2005**
   - Downloading MODIS LST products for the same period
   - Collecting ground truth data from **FLUXNET** in-situ stations

2. **Preprocessing**
   - Geolocation matching of AMSR-E and MODIS data
   - Filtering of cloud-contaminated MODIS pixels
   - Temporal alignment of observations

3. **Algorithm Development**
   - Implementing a regression model to estimate LST from AMSR-E TBs
   - Using MODIS as ground truth for training
   - Validating with FLUXNET where possible

4. **Evaluation**
   - Compare regression-based LST with MODIS reference
   - Analyze seasonal and spatial biases
   - Test cross-sensor compatibility with WindSat (optional)

## 🛠 Technologies

- **Language**: Python
- **Libraries**: `numpy`, `pandas`, `xarray`, `matplotlib`, `scikit-learn`, `netCDF4`, `pyhdf`, `rasterio`
- **Satellite Data**:
  - AMSR-E (37 GHz TB)
  - MODIS (Terra/Aqua LST)
  - WindSat (optional)
- **In-situ Data**:
  - FLUXNET LST and meteorological observations

## 📁 Repository Structure

```
.
├── data/
│   ├── processed/         
│   │   ├── modis/                        # .hdf MODIS files converted to .csv
│   │   ├── fluxnet/    
│   │   │   └── stations/                  # List of fluxnet stations with their coordinates
│   │   └── amsre/
│   │   │   ├── dates (YYYY - MM - DD)/   # Contains brightness temperatures when the orbit is ascending and descending
│   │   │   └── matched /                  # Match brightness temperature and temperature for a specific station
│   ├── raw/               
│   │   ├── modis/                        # .hdf MODIS files
│   │   ├── fluxnet/                      # Table of temperatures recorded at various stations
│   │   └── amsre/                        # AMSR_E files (.hdf)
├── docs/                                 # Useful references  for understanding the project
├── notebooks/
│   └── test.ipynb                        # Perform tests on pieces of code
├── outputs/
│   ├── amsre/     
│   │   ├── dates/                        # Contains maps with gloss temperatures and plotting of temperatures as a function of TB with linear regression (.png)
│   ├── fluxnet/  
│   │   ├── seasonal_evolution/           # Time trend in temperature for each station
│   │   ├── seaonal_temp_tb/              # Comparison of temperature and TB over time
│   │   └── stationswise_regression/      # Linear regression of temperature by station and overall
│   └── modis/
├──src/
│   ├── modis/
│   │   ├── download.py                   # Download data from modis
│   │   ├── process.py                    # Data processing
│   │   └── utils.py      
│   ├── amsre/                                              
│   │   ├── download.py                                        # Downloading data with earthaccess
│   │   ├── generate_daily_matches.py                          # Match temperature and TB datas
│   │   ├── match_tb_fluxnet.py                                # Return the file of the match (.csv)
│   │   ├── plot_brightness_vs_temperature_and_regression.py    
│   │   ├── plot_regressions.py                                 
│   │   ├── plot_temp_evolution.py                             # Plot the temporal comparison of temperature and TB (.png)
│   │   ├── plot.py                                            # Plot the TB map (.png)
│   │   └── process.py                                         # data processing and backup (.csv)
├── main.py                               # Supporting documentation and references  
├── requirements.txt
└── README.md                             # Project description
```

## 🧑‍💻 Code Description
### Main Script: ```main.py```
The main script, main.py, integrates various steps of the project, including data processing, regression algorithm application, and graph generation. Here are the key functionalities:

1. Data Processing:

- Reading preprocessed AMSR-E, MODIS, and FLUXNET data.

- Performing matching algorithms to associate surface temperature data with brightness temperature data.

2. Graph Generation:

- Visualizing seasonal temperature evolution using in-situ data and AMSR-E data.

- Comparing AMSR-E brightness temperature with MODIS and FLUXNET LST.

- Generating scatter plots and seasonal evolution graphs for different stations.

3. Saving Results:

- Generated graphs are saved in dedicated subfolders under ```outputs/fluxnet/seasonal_temp_tb```.

### Modules and Functions
```src/amsre/plot_temp_evolution.py```
The ```plot_temp_evolution.py module``` is responsible for generating the graphs. Here are the main functions:

1. ```plot_seasonal_temp_evolution()```
This function generates seasonal temperature evolution graphs, comparing in-situ temperatures with AMSR-E temperatures.

- It takes as input a CSV file containing surface temperatures and generates one graph per station, with temperature plotted against the day of the year (DOY).

- If AMSR-E data is available, it overlays the AMSR-E brightness temperatures on top of the in-situ temperatures.

2. ```plot_seasonal_temp_with_tb_evolution()```
This function generates a comparative graph between in-situ LST and AMSR-E brightness temperature (37 GHz) over the entire period.

- It uses matched data files in the data/processed/amsre/matched folder to compute daily averaged temperatures for each station.

- It plots a seasonal evolution graph of in-situ LST temperatures against AMSR-E brightness temperatures, allowing you to visualize regression performance and fitting with MODIS data.

#### Example Usage
- Running the main script
To run the analysis and generate the graphs, simply execute main.py in your Python environment after placing the necessary data files in the correct folder.

```bash 
python main.py
```

- Generate Seasonal Graphs
To generate seasonal graphs for FLUXNET stations and AMSR-E data, use the function **```plot_seasonal_temp_evolution()```** by providing the path to a CSV file as input.

## ⚙️ Dependencies
The required dependencies are listed in the requirements.txt file. To install the dependencies, run the following command in your terminal:

```bash 
pip install -r requirements.txt
```

## 📝 Execution Notes

### Error Handling
When running the main script or generating graphs, the following issues may arise:

- **Incorrect Date Format in CSV Files**:
If a data file does not follow the expected date format (YYYYMMDD), the program will ignore it and display an error message indicating the problematic file.

- **Missing Required Columns**:
If a matched data file does not contain the necessary columns (brightness_temp_37v, temperature), it will be ignored, and an error message will be displayed.

- **Outlier Data**:
Filters are applied to remove temperature outliers (e.g., temperatures below 180 K or above 330 K).

### Generated Graphs
Seasonal Graphs for Each Station:
Each station in the FLUXNET data will have a graph showing the evolution of temperature over the course of the year. If AMSR-E data is available, it will also be plotted for comparison.

- **Comparative Seasonal Graph of LST and AMSR-E TB**:
This graph compares the seasonal evolution of surface temperature estimated through regression (in-situ LST) with AMSR-E brightness temperature (37 GHz).

## 📊 Results
Results are saved in the ```outputs/``` directory as PNG graphs. The following subfolders are used for saving the results:

- ```outputs/fluxnet/seasonal_temp_tb/``` : Graphs comparing LST temperatures with AMSR-E brightness temperatures for all stations.

- ```outputs/fluxnet/seasonal_evolution/``` : Seasonal graphs for each individual station.

## 🔧 Debugging & Contributions

- If you encounter issues while running the scripts, here are some troubleshooting steps:

- Ensure that the data files are correctly formatted and placed in the appropriate folders (```data/processed/amsre/matched``` and ```data/processed/modis```).

- Make sure all dependencies are installed using pip install -r requirements.txt.

- For any questions or suggestions for modification, feel free to open an issue on GitHub or contribute via pull request.

## 📚 Additional References

- [CIMRL2PAD-UVEG-TEC-RAS-D2 Technical Note](./docs/CIMRL2PAD-UVEG-TEC-RAS-D2.pdf) — describes R&D activities for CIMR L2 product algorithm development
- [CIMR-MRD-v5.0-20230211 Requirements document](./docs/CIMR-MRD-v5.0-20230211_(Issued).pdf) - gives mission requirements
- [Holmes_et_al_2009 Journal of Geophysical Research](./docs/Holmes_et_al_2009.pdf) - Explains the physical aspect of the project
- [Jimenez_et_al_2017 Journal of Geophysical Research about Atmospheres](./docs/Jimenez_et_al_2017.pdf) - Explains the physical aspect of the project concerning the atmosphere


## 👤 Author

**Kevin DUGARD**  
Intern at ENSEA, Cergy (France), Internship at UPC, Barcelona (Spain)
Supervised by **Mercè Vall-Llossera Ferran**, UPC associate professor
