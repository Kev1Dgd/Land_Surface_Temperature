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
│   ├── processed/         # .hdf files converted to .csv
│   │   ├── modis/ 
│   │   ├── fluxnet/ 
│   │   └── amsre
│   ├── raw/               # .hdf files
│   │   ├── modis/
│   │   └── amsre
├── notebooks/            # Jupyter notebooks for analysis and visualization
├── outputs/
│   ├── amsre/         
│   ├── fluxnet/         
│   └── modis/
├──src/
│   ├── modis/
│   │   ├── download.py         # Download data from modis
│   │   ├── process.py          # Data processing
│   │   └── utils.py      
├── amsre/
│   │   ├── download.py  
│   │   ├── process.py   
│   │   └── plot.py   
├── fluxent/
│   │   ├── availability.py  
│   │   ├── filter.py 
│   │   ├── load.py 
│   │   ├── transform.py   
│   │   └── utils.py    
├── models/
│   ├── regression.py       
├── main.py
├── docs/                 # Supporting documentation and references  
├── requirements.txt
└── README.md             # Project description
```

## 📚 References

- [CIMRL2PAD-UVEG-TEC-RAS-D2 Technical Note](./docs/CIMRL2PAD-UVEG-TEC-RAS-D2.pdf) — describes R&D activities for CIMR L2 product algorithm development
- Algorithm Theoretical Basis Documents (ATBDs) for AMSR-E and MODIS LST retrieval
- Regression methodologies for passive microwave LST estimation

## 👤 Author

**Kevin DUGARD**  
Intern at ENSEA, Cergy (France), Internship at UPC, Barcelona (Spain)
Supervised by Mercè Vall-Llossera Ferran, UPC associate professor
