# 🌍 Land Surface Temperature Estimation from Satellite Data (AMSR-E, MODIS & Land Cover)

This project explores the estimation of **Land Surface Temperature (LST)** using satellite observations, as part of preparatory R&D work for the upcoming **Copernicus Imaging Microwave Radiometer (CIMR)** mission. Since CIMR is not yet launched, the study uses heritage sensors with similar characteristics — **AMSR-E**, **MODIS**, and optionally **WindSat** — in combination with **land cover data** and **FLUXNET** in-situ observations.

> 📌 **Current focus**: Europe only, to limit data volume and ensure regional consistency.

---

## 🎯 Objectives

- Estimate LST using:
  - **Microwave Brightness Temperatures (TBs)** from AMSR-E (19 GHz & 37 GHz)
  - **Thermal LST** from MODIS (as reference/training)
  - **Land cover classification** (to account for surface type)
  - **In-situ data** from FLUXNET (for model validation)
- Develop and evaluate **regression and machine learning models** to retrieve LST
- Contribute to **CIMR L2 algorithm design**

---

## 🔍 Methodology

1. **Data Acquisition**
   - AMSR-E TBs for 2005
   - MODIS Terra/Aqua LST products
   - FLUXNET LST & meteorological data
   - Land cover classification (0.25° resolution)

2. **Preprocessing**
   - Geolocation & time matching of MODIS & AMSR-E data
   - Filtering of cloudy MODIS pixels
   - Extraction of TBs for ascending/descending orbits
   - Matching with FLUXNET stations
   - Land cover extraction per station/grid cell

3. **Modeling**
   - Linear regression between TBs and surface temperatures
   - Machine learning using TBs, MODIS LST, and land cover as inputs
   - Use of MODIS as ground truth for training, FLUXNET for validation

4. **Evaluation**
   - Scatter plots, seasonal trends, R², RMSE
   - Station-wise and global analysis
   - Visual diagnostics on model performance

---

## 🧠 Machine Learning Focus

A regression model is currently under development with the following characteristics:

- **Inputs**:
  - Brightness temperatures from AMSR-E at 19 GHz and 37 GHz
  - MODIS LST
  - Land cover class
- **Output**:
  - Land Surface Temperature (LST)
- **Training**: MODIS LST
- **Validation**: FLUXNET temperature (where available)

---

## ⚙️ Tools & Technologies

- **Python Libraries**: `numpy`, `pandas`, `xarray`, `matplotlib`, `scikit-learn`, `netCDF4`, `pyhdf`, `rasterio`
- **Data formats**: `.hdf`, `.nc`, `.csv`, `.png`
- **Data Sources**:
  - AMSR-E, MODIS (NASA)
  - Land Cover (0.25° resolution)
  - FLUXNET (in-situ stations)

---

## 📁 Repository Structure
```
.
├── data/
│ ├── raw/ # Original satellite, land cover, FLUXNET data
│ ├── processed/                                               # Cleaned & matched datasets
│ ├── analysis/                                                # Regression metrics, analysis tables
├── outputs/                                                   # All plots (.png)
│ ├── amsre/
│ ├── fluxnet/
│ └── modis/
├── notebooks/                                                 # Development notebooks
├── src/
│ ├── amsre/                                                   # AMSR-E processing, matching, plotting
│ ├── modis/                                                   # MODIS downloading, processing, analysis
│ ├── land_cover/                                              # Land cover extraction and cleaning
│ └── merge/                                                   # Data fusion scripts
├── docs/                                                      # Technical notes and references
├── main.py                                                    # Main script for executing pipeline
├── requirements.txt
└── README.md
```

---

## 🧪 Code Overview

The `main.py` script orchestrates the full pipeline:

- Loads matched AMSR-E / MODIS / FLUXNET data
- Applies regressions
- Generates seasonal plots and scatter graphs
- Saves outputs to `outputs/` folder

You can also run specific modules:
- `src/amsre/plot_temp_evolution.py` → seasonal plots
- `src/amsre/plot_regressions.py` → scatter plots & regression fits

---

## 📊 Outputs

Outputs are saved in `.png` format under `outputs/`. Main categories:

- `outputs/amsre/dates/`: TB & LST maps by date
- `outputs/amsre/stations/`: Station-wise regressions
- `outputs/fluxnet/seasonal_evolution/`: LST trends per station
- `outputs/fluxnet/seasonal_temp_tb/`: Seasonal plots TB vs LST
- `outputs/modis/dates/`: LST maps from MODIS

---

## ⚠️ Execution Notes

- **Missing Columns**: CSVs missing expected fields (e.g. `brightness_temp_37v`, `temperature`) are skipped
- **Invalid Dates**: Files with malformed dates are ignored
- **Outliers**: Values <180K or >330K filtered out
- **Matching Errors**: If no match is found between TB and LST, the date is skipped

---

## 🧑‍💻 Example Usage

Run the main analysis:

```bash
python main.py
```

Or generate seasonal graphs manually:
```bash
from src.amsre.plot_temp_evolution import plot_seasonal_temp_evolution
plot_seasonal_temp_evolution("data/processed/amsre/matched/station_x.csv")
```

---

## 🔧 Installation

Instakk dependencies with : 

```bash 
pip install -r requirements.txt
```

---

## 📚 References

- 📄 [CIMRL2PAD-UVEG-TEC-RAS-D2.pdf](./docs/CIMRL2PAD-UVEG-TEC-RAS-D2.pdf) — Technical document for CIMR L2 algorithm development  
- 📄 [CIMR-MRD-v5.0-20230211_(Issued).pdf](./docs/CIMR-MRD-v5.0-20230211_(Issued).pdf) — *Copernicus Imaging Microwave Radiometer (CIMR) Mission Requirements Document*  
- 📄 [Holmes_et_al_2009.pdf](./docs/Holmes_et_al_2009.pdf) — *Land surface temperature from Ka band (37 GHz) passive microwave observations*, JGR, 2009  
- 📄 [Jimenez_et_al_2017.pdf](./docs/Jimenez_et_al_2017.pdf) — *Inversion of AMSR-E observations for land surface temperature estimation: Methodology and evaluation*, JGR: Atmospheres, 2017

---

## 👤 Author
**Kevin DUGARD**  
Student at ENSEA, Cergy (France), Internship at UPC, Barcelona (Spain)
Supervised by Mercè Vall-Llossera Ferran, UPC associate professor
