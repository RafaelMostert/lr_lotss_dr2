# %% 
from glob import glob
import pandas as pd
from time import time
import multiprocessing
import pickle
import os
import sys
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table, join
from astropy import units as u
#from dotenv import load_dotenv, find_dotenv

BASE_LR_PATH = os.getenv('BASE_LR_PATH')
sys.path.append(os.path.join(BASE_LR_PATH, 'src'))
from mltier1 import MultiMLEstimator, parallel_process, get_sigma_all

# LOAD CONFIG from ENV file
# Save as .env
#LEGACY_DATA_PATH=/disk02/jsm/Legacy_data-south-13h/Legacy
#UNWISE_DATA_PATH=/disk02/jsm/Legacy_data-south-13h/unWISE
#REGION=s13a
#load_dotenv(find_dotenv())
COMBINED_DATA_PATH = os.getenv("COMBINED_DATA_PATH")
PARAMS_PATH = os.getenv("PARAMS_PATH")
THRESHOLD = os.getenv("LR_THRESHOLD")
RADIO_CATALOGUE = os.getenv("RADIOCAT")
OUTPUT_RADIO_CATALOGUE = os.getenv("LRCAT")
FIELD = os.getenv('FIELD')

# Default config parameters
base_optical_catalogue = COMBINED_DATA_PATH
hdf_pruned_optical = base_optical_catalogue.replace('.fits',
        f'_{FIELD}.fits')
#params = pickle.load(open(PARAMS_PATH, "rb"))
params = [pd.read_pickle(os.path.join(BASE_LR_PATH,'data/params',f))
        for f in ['lofar_params-n13c.pckl',
            'lofar_params-s0a.pckl','lofar_params-s13a.pckl']]

colour_limits = np.array([0.7, 1.2, 1.5, 2. , 2.4, 2.8, 3.1, 3.6, 4.1])
threshold = float(THRESHOLD)
max_major = 15
radius = 15

# input_catalogue = os.path.join(
#     os.path.join(data_path, "samples", "LoTSS_DR2_rolling.gaus_0h.fits"))
# output_catalogue = os.path.join(
#     os.path.join(data_path, "samples", "LoTSS_DR2_rolling.gaus_0h.lr.fits"))
input_catalogue = RADIO_CATALOGUE
output_catalogue = OUTPUT_RADIO_CATALOGUE

# %% 
#bin_list, centers, Q_0_colour, n_m, q_m = params
centers = np.mean([p[1] for p in params], axis=0)
Q_0_colour = np.mean([p[2] for p in params], axis=0)
n_m = np.mean([p[3] for p in params], axis=0)
q_m = np.mean([p[4] for p in params], axis=0)


## Load the catalogues
print("Load input catalogue")
lofar= Table.read(input_catalogue)
print("Load optical catalogue")
if os.path.exists(hdf_pruned_optical):
    combined = Table.read(hdf_pruned_optical)
else:
    combined = Table.read(base_optical_catalogue)
    min_ra = np.min(lofar['RA'])-0.01
    max_ra = np.max(lofar['RA'])+0.01
    min_dec = np.min(lofar['DEC'])-0.01
    max_dec = np.max(lofar['DEC'])+0.01
    start = time()
    print("Pruning optical cat step 1/4")
    combined = combined[combined['RA']> min_ra]
    print(f"Pruning optical cat step 2/4. (Seconds elapsed on prev step: {time()-start:.0f})")
    combined = combined[combined['RA']< max_ra]
    print(f"Pruning optical cat step 3/4. (Seconds elapsed on prev step: {time()-start:.0f})")
    combined = combined[combined['DEC']> min_dec]
    print(f"Pruning optical cat step 4/4. (Seconds elapsed on prev step: {time()-start:.0f})")
    combined = combined[combined['DEC']< max_dec]
    combined.write(hdf_pruned_optical, format="fits")

## Get the coordinates
coords_combined = SkyCoord(combined['RA'], 
                        combined['DEC'], 
                        unit=(u.deg, u.deg), 
                        frame='icrs')
coords_lofar = SkyCoord(lofar['RA'], 
                    lofar['DEC'], 
                    unit=(u.deg, u.deg), 
                    frame='icrs')

## Get the colours for the combined catalogue
print("Get auxiliary columns")
combined["colour"] = combined["MAG_R"] - combined["MAG_W1"]
combined_aux_index = np.arange(len(combined))
combined_legacy = (
    ~np.isnan(combined["MAG_R"]) & 
    ~np.isnan(combined["MAG_W1"]) & 
    ~np.isnan(combined["MAG_W2"])
)
combined_wise =(
    np.isnan(combined["MAG_R"]) & 
    ~np.isnan(combined["MAG_W1"])
)
combined_wise2 =(
    np.isnan(combined["MAG_R"]) & 
    np.isnan(combined["MAG_W1"])
)

# Start with the W2-only, W1-only, and "less than lower colour" bins
colour_bin_def = [{"name":"only W2", "condition": combined_wise2},
                {"name":"only WISE", "condition": combined_wise},
                {"name":"-inf to {}".format(colour_limits[0]), 
                "condition": (combined["colour"] < colour_limits[0])}]

# Get the colour bins
for i in range(len(colour_limits)-1):
    name = "{} to {}".format(colour_limits[i], colour_limits[i+1])
    condition = ((combined["colour"] >= colour_limits[i]) & 
                (combined["colour"] < colour_limits[i+1]))
    colour_bin_def.append({"name":name, "condition":condition})

# Add the "more than higher colour" bin
colour_bin_def.append({"name":"{} to inf".format(colour_limits[-1]), 
                    "condition": (combined["colour"] >= colour_limits[-1])})

# Apply the categories
combined["category"] = np.nan
for i in range(len(colour_bin_def)):
    combined["category"][colour_bin_def[i]["condition"]] = i

## Define number of CPUs
n_cpus_total = multiprocessing.cpu_count()
n_cpus = max(1, n_cpus_total-1)
print(f"Use {n_cpus} CPUs")

## Start matching
print("X-match")
idx_lofar, idx_i, d2d, d3d = search_around_sky(
    coords_lofar, coords_combined, radius*u.arcsec
    )
idx_lofar_unique = np.unique(idx_lofar)
def apply_ml(i, likelihood_ratio_function):
    idx_0 = idx_i[idx_lofar == i]
    d2d_0 = d2d[idx_lofar == i]
    
    category = combined["category"][idx_0].astype(int)
    mag = combined["MAG_R"][idx_0]
    mag[category == 0] = combined["MAG_W2"][idx_0][category == 0]
    mag[category == 1] = combined["MAG_W1"][idx_0][category == 1]
    
    lofar_ra = lofar[i]["RA"]
    lofar_dec = lofar[i]["DEC"]
    lofar_pa = lofar[i]["PA"]
    lofar_maj_err = lofar[i]["E_Maj"]
    lofar_min_err = lofar[i]["E_Min"]
    c_ra = combined["RA"][idx_0]
    c_dec = combined["DEC"][idx_0]
    c_ra_err = np.ones_like(c_ra)*0.6/3600.
    c_dec_err = np.ones_like(c_ra)*0.6/3600.
    
    sigma_0_0, det_sigma = get_sigma_all(lofar_maj_err, lofar_min_err, lofar_pa, 
                    lofar_ra, lofar_dec, 
                    c_ra, c_dec, c_ra_err, c_dec_err)

    lr_0 = likelihood_ratio_function(mag, d2d_0.arcsec, sigma_0_0, det_sigma, category)
    
    chosen_index = np.argmax(lr_0)
    result = [combined_aux_index[idx_0[chosen_index]], # Index
            (d2d_0.arcsec)[chosen_index],                        # distance
            lr_0[chosen_index]]                                  # LR
    return result
likelihood_ratio = MultiMLEstimator(Q_0_colour, n_m, q_m, centers)
def ml(i):
    return apply_ml(i, likelihood_ratio)
print("Run LR")
res = parallel_process(idx_lofar_unique, ml, n_jobs=n_cpus)
lofar["lr"] = np.nan                   # Likelihood ratio
lofar["lr_dist"] = np.nan              # Distance to the selected source
lofar["lr_index"] = np.nan             # Index of the optical source in combined
(lofar["lr_index"][idx_lofar_unique], 
    lofar["lr_dist"][idx_lofar_unique], 
    lofar["lr"][idx_lofar_unique]) = list(map(list, zip(*res)))

## 
lofar["lrt"] = lofar["lr"]
lofar["lrt"][np.isnan(lofar["lr"])] = 0
lofar["lr_index_sel"] = lofar["lr_index"]
lofar["lr_index_sel"][lofar["lrt"] < threshold] = np.nan

## Save combined matches
combined["lr_index_sel"] = combined_aux_index.astype(float)
print("Combine catalogues")
pwl = join(lofar, combined, join_type='left', keys='lr_index_sel')
print("Clean catalogues")
print("type of pwl is:", type(pwl))
for col in pwl.colnames:
    try:
        fv = pwl[col].fill_value
        if (isinstance(fv, np.float64) and (fv != 1e+20)):
            print(col, fv)
            pwl[col].fill_value = 1e+20
        print("Colname is:", col)
    except:
        pass
print("Save output")
pwl["RA_2"].name = "ra"
pwl["DEC_2"].name = "dec"
pwl["RA_1"].name = "RA"
pwl["DEC_1"].name = "DEC"
pwl.filled().write(output_catalogue, format="fits", overwrite=True)

    
