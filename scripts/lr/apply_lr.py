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
FIELD_DATA = os.getenv('FIELD_DATA')
TEMP_RESULTS = os.getenv('TEMP_RESULTS')
sys.path.append(os.path.join(BASE_LR_PATH, 'src'))
from mltier1 import MultiMLEstimator, parallel_process, get_sigma_all

# LOAD CONFIG from ENV file
# Save as .env
#LEGACY_DATA_PATH=/disk02/jsm/Legacy_data-south-13h/Legacy
#UNWISE_DATA_PATH=/disk02/jsm/Legacy_data-south-13h/unWISE
#REGION=s13a
#load_dotenv(find_dotenv())
COMBINED_CAT_NAME = os.getenv("COMBINED_CAT_NAME")
CATALOGUE_PATH = os.getenv("CATALOGUE_PATH")
PARAMS_PATH = os.getenv("PARAMS_PATH")
THRESHOLD = os.getenv("LR_THRESHOLD")
FIELD = os.getenv('FIELD')
overwrite = bool(int(os.getenv('PIPE_OVERWRITE')))

srl_input = os.path.join(FIELD_DATA,os.getenv('SRL_NAME'))
gaus_input = os.path.join(FIELD_DATA,os.getenv('GAUS_NAME'))
srl_output = os.path.join(TEMP_RESULTS,os.getenv('SRL_LR_NAME'))
gaus_output = os.path.join(TEMP_RESULTS,os.getenv('GAUS_LR_NAME'))
if os.path.exists(srl_output) and os.path.exists(gaus_output) and not overwrite:
    print("DONE: Calculated crossmatch likelihoodratios for all sources and gaussians.")
    exit()

# Test
#testlr= Table.read("/data1/tap/data/catalogues/lr/LoTSS_DR2_v100.gaus_13h.lr-full.P21.fits")
#testlr_gaus=Table.read("/data1/tap/data/catalogues/lr/LoTSS_DR2_v100.gaus_13h.lr-full.P21.fits")
#print("TEST LR catalogue has the following columns:", testlr.colnames)
#print("TEST gaus LR catalogue has the following columns:", testlr_gaus.colnames)

# Default config parameters
base_optical_catalogue = os.path.join(CATALOGUE_PATH,COMBINED_CAT_NAME)
os.makedirs(os.path.join(CATALOGUE_PATH,'combined_subcats'),exist_ok=True)
hdf_pruned_optical = os.path.join(CATALOGUE_PATH,'combined_subcats',
        COMBINED_CAT_NAME.replace('.fits',f'_{FIELD}.fits'))
#params = pickle.load(open(PARAMS_PATH, "rb"))
params = [pd.read_pickle(os.path.join(BASE_LR_PATH,'data/params',f))
        for f in ['lofar_params-n13c.pckl',
            'lofar_params-s0a.pckl','lofar_params-s13a.pckl']]

colour_limits = np.array([0.7, 1.2, 1.5, 2. , 2.4, 2.8, 3.1, 3.6, 4.1])
threshold = float(THRESHOLD)
max_major = 15
radius = 15


# %% 
#bin_list, centers, Q_0_colour, n_m, q_m = params
print(params[0].info())
centers = np.mean([p[1] for p in params], axis=0)
Q_0_colour = np.mean([p[2] for p in params], axis=0)
n_m = np.mean([p[3] for p in params], axis=0)
q_m = np.mean([p[4] for p in params], axis=0)


## Load the catalogues
print("Load input catalogue")
lofar= Table.read(srl_input)
lofar_gaus= Table.read(gaus_input)
print("Load optical catalogue")
if os.path.exists(hdf_pruned_optical):
    combined = Table.read(hdf_pruned_optical)
else:
    print("Optical cat subset for this field is not yet created, proceeding to make now.")
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
    print("Final optical cat len:", len(combined))
    if len(combined)==0:
        print("Combined optical cat does not contain sources for this field.\n"
                "FAILED: The pipeline unsuccesfully ends here.")
        exit(1) # combined cat does not contain sources for this field
    #combined.write(hdf_pruned_optical, format="fits")

## Get the coordinates
coords_combined = SkyCoord(combined['RA'], 
                        combined['DEC'], 
                        unit=(u.deg, u.deg), 
                        frame='icrs')
coords_lofar = SkyCoord(lofar['RA'], 
                    lofar['DEC'], 
                    unit=(u.deg, u.deg), 
                    frame='icrs')
coords_lofar_gaus = SkyCoord(lofar_gaus['RA'], 
                    lofar_gaus['DEC'], 
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
# For sourcelist
idx_lofar, idx_i, d2d, d3d = search_around_sky(
    coords_lofar, coords_combined, radius*u.arcsec
    )
idx_lofar_unique = np.unique(idx_lofar)
# For gauslist
idx_lofar_gaus, idx_i_gaus, d2d_gaus, d3d_gaus = search_around_sky(
    coords_lofar_gaus, coords_combined, radius*u.arcsec
    )
idx_lofar_unique_gaus = np.unique(idx_lofar_gaus)

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

def apply_ml_gaus(i, likelihood_ratio_function):
    idx_0_gaus = idx_i_gaus[idx_lofar_gaus == i]
    d2d_0_gaus = d2d_gaus[idx_lofar_gaus == i]
    
    category_gaus = combined["category"][idx_0_gaus].astype(int)
    mag_gaus = combined["MAG_R"][idx_0_gaus]
    mag_gaus[category_gaus == 0] = combined["MAG_W2"][idx_0_gaus][category_gaus == 0]
    mag_gaus[category_gaus == 1] = combined["MAG_W1"][idx_0_gaus][category_gaus == 1]
    
    lofar_ra_gaus = lofar_gaus[i]["RA"]
    lofar_dec_gaus = lofar_gaus[i]["DEC"]
    lofar_pa_gaus = lofar_gaus[i]["PA"]
    lofar_maj_err_gaus = lofar_gaus[i]["E_Maj"]
    lofar_min_err_gaus = lofar_gaus[i]["E_Min"]
    c_ra_gaus = combined["RA"][idx_0_gaus]
    c_dec_gaus = combined["DEC"][idx_0_gaus]
    c_ra_err_gaus = np.ones_like(c_ra_gaus)*0.6/3600.
    c_dec_err_gaus = np.ones_like(c_ra_gaus)*0.6/3600.
    
    sigma_0_0_gaus, det_sigma_gaus = get_sigma_all(lofar_maj_err_gaus, lofar_min_err_gaus,
            lofar_pa_gaus, 
                    lofar_ra_gaus, lofar_dec_gaus, 
                    c_ra_gaus, c_dec_gaus, c_ra_err_gaus, c_dec_err_gaus)

    lr_0_gaus = likelihood_ratio_function(mag_gaus, d2d_0_gaus.arcsec, sigma_0_0_gaus,
            det_sigma_gaus, category_gaus)
    
    chosen_index_gaus = np.argmax(lr_0_gaus)
    result_gaus = [combined_aux_index[idx_0_gaus[chosen_index_gaus]], # Index
            (d2d_0_gaus.arcsec)[chosen_index_gaus],                        # distance
            lr_0_gaus[chosen_index_gaus]]                                  # LR
    return result_gaus

likelihood_ratio = MultiMLEstimator(Q_0_colour, n_m, q_m, centers)
def ml(i):
    return apply_ml(i, likelihood_ratio)
def ml_gaus(i):
    return apply_ml_gaus(i, likelihood_ratio)
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
print("Save sourcelist LR output")
pwl["RA_2"].name = "ra"
pwl["DEC_2"].name = "dec"
pwl["RA_1"].name = "RA"
pwl["DEC_1"].name = "DEC"
pwl.filled().write(srl_output, format="fits", overwrite=True)

# Run now for gaussian list
res = parallel_process(idx_lofar_unique_gaus, ml_gaus, n_jobs=n_cpus)
lofar_gaus["lr"] = np.nan                   # Likelihood ratio
lofar_gaus["lr_dist"] = np.nan              # Distance to the selected source
lofar_gaus["lr_index"] = np.nan             # Index of the optical source in combined
(lofar_gaus["lr_index"][idx_lofar_unique_gaus], 
    lofar_gaus["lr_dist"][idx_lofar_unique_gaus], 
    lofar_gaus["lr"][idx_lofar_unique_gaus]) = list(map(list, zip(*res)))

## 
lofar_gaus["lrt"] = lofar_gaus["lr"]
lofar_gaus["lrt"][np.isnan(lofar_gaus["lr"])] = 0
lofar_gaus["lr_index_sel"] = lofar_gaus["lr_index"]
lofar_gaus["lr_index_sel"][lofar_gaus["lrt"] < threshold] = np.nan

## Save combined matches
combined["lr_index_sel"] = combined_aux_index.astype(float)
print("Combine catalogues")
pwl = join(lofar_gaus, combined, join_type='left', keys='lr_index_sel')
print("Clean catalogues")
print("type of pwl is:", type(pwl))
for col in pwl.colnames:
    try:
        fv = pwl[col].fill_value
        if (isinstance(fv, np.float64) and (fv != 1e+20)):
            print(col, fv)
            pwl[col].fill_value = 1e+20
    except:
        pass
print("Save Gaus LR output")
pwl["RA_2"].name = "ra"
pwl["DEC_2"].name = "dec"
pwl["RA_1"].name = "RA"
pwl["DEC_1"].name = "DEC"
pwl.filled().write(gaus_output, format="fits", overwrite=True)
print("Final LR catalogue has the following columns:", pwl.colnames)
