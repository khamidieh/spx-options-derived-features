###############################################################################
# 
#   SPX OPTIONS FEATURE EXTRACTION SCRIPT
#
#   This script processes raw CBOE SPX/SPXW option files and constructs a daily
#   dataset of *derived features* such as:
#       - Put/Call volume ratios
#       - Open interest ratios
#       - Notional volume
#       - Short- vs. long-dated volume concentration
#       - Normalized at-the-money (ATM) straddle prices
#       - Cross-sectional implied volatility statistics
#       - Volatility skew measures
#       - Moneyness features
#
#   INPUT DATA (Required)
#   ----------------------
#   • CSV files containing raw option records purchased directly from CBOE.
#   • Must include the following columns:
#         'quote_date', 'expiration', 'strike', 'option_type',
#         'trade_volume', 'bid_1545', 'ask_1545',
#         'active_underlying_price_1545', 'implied_volatility_1545',
#         'open_interest', 'root'
#
#   LICENSE / DATA USE NOTE
#   ------------------------
#   • CBOE prohibits redistribution of raw option data purchased directly
#     from them.
#   • However, *derived features* (aggregations, transformations, statistics)
#     may be published for non-commercial and non-competitive academic use.
#   • This script outputs TWO datasets:
#
#       (1) spx_options_features_data.csv
#            → Full feature set (for personal use only; contains no raw
#              CBOE fields but uses intermediate columns not intended for
#              reposting).
#
#       (2) spx_options_features_data_public.csv
#            → Safe, public version containing ONLY derived variables intended
#              for sharing or publication in conjunction with the paper below.
#
#   CITATION
#   --------
#   If you use this script or the derived features in your research, please cite:
#
#       Hamidieh, Kam (2025). 
#       "A Daily Feature Dataset Derived from S&P 500 Index (SPX) Options."
#       SSRN Working Paper. 
#       [URL to be inserted]
#
#   USAGE
#   -----
#   1. Set `data_location` to the folder containing your raw CBOE CSV files.
#   2. Set `results_location` to the folder where you want outputs stored.
#   3. Run the script. Processing 2,000+ files typically takes 5–10 minutes.
#
#   OUTPUTS
#   -------
#   - spx_options_features_data.csv
#   - spx_options_features_data_public.csv    (safe to share)
#   - spx_features_plots.pdf                 (time-series plots of each feature)
#
###############################################################################


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import datetime
from matplotlib.backends.backend_pdf import PdfPages

# This is where SPX options data are:
data_location    = 'C:\\Users\\khami\\Documents\\TRADING\\cboe_data\\cboe_spx_data'

# This is where the results will go:
results_location = 'C:\\Users\\khami\\Documents\\TRADING\\cboe_data\\spx_options_features_data'


###################################################################################
###################################################################################
###################################################################################
###
### Function to get the nearest at money IV values for dte 1 to max_dte
###

def get_nearest_iv(options, max_dte):
    """
    For each trading date and DTE, find the implied volatility (IV)
    of the nearest at-the-money call and put options.

    Parameters
    ----------
    options : DataFrame
        Must contain columns:
        'quote_date', 'dte', 'strike', 'underlying_price',
        'option_type', and 'iv'.

    max_dte : int
        Maximum days-to-expiration to include (inclusive).

    Returns
    -------
    DataFrame with columns:
        'quote_date', 'dte', 'atm_call_iv', 'atm_put_iv'
    """

    # --------------------------------------------------------
    # 1. Make a safe copy and filter DTE range
    # --------------------------------------------------------
    x = options.copy()

    keep_lower = x["dte"] >= 1
    keep_upper = x["dte"] <= max_dte
    keep_mask = keep_lower & keep_upper
    x = x.loc[keep_mask].copy()

    if len(x) == 0:
        result = pd.DataFrame(columns=["quote_date", "dte", "atm_call_iv", "atm_put_iv"])
        return result

    # --------------------------------------------------------
    # 2. Compute distance to at-the-money
    # --------------------------------------------------------
    x["dist_atm"] = (x["strike"] - x["underlying_price"]).abs()

    # --------------------------------------------------------
    # 3. Process CALLS
    # --------------------------------------------------------
    calls = x[x["option_type"] == "C"].copy()
    calls = calls.sort_values(by=["quote_date", "dte", "dist_atm"])

    # Keep the closest-to-ATM call per (date, dte)
    calls = calls.drop_duplicates(subset=["quote_date", "dte"], keep="first")

    calls = calls.loc[:, ["quote_date", "dte", "iv"]]
    calls = calls.rename(columns={"iv": "atm_call_iv"})

    # --------------------------------------------------------
    # 4. Process PUTS
    # --------------------------------------------------------
    puts = x[x["option_type"] == "P"].copy()
    puts = puts.sort_values(by=["quote_date", "dte", "dist_atm"])

    # Keep the closest-to-ATM put per (date, dte)
    puts = puts.drop_duplicates(subset=["quote_date", "dte"], keep="first")

    puts = puts.loc[:, ["quote_date", "dte", "iv"]]
    puts = puts.rename(columns={"iv": "atm_put_iv"})

    # --------------------------------------------------------
    # 5. Merge CALL and PUT results
    # --------------------------------------------------------
    result = pd.merge(
        left=calls,
        right=puts,
        on=["quote_date", "dte"],
        how="outer"
    )

    result = result.sort_values(by=["quote_date", "dte"])
    result = result.reset_index(drop=True)

    return result

###################################################################################
###################################################################################
###################################################################################
###
### Function to get the normalized atm straddle:
###

def get_normalized_straddle(options, max_dte):
    """
    Compute normalized_straddle = (ATM call price + ATM put price) / (underlying_price * sqrt(dte))
    for each (quote_date, dte) with 1 <= dte <= max_dte.

    Expects columns in `options`:
        'quote_date', 'dte', 'strike', 'underlying_price',
        'option_type', 'option_price'

    Returns
    -------
    DataFrame with columns:
        'quote_date', 'dte', 'normalized_straddle'
    """

    # 1) Copy and keep DTE in [1, max_dte]
    x = options.copy()
    x = x[(x["dte"] >= 1) & (x["dte"] <= max_dte)].copy()

    if x.empty:
        return pd.DataFrame(columns=["quote_date", "dte", "normalized_straddle"])

    # 2) Keep rows with valid option_price
    x = x[x["option_price"].notna()].copy()
    if x.empty:
        return pd.DataFrame(columns=["quote_date", "dte", "normalized_straddle"])

    # 3) Distance to ATM (by strike vs underlying)
    x["dist_atm"] = (x["strike"] - x["underlying_price"]).abs()

    # 4) Nearest-ATM CALL per (quote_date, dte)
    calls = x[x["option_type"] == "C"].sort_values(["quote_date", "dte", "dist_atm"])
    calls = calls.drop_duplicates(subset=["quote_date", "dte"], keep="first")
    calls = calls.loc[:, ["quote_date", "dte", "underlying_price", "option_price"]]
    calls = calls.rename(columns={"option_price": "atm_call_price"})

    # 5) Nearest-ATM PUT per (quote_date, dte)
    puts = x[x["option_type"] == "P"].sort_values(["quote_date", "dte", "dist_atm"])
    puts = puts.drop_duplicates(subset=["quote_date", "dte"], keep="first")
    puts = puts.loc[:, ["quote_date", "dte", "option_price"]]
    puts = puts.rename(columns={"option_price": "atm_put_price"})

    # 6) Merge nearest call/put
    z = pd.merge(left=calls, right=puts, on=["quote_date", "dte"], how="outer")
    z = z.sort_values(by=["quote_date", "dte"]).reset_index(drop=True)

    # 7) Compute normalized_straddle row by row:
    z["normalized_straddle"] = np.nan
    for i in range(len(z)):
        dte_val = z.at[i, "dte"]
        call_px = z.at[i, "atm_call_price"]
        put_px  = z.at[i, "atm_put_price"]
        underlying = z.at[i, "underlying_price"]

        if pd.notna(dte_val) and dte_val > 0 and pd.notna(call_px) and pd.notna(put_px) and pd.notna(underlying) and underlying > 0:
            denom = underlying * np.sqrt(dte_val/365)
            z.at[i, "normalized_straddle"] = (call_px + put_px) / denom

    # 8) Keep only valid results
    z = z[z["normalized_straddle"].notna()].reset_index(drop=True)
    result = z.loc[:, ["quote_date", "dte", "normalized_straddle"]]
    
    return result


###################################################################################
###################################################################################
###################################################################################
###
### Compute Skewness:
###

def compute_vol_skew(x, max_dte):
    """
    Compute volatility skew = median(IV OTM calls) - median(IV OTM puts)
    restricted to 1 <= dte <= max_dte, excluding IV = 0.

    Parameters
    ----------
    x : DataFrame
        Must contain columns: ['quote_date','dte','strike','underlying_price','option_type','iv']

    Returns
    -------
    DataFrame
        Columns: ['quote_date','dte','vol_skew']
    """
    results = []

    # Filter to valid range and exclude IV = 0
    x = x[(x["dte"] >= 1) & (x["dte"] <= max_dte) & (x["iv"] > 0)]

    # Group by date and dte
    for (date, dte), group in x.groupby(["quote_date", "dte"]):
        # OTM puts: strike < underlying
        otm_puts = group[(group["option_type"] == "P") & (group["strike"] < group["underlying_price"])]
        # OTM calls: strike > underlying
        otm_calls = group[(group["option_type"] == "C") & (group["strike"] > group["underlying_price"])]

        if not otm_puts.empty and not otm_calls.empty:
            skew = otm_calls["iv"].median() - otm_puts["iv"].median()
        else:
            skew = np.nan

        results.append({"quote_date": date, "dte": dte, "vol_skew": skew})
        
    result = pd.DataFrame(results).sort_values(["quote_date","dte"]).reset_index(drop=True)

    return result


###################################################################################
###################################################################################
###################################################################################
###
### This file processes a single file for feature extraction:
###

def extract_features(x):
    #
    # Get volume information first
    #
    put_volume   = np.nansum(x[ x['option_type'] == 'P']['trade_volume'])
    call_volume  = np.nansum(x[ x['option_type'] == 'C']['trade_volume'])
    total_volume = np.nansum(x['trade_volume'])
    #
    if (call_volume != 0.0):
        put_to_call_volume  = put_volume/call_volume
    else:
        put_to_call_volume = np.nan
    #
    # Get open interest information:
    #
    put_open_interest   = np.nansum(x[ x['option_type'] == 'P']['open_interest'])
    call_open_interest  = np.nansum(x[ x['option_type'] == 'C']['open_interest'])
    total_open_interest = np.nansum(x['open_interest'])
    if (call_open_interest != 0.0):
        put_to_call_open_interest  = put_open_interest/call_open_interest
    else:
        put_to_call_open_interest = np.nan
    
    # option price is middle of bid/ask
    x['option_price'] = np.where(x['bid_1545'].notna() & x['ask_1545'].notna(), (x['bid_1545'] + x['ask_1545']) / 2, np.nan)

    # Compute notional volume = trade_volume * option_price for all options
    x['notional'] = x['trade_volume'] * x['option_price']
    total_notional = np.nansum(x['notional'])
    
    # Compute dte first for all options (needed for short/long volume)
    x['dte'] = (x['expiration'] - x['quote_date']).dt.days

    # Compute short- vs. long-dated volume concentration using ALL options
    short_volume = np.nansum(x[x['dte'] <= 10]['trade_volume'])
    long_volume  = np.nansum(x[x['dte'] > 30]['trade_volume'])
    if long_volume != 0:
        short_to_long_volume = short_volume / long_volume
    else:
        short_to_long_volume = np.nan

    # Now restrict to SPXW for downstream IV and straddle computations
    df = x.loc[x['root'] == 'SPXW'].copy()

    # drop bid/ask, and root is not needed
    df.drop(columns = ['bid_1545', 'ask_1545', 'root'], inplace = True)

    # Rename some columns
    df.rename(columns={'active_underlying_price_1545': 'underlying_price', 'implied_volatility_1545': 'iv'}, inplace=True)

    # Get dte
    df['dte'] = (df['expiration'] - df['quote_date']).dt.days
    
    
    # Compute moneyness features; Useful to see whether trading is concentrated in OTM or ATM strikes:
    moneyness = df['strike'] / df['underlying_price']
    avg_moneyness = np.nanmean(moneyness)
    sd_moneyness  = np.nanstd(moneyness)
        
    # Get normalized straddle
    straddle_result = get_normalized_straddle(df, 60)
    mean_normalized_straddle_price = np.nanmean(straddle_result['normalized_straddle'])
    sd_normalized_straddle_price   = np.nanstd(straddle_result['normalized_straddle'])
    num_expirations = straddle_result["dte"].nunique()
    
    # Get the at the money iv for put and calls for dte between 1 and max_dte days
    
    iv_result = get_nearest_iv(df, 60)
    
    # Non-zero and non-na masks
    call_mask_notnan = iv_result["atm_call_iv"].notna()
    put_mask_notnan  = iv_result["atm_put_iv"].notna()
    call_mask_nonzero = iv_result["atm_call_iv"] != 0
    put_mask_nonzero  = iv_result["atm_put_iv"]  != 0
    
    valid_mask = call_mask_notnan & call_mask_nonzero & put_mask_notnan & put_mask_nonzero
    
   
    # Work only with valid rows
    iv_ok = iv_result.loc[valid_mask].copy()
    
    # Simple average (both sides are non-zero by construction)
    iv_ok["iv"] = 0.5 * (iv_ok["atm_call_iv"] + iv_ok["atm_put_iv"])

    # Downstream features
    mean_iv = np.nanmean(iv_ok["iv"])
    sd_iv   = np.nanstd(iv_ok["iv"])
       
    # Skewness
    skew_df   = compute_vol_skew(df, 60)
    mean_skew = np.nanmean(skew_df['vol_skew'])
    sd_skew   = np.nanstd(skew_df['vol_skew'])


    # Get the day of the week
    day_name = df['quote_date'].dt.day_name().iloc[0]
    date_    = df['quote_date'].iloc[0].strftime('%Y-%m-%d')
    
    #Get SPX value
    spx = df['underlying_price'].iloc[0]
    
    # Collect
    out = {
        'date':date_, 
        'day_name':day_name,
        'spx':spx,
        'put_volume':put_volume, 
        'call_volume':call_volume, 
        'total_volume':total_volume,
        'short_volume': short_volume,
        'long_volume': long_volume,
        'short_to_long_volume': short_to_long_volume,
        'notional': total_notional,
        'put_to_call_volume':put_to_call_volume,
        'put_open_interest':put_open_interest,
        'call_open_interest':call_open_interest,
        'total_open_interest':total_open_interest,
        'put_to_call_open_interest':put_to_call_open_interest,
        'num_expirations': num_expirations,
        'mean_normalized_straddle_price': mean_normalized_straddle_price,
        'sd_normalized_straddle_price': sd_normalized_straddle_price,
        'mean_iv':mean_iv,
        'sd_iv':sd_iv,
        'mean_skew':mean_skew,
        'sd_skew':sd_skew,
        'avg_moneyness': avg_moneyness,
        'sd_moneyness': sd_moneyness
    }
    
    return out

###
### Change directory:
###

os.chdir(data_location)

# Get the list of files
file_list = os.listdir()

# read in the file
usecols = ['quote_date','expiration','strike','option_type','trade_volume',
           'bid_1545','ask_1545','active_underlying_price_1545',
           'implied_volatility_1545','open_interest','root']

records = []

len_file_list = len(file_list)

print('---------------------------------------')
print('Number of files to process = ', len_file_list)
print('---------------------------------------')


start_time = time.time() # Expect around 5-10 minutes for 2118 files.

counter = 1
for single_file in file_list:
    print('Processing ', counter, 'out of', len_file_list)
    tmp_data = pd.read_csv(single_file, usecols=usecols, parse_dates=['quote_date','expiration'])
    features = extract_features(tmp_data)
    records.append(features)
    counter = counter + 1
end_time = time.time()     

print('--------------------------------------------------------')  
print('Time Taken (seconds) = ', np.round(end_time-start_time,0) )
print('--------------------------------------------------------')  
    
###
### Collect results
###

df = pd.DataFrame(records)
os.chdir(results_location)
df.to_csv('spx_options_features_data.csv', index = False)

df['date'] = pd.to_datetime(df['date'])

###
### Plot the results:
###

# Exclude unwanted columns
exclude_cols = ['date', 'day_name', 'spx']
cols_to_plot = [c for c in df.columns if c not in exclude_cols]

print(cols_to_plot)
# Output PDF file
output_path = 'spx_features_plots.pdf'

with PdfPages(output_path) as pdf:
    for col in cols_to_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df[col], linewidth=1.5)
        plt.title(f"{col} vs Date", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel(col, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
###
### Save the data the will be shared to avoid issues with CBOE:
###

public_columns = ['date', 'day_name','total_volume', 'short_volume','long_volume', 'short_to_long_volume', 'put_to_call_volume', \
                  'notional', 'total_open_interest','put_to_call_open_interest','num_expirations','mean_normalized_straddle_price', \
                      'sd_normalized_straddle_price', 'mean_skew', 'sd_skew']
df_public = df[public_columns].copy()
df_public.to_csv('spx_options_features_data_public.csv', index = False)

print(df_public.info())
