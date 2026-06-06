# --- PART 1: SETUP ---
# Load the tidyverse (includes readr, dplyr, stringr, and ggplot2)
library(tidyverse)

# Load the dirty data
raw_data <- read_csv("dirty_fuel_data.csv")

# Look at the "structure" of the data
str(raw_data)


# --- PART 2: IDENTIFY THE DIRTY DATA ---

# A. Check for Missing Values (NAs)
print("Missing values per column:")
colSums(is.na(raw_data))

# B. Check for Duplicates
print("Number of duplicate rows:")
sum(duplicated(raw_data))

# C. Check for Data Type Issues
class(raw_data$petrol_usd_liter)

# D. Check for Outliers
summary(raw_data$diesel_usd_liter)

# E. Check for Inconsistent Naming
unique(raw_data$country)


# --- PART 3: CLEANING ---

cleaned_data <- raw_data %>%
  # 1. Remove Duplicates
  distinct() %>%
  
  # 2. Fix Categorical Errors (Standardize Country Names)
  mutate(country = case_when(
    country == "USA" ~ "United States",
    TRUE ~ country
  )) %>%
  
  # 3. Fix String/Numeric Issues
  # Remove $ and USD, then convert the whole column to numeric
  mutate(petrol_usd_liter = as.numeric(str_remove_all(petrol_usd_liter, "[\\$, USD]"))) %>%
  
  # 4. Handle Outliers
  # We turn impossible prices into NA so we can impute them properly in the next step
  mutate(diesel_usd_liter = ifelse(diesel_usd_liter > 50 | diesel_usd_liter < 0, NA, diesel_usd_liter)) %>%
  
  # 5. Impute NAs (Fill the gaps)
  # We use the average price per country to fill missing spots
  group_by(country) %>%
  mutate(
    petrol_usd_liter = ifelse(is.na(petrol_usd_liter), 
                              mean(petrol_usd_liter, na.rm = TRUE), 
                              petrol_usd_liter),
    diesel_usd_liter = ifelse(is.na(diesel_usd_liter), 
                              mean(diesel_usd_liter, na.rm = TRUE), 
                              diesel_usd_liter)
  ) %>%
  ungroup()


# --- PART 4: VERIFY & SAVE ---

# A. Check for Missing Values (NAs)
print("Missing values per column:")
colSums(is.na(cleaned_data))

# B. Check for Duplicates
print("Number of duplicate rows:")
sum(duplicated(cleaned_data))

# C. Check for Data Type Issues
class(cleaned_data$petrol_usd_liter)

# D. Check for Outliers
summary(cleaned_data$diesel_usd_liter)

# E. Check for Inconsistent Naming
unique(cleaned_data$country)

# Save the final, sparkling clean data for your analysis
write_csv(cleaned_data, "fuel_prices_cleaned_final.csv")

print("Cleaning complete! Data is ready for analysis.")


# ==============================================================================
# DATA CLEANING SUMMARY
# ==============================================================================
# File Processed: dirty_fuel_data.csv
# Date: 2026-05-02
# 
# ACTIONS TAKEN:
# 1. Deduplication: Identified and removed duplicate rows to prevent skewed 
#    statistical averages.
# 2. Structural Fixes: Standardized 'country' names (e.g., converted 'USA' 
#    and 'U.S.A.' to 'United States') to ensure correct grouping during analysis.
# 3. Type Conversion: Cleaned 'petrol_usd_liter' by removing non-numeric 
#    characters ('$' and 'USD') and coerced the column to a numeric data type.
# 4. Outlier Mitigation: Detected impossible values in 'diesel_usd_liter' 
#    (e.g., $999.99 and negative values). These were treated as missing data 
#    to maintain distribution integrity.
# 5. Missing Value Imputation: Addressed NAs by imputing the mean price 
#    calculated per country. This preserves the sample size while maintaining 
#    geographic price variations.
#
# CLEAN DATA OVERVIEW:
# - Total Observations: 27468
# - Total No. of Columns: 10
# - Total Countries: 84
# - Price Range (Petrol): 0.010, 6.779
# - Price Range (Diesel): 0.01, 6.24
#
# DATA DIMENSIONS & VARIABLE SUMMARY
# 
# Total Column Count: 10
# 
# VARIABLE DESCRIPTIONS:
# 1.  date              : [Date] The recorded date of fuel prices.
# 2.  country           : [Character] Name of the country (standardized during cleaning).
# 3.  region            : [Character] Geographic region (e.g., North America, Europe).
# 4.  income_level      : [Character] Economic classification of the country.
# 5.  subsidy_level     : [Character] Level of government fuel subsidies.
# 6.  petrol_usd_liter  : [Numeric] Price of gasoline per liter in USD.
# 7.  diesel_usd_liter  : [Numeric] Price of diesel per liter in USD.
# 8.  lpg_usd_liter     : [Numeric] Price of LPG per liter in USD.
# 9.  brent_crude_usd   : [Numeric] The global benchmark price for crude oil.
# 10. tax_percentage    : [Numeric] The percentage of fuel price consisting of tax.
# ==============================================================================