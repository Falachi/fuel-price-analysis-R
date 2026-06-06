# Install packages 
install.packages("tidyverse")
install.packages("lubridate")

# Load required libraries
library(tidyverse) # Core data science packages (dplyr, tidyr, stringr, readr)
library(lubridate) # Specialized package for date conversion

# Read the raw CSV file into a dataframe
df <- read_csv("data_clean/fuel_prices_cleaned_final.csv")

# Data Transformation (Formatting & Reshaping)
# 1. Date Conversion & Feature Extraction
# Command: mutate() and lubridate functions
df_clean <- df %>%
  mutate(
    # Convert string to Date format. 'dmy' parses Day/Month/Year correctly.
    date = dmy(date), 
    
    # Extract temporal features for downstream grouping
    year = year(date),
    month = month(date),
    quarter = quarter(date)
  )

# 2. Reshaping from Wide to Long Format
# Command: pivot_longer() collapses the 3 fuel columns into a single dimension
df_long <- df_clean %>%
  pivot_longer(
    cols = c(petrol_usd_liter, diesel_usd_liter, lpg_usd_liter), 
    names_to = "fuel_type",          # New column holding the categorical fuel names
    values_to = "retail_price_usd"   # New column holding the numerical prices
  )

#Data Cleaning (String manipulation)
# Command: str_replace() and str_to_title() to fix messy text data
df_long <- df_long %>%
  mutate(
    # Remove the redundant "_usd_liter" text from the fuel types
    fuel_type = str_replace(fuel_type, "_usd_liter", ""),
    
    # Capitalize the first letter for clean presentation (e.g., "petrol" -> "Petrol")
    fuel_type = str_to_title(fuel_type) 
  )

#Data Discretisation (Categorisation/ Binning)
# Command: cut(), quantile(), and case_when() to convert continuous numbers into categories
df_long <- df_long %>%
  mutate(
    # A. Retail Price Categories (Using equal statistical thirds)
    price_category = cut(
      retail_price_usd,
      breaks = quantile(retail_price_usd, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE),
      labels = c("Low Retail", "Medium Retail", "High Retail"),
      include.lowest = TRUE
    ),
    
    # B. Tax Categories (Using equal statistical thirds)
    tax_category = cut(
      tax_percentage,
      breaks = quantile(tax_percentage, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE),
      labels = c("Low Tax", "Medium Tax", "High Tax"),
      include.lowest = TRUE
    ),
    
    # C. Crude Oil Categories (Using specific rule-based thresholds)
    crude_category = case_when(
      brent_crude_usd < 50 ~ "Cheap Crude",
      brent_crude_usd >= 50 & brent_crude_usd <= 80 ~ "Normal Crude",
      brent_crude_usd > 80 ~ "Expensive Crude",
      TRUE ~ "Unknown" 
    )
  )

#Data Aggregation
#DATASET 1: For Q1 (Monthly Regression)
q1_monthly_data <- df_long %>%
  # Included tax_category so it doesn't get deleted!
  group_by(country, region, income_level, tax_category, year, month, fuel_type) %>%
  summarize(
    avg_retail_price = mean(retail_price_usd, na.rm = TRUE),
    avg_crude_price = mean(brent_crude_usd, na.rm = TRUE),
    .groups = 'drop' 
  ) %>%
  # NEW: Create Historical Behavior (Lagged Variable)
  arrange(country, fuel_type, year, month) %>% # Sort chronologically first
  group_by(country, fuel_type) %>%
  mutate(
    # This grabs the crude price from 1 month ago
    historical_crude_price = lag(avg_crude_price, n = 1) 
  ) %>%
  ungroup() %>%
  drop_na(historical_crude_price) # Drops the very first month which has no history

print("--- Q1 Monthly Data Ready ---")
head(q1_monthly_data)


#DATASET 2: For Q2 (Quarterly Classification)
q2_quarterly_data <- df_long %>%
  group_by(country, region, income_level, tax_category, year, quarter, fuel_type) %>%
  summarize(
    avg_retail_price = mean(retail_price_usd, na.rm = TRUE),
    avg_crude_price = mean(brent_crude_usd, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  # NEW: Calculate Percentage Changes and "Impact Level" Class
  arrange(country, fuel_type, year, quarter) %>% # Sort chronologically first
  group_by(country, fuel_type) %>%
  mutate(
    # Calculate Quarter-over-Quarter % Change
    retail_pct_change = (avg_retail_price - lag(avg_retail_price)) / lag(avg_retail_price) * 100,
    crude_pct_change = (avg_crude_price - lag(avg_crude_price)) / lag(avg_crude_price) * 100
  ) %>%
  ungroup() %>%
  mutate(
    # Create the Target Variable for Classification based on the price change
    impact_level = case_when(
      abs(retail_pct_change) > 5 ~ "High Impact",
      abs(retail_pct_change) >= 2 ~ "Medium Impact",
      abs(retail_pct_change) < 2 ~ "Low Impact",
      TRUE ~ NA_character_
    )
  ) %>%
  drop_na(impact_level)

print("--- Q2 Quarterly Data Ready ---")
head(q2_quarterly_data)

# Extracting datasets 
write_csv(q1_monthly_data, "data_processed/export/Q1_Monthly_Regression_Data.csv")
write_csv(q2_quarterly_data, "data_processed/export/Q2_Quarterly_Classification_Data.csv")

print("Success! Both CSV files have been extracted and saved to your folder.")