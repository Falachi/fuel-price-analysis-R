library(tidymodels)
library(tidyverse)

# import the data
df <- read_csv("data_processed/Q2_Quarterly_Classification_Data.csv")
# View(df)

glimpse(df)
summary(df)

# get unique values
unique(df$tax_category)
unique(df$income_level)
unique(df$impact_level)

tax_levels <- c("Low Tax", "Medium Tax", "High Tax")
income_levels <- c("Low", "Middle", "High")
impact_levels <- c("Low Impact", "Medium Impact", "High Impact")

# encode data

# nominal data
df$country <- as.factor(df$country)
df$region <- as.factor(df$region)

# ordinal data
df$tax_category <- factor(
  df$tax_category,
  levels = tax_levels,
  ordered = TRUE
)
df$income_level <- factor(
  df$income_level,
  levels = income_levels,
  ordered = TRUE
)
df$impact_level <- factor(
  df$impact_level,
  levels = impact_levels,
  ordered = TRUE
)
glimpse(df)

# Modeling

# Install and load the required package
install.packages("rsample")
library(rsample)

# 1. Split into Training and Testing (e.g., 80/20)
set.seed(123)

library(dplyr)
install.packages("tsibble")
library(tsibble)

df |>
  mutate(time_index = yearquarter(paste(year, "Q", quarter, sep = ""))) |>
  group_by(country, fuel_type, time_index) |>
  summarise(unique_impacts = n_distinct(impact_level), .groups = "drop") |>
  filter(unique_impacts > 1)

df_ts <- df |>
  mutate(time_index = yearquarter(paste(year, "Q", quarter, sep = ""))) |>
  # Adding tax_category to the keys separates the data properly
  as_tsibble(index = time_index, key = c(country, fuel_type, tax_category))

# View(df_ts)
glimpse(df_ts)

range(df_ts$time_index)

# Create a time-based split

# Train: 2020 Q1 to 2024 Q4
train_data_raw <- df_ts |>
  filter(time_index <= yearquarter("2024 Q4"))

# Test: 2025 Q1 to 2026 Q2 (The rest of the data)
test_data_raw <- df_ts |>
  filter(time_index >= yearquarter("2025 Q1"))

glimpse(train_data_raw)
glimpse(test_data_raw)

# Convert test data to standard tibble and turn yearquarter into a standard Date
test_data <- test_data_raw |>
  as_tibble() |>
  mutate(time_index = as.Date(time_index))

train_data <- train_data_raw |>
  as_tibble() |>
  mutate(time_index = as.Date(time_index)) # Converts "2020 Q1" to "2020-01-01"

# save the data
write_csv(train_data, "data_split/classification/train_data.csv")
write_csv(test_data, "data_split/classification/test_data.csv")

# generic folds
library(timetk)
folds <- time_series_cv(
  data        = train_data,
  date_var    = time_index,
  initial     = "36 months", # Equivalent to 12 quarters
  assess      = "6 months", # Equivalent to 2 quarters
  skip        = "3 months", # Equivalent to 1 quarter
  cumulative  = TRUE
)

# Generic Recipe for Classification Models
classification_recipe <- recipe(impact_level ~ ., data = train_data) |>
  # 1. Convert the Date object into a raw number so XGBoost can read it
  step_mutate(time_index = as.numeric(time_index)) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors(), -all_outcomes()) |>
  step_zv(all_predictors())

# Modeling it based on that (3 models)

# Analysis of the results and comparison of the models

# Ordinal Logistic Regression, Random Forest, XGBoost, (SVM)

# 3 Roles:
# 1. Splitting, 1 model
# 2. 2 models
# 3. Analysis
