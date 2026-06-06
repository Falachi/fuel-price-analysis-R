# @title
# import the libraries
library(readr)
library(tidyverse)

# import the data
df<-read_csv("data_processed/Q1_Monthly_Regression_Data.csv")
View(df)

glimpse(df)
summary(df)

# get unique values
unique(df$tax_category)
unique(df$income_level)

tax_levels <- c("Low Tax", "Medium Tax", "High Tax")
income_levels <- c("Low", "Middle", "High")

# encode data

# nominal data
df$country <- as.factor(df$country)
df$region <- as.factor(df$region)

# ordinal data
df$tax_category <- factor(df$tax_category, levels = tax_levels, ordered = TRUE)
df$income_level <- factor(
  df$income_level,
  levels = income_levels,
  ordered = TRUE
)

glimpse(df)

# Modeling
install.packages("glmnet")
install.packages("xgboost")
install.packages("ranger")
install.packages("tidymodels")
library(tidymodels)

# 1. Split into Training and Testing (e.g., 80/20)
set.seed(123)
data_split <- initial_split(df, prop = 0.80, strata = avg_retail_price)
train_data <- training(data_split)
test_data <- testing(data_split)

# 2. Create Cross-Validation folds for tuning/comparison
folds <- vfold_cv(train_data, v = 5, strata = avg_retail_price)

# 3. Build the preprocessing recipe
price_recipe <- recipe(avg_retail_price ~ ., data = train_data) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors(), -all_outcomes()) |>
  # Create an interaction because tax levels change how crude affects retail
  step_interact(terms = ~ starts_with("tax_category"):avg_crude_price) |>
  step_zv(all_predictors())

# Model 1: Linear Regression (Elastic Net)
linear_spec <- linear_reg(penalty = tune(), mixture = tune()) |>
  set_engine("glmnet") |>
  set_mode("regression")

# Model 2: XGBoost (Tree-based ensemble)
xgboost_spec <- boost_tree(
  trees = 1000,
  tree_depth = tune(), # Tune this
  learn_rate = tune(), # Tune this
  min_n = tune() # Tune this
) |>
  set_engine("xgboost") |>
  set_mode("regression")

# Model 3: Random Forest (ranger)

rf_spec <- rand_forest(
  trees = 500,
  mtry = tune(),
  min_n = tune()
) |>
  set_engine(
    "ranger",
    num.threads = 1   # Prevents CPU cores from fighting each other
  ) |>
  set_mode("regression")

# Combine recipe and models into a workflow set
model_set <- workflow_set(
  preproc = list(base_rec = price_recipe),
  models = list(
    linear = linear_spec,
    xgb = xgboost_spec,
    rf = rf_spec
  )
)

# Run the tuning/comparison (this grid-searches parameters for BOTH models)
tune_results <- model_set |>
  workflow_map(
    seed = 123,
    resamples = folds,
    grid = 10, # Tries 10 different combinations of parameters for each
    metrics = metric_set(rmse, mae, rsq)
  )

# Plot the results to see the winner
autoplot(tune_results,metric="rmse")

# Get the best parameters and models

# 1. Extract the tuning results for your winning model (base_rec_linear/base_rec_xgb/base_rec_rf)
winning_tune_res <- tune_results |>
  extract_workflow_set_result("base_rec_rf")

# 2. See the exact top parameter values (lowest RMSE)
best_parameters <- winning_tune_res |>
  select_best(metric = "rmse")

print(best_parameters)

# Get result and evaluate
# 1. Grab the best results for the winning model (e.g., xgb)
best_results <- tune_results |>
  extract_workflow_set_result("base_rec_rf") |>
  select_best(metric = "rmse")

# 2. Finalize the workflow with these best parameters
final_wf <- tune_results |>
  extract_workflow("base_rec_rf") |>
  finalize_workflow(best_results)

# 3. Fit to the entire training set and test on the testing set
final_fit <- last_fit(final_wf, data_split)

# Extract the fully trained model structure
final_model_fit <- final_fit |>
  extract_fit_parsnip()

print(final_model_fit)

# 1. Define the extra metrics you want to calculate
extra_metrics <- metric_set(rmse, rsq, mae, mape, huber_loss)

# 2. Extract the test predictions and calculate the metrics
final_fit |>
  augment() |>
  extra_metrics(truth = avg_retail_price, estimate = .pred)