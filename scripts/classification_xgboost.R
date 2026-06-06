library(tidymodels)

# preprocess the data and split into training and testing sets
source("scripts/classification_split.R")

# Model: XGBoost
set.seed(123)

# Build the XGBoost model specification
xgboost_spec <- boost_tree(
  trees = 1000,
  tree_depth = tune(), # Tune this
  learn_rate = tune(), # Tune this
  min_n = tune() # Tune this
) |>
  set_engine("xgboost") |>
  set_mode("classification")

# Combine recipe and model into a workflow
xgboost_workflow <- workflow() |>
  add_model(xgboost_spec) |>
  add_recipe(classification_recipe)

# install.packages("doParallel")
library(doParallel)
cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

parallel_control <- control_grid(
  save_pred = TRUE,
  parallel_over = "everything",
  verbose = FALSE,
)

# Tune the model using cross-validation
xgboost_tune <- xgboost_workflow |>
  tune_grid(
    resamples = folds,
    grid = 20, # Number of random combinations to try
    control = parallel_control
  )

stopCluster(cl)

# Select the best hyperparameters based on accuracy
best_xgboost <- xgboost_tune |>
  select_best(metric = "accuracy")
print(best_xgboost)

# Finalize the workflow with the best hyperparameters
final_xgboost <- xgboost_workflow |>
  finalize_workflow(best_xgboost) |>
  fit(data = train_data)

# 1. Define the exact metric set you want
my_metrics <- metric_set(accuracy, precision, recall, f_meas)

# 2. Get your standard class predictions and bind them to the test data
xgboost_predictions <- predict(final_xgboost, test_data) |>
  bind_cols(test_data)

# 3. Calculate all four metrics at the exact same time
xgboost_metrics <- xgboost_predictions |>
  my_metrics(truth = impact_level, estimate = .pred_class)

print(xgboost_metrics)

# Get and plot the confusion matrix
xgboost_conf_matrix <- xgboost_predictions |>
  conf_mat(truth = impact_level, estimate = .pred_class)

print(xgboost_conf_matrix)

autoplot(xgboost_conf_matrix, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "red") +
  labs(title = "XGBoost Confusion Matrix", x = "Predicted", y = "Actual") +
  theme_minimal()

# install.packages("bundle")
library(bundle)

# 1. "Bundle" the model so the background XGBoost architecture
# is safely captured
bundled_model <- bundle(final_xgboost)

# 2. Save it to your project folder
saveRDS(bundled_model, file = "models/classification/final_xgboost_model.rds")

cat("Model saved successfully!")

library(vip)
install.packages("vip")

final_xgboost |>
  extract_fit_parsnip() |>
  vip(num_features = 10)
