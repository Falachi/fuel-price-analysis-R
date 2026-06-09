# ============================================================
# Multinomial Logistic Regression Classification Model
# Classification of Impact Level of Global Fuel Price Changes
# ============================================================

# ------------------------------------------------------------
# 1. Install and load required libraries
# ------------------------------------------------------------

# Install packages if not already installed
packages <- c("dplyr", "caret", "nnet", "ggplot2", "pROC")

installed_packages <- packages %in% rownames(installed.packages())

if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

# Load libraries
library(dplyr)
library(caret)
library(nnet)
library(ggplot2)
library(pROC)

# ------------------------------------------------------------
# 2. Load training and testing datasets
# ------------------------------------------------------------

# Use this if files are in the same folder as your R script / Rmd file
# train_df <- read.csv("train_df.csv")
# test_df  <- read.csv("test_df.csv")

# If R cannot find the files, use this instead:
train_df <- read.csv(file.choose())
test_df  <- read.csv(file.choose())

# ------------------------------------------------------------
# 3. Check dataset structure
# ------------------------------------------------------------

str(train_df)
str(test_df)

names(train_df)
names(test_df)

# ------------------------------------------------------------
# 4. Prepare target variable: impact_level
# ------------------------------------------------------------

# Remove extra spaces if any
train_df$impact_level <- trimws(train_df$impact_level)
test_df$impact_level  <- trimws(test_df$impact_level)

# Convert impact_level into factor with fixed class order
train_df$impact_level <- factor(
  train_df$impact_level,
  levels = c("Low Impact", "Medium Impact", "High Impact")
)

test_df$impact_level <- factor(
  test_df$impact_level,
  levels = c("Low Impact", "Medium Impact", "High Impact")
)

# Check class distribution
cat("Training class distribution:\n")
print(table(train_df$impact_level))

cat("Testing class distribution:\n")
print(table(test_df$impact_level))

# ------------------------------------------------------------
# 5. Prepare predictor variables
# ------------------------------------------------------------

# Predictors used:
# quarter            = quarterly/time effect
# country            = country-level differences
# fuel_type          = petrol/diesel/LPG differences
# crude_pct_change   = global crude oil price movement

# Convert categorical predictors into factors
categorical_cols <- c("quarter", "country", "fuel_type")

for (col in categorical_cols) {
  train_df[[col]] <- factor(train_df[[col]])
  test_df[[col]]  <- factor(test_df[[col]], levels = levels(train_df[[col]]))
}

# ------------------------------------------------------------
# 6. Scale numerical predictor
# ------------------------------------------------------------

# Scaling helps multinomial logistic regression handle numeric variables better
crude_mean <- mean(train_df$crude_pct_change, na.rm = TRUE)
crude_sd   <- sd(train_df$crude_pct_change, na.rm = TRUE)

train_df$crude_pct_change_scaled <- 
  (train_df$crude_pct_change - crude_mean) / crude_sd

test_df$crude_pct_change_scaled <- 
  (test_df$crude_pct_change - crude_mean) / crude_sd

# ------------------------------------------------------------
# 7. Remove rows with missing values in model variables
# ------------------------------------------------------------

model_vars <- c(
  "impact_level",
  "quarter",
  "country",
  "fuel_type",
  "crude_pct_change_scaled"
)

train_model <- train_df %>%
  select(all_of(model_vars)) %>%
  na.omit()

test_model <- test_df %>%
  select(all_of(model_vars)) %>%
  na.omit()

# Check final number of rows used
cat("Training rows used:", nrow(train_model), "\n")
cat("Testing rows used:", nrow(test_model), "\n")

# ------------------------------------------------------------
# 8. Build Multinomial Logistic Regression model
# ------------------------------------------------------------

set.seed(123)

multi_log_model <- multinom(
  impact_level ~ quarter + country + fuel_type + crude_pct_change_scaled,
  data = train_model,
  maxit = 1000,
  trace = FALSE
)

# View model summary
summary(multi_log_model)

# ------------------------------------------------------------
# 9. Predict class labels on test data
# ------------------------------------------------------------

pred_class <- predict(
  multi_log_model,
  newdata = test_model,
  type = "class"
)

pred_class <- factor(
  pred_class,
  levels = c("Low Impact", "Medium Impact", "High Impact")
)

# ------------------------------------------------------------
# 10. Predict class probabilities
# ------------------------------------------------------------

pred_prob <- predict(
  multi_log_model,
  newdata = test_model,
  type = "probs"
)

# View first few predicted probabilities
head(pred_prob)

# ------------------------------------------------------------
# 11. Confusion Matrix
# ------------------------------------------------------------

conf_matrix <- confusionMatrix(
  pred_class,
  test_model$impact_level
)

conf_matrix

# ------------------------------------------------------------
# 12. Extract Accuracy, Precision, Recall and F1-score
# ------------------------------------------------------------

# Overall accuracy
accuracy <- conf_matrix$overall["Accuracy"]

# Per-class metrics
precision <- conf_matrix$byClass[, "Pos Pred Value"]
recall    <- conf_matrix$byClass[, "Sensitivity"]

f1_score <- 2 * ((precision * recall) / (precision + recall))

metrics_df <- data.frame(
  Class = rownames(conf_matrix$byClass),
  Precision = round(precision, 4),
  Recall = round(recall, 4),
  F1_Score = round(f1_score, 4),
  Balanced_Accuracy = round(conf_matrix$byClass[, "Balanced Accuracy"], 4)
)

# Print results
cat("Overall Accuracy:", round(accuracy, 4), "\n")
print(metrics_df)

# ------------------------------------------------------------
# 13. Macro-average metrics
# ------------------------------------------------------------

macro_precision <- mean(precision, na.rm = TRUE)
macro_recall <- mean(recall, na.rm = TRUE)
macro_f1 <- mean(f1_score, na.rm = TRUE)

macro_metrics <- data.frame(
  Metric = c("Accuracy", "Macro Precision", "Macro Recall", "Macro F1-score"),
  Value = round(c(accuracy, macro_precision, macro_recall, macro_f1), 4)
)

macro_metrics

# ------------------------------------------------------------
# 14. Visualise Confusion Matrix as Heatmap
# ------------------------------------------------------------

cm_table <- as.data.frame(conf_matrix$table)

ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 5, fontface = "bold") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Confusion Matrix Heatmap",
    subtitle = "Multinomial Logistic Regression",
    x = "Actual Impact Level",
    y = "Predicted Impact Level",
    fill = "Count"
  ) +
  theme_minimal()

# ------------------------------------------------------------
# 15. Multiclass ROC/AUC
# ------------------------------------------------------------

# Multiclass AUC using pROC
multi_auc <- multiclass.roc(
  response = test_model$impact_level,
  predictor = pred_prob
)

multi_auc_value <- auc(multi_auc)

cat("Multiclass AUC:", round(as.numeric(multi_auc_value), 4), "\n")

# ------------------------------------------------------------
# 16. One-vs-All ROC Curves for each class
# ------------------------------------------------------------

# Low Impact vs Others
roc_low <- roc(
  response = ifelse(test_model$impact_level == "Low Impact", 1, 0),
  predictor = pred_prob[, "Low Impact"]
)

# Medium Impact vs Others
roc_medium <- roc(
  response = ifelse(test_model$impact_level == "Medium Impact", 1, 0),
  predictor = pred_prob[, "Medium Impact"]
)

# High Impact vs Others
roc_high <- roc(
  response = ifelse(test_model$impact_level == "High Impact", 1, 0),
  predictor = pred_prob[, "High Impact"]
)

# Plot ROC curves
plot(
  roc_low,
  col = "blue",
  lwd = 2,
  main = "One-vs-All ROC Curves for Multinomial Logistic Regression"
)

plot(
  roc_medium,
  col = "orange",
  lwd = 2,
  add = TRUE
)

plot(
  roc_high,
  col = "darkgreen",
  lwd = 2,
  add = TRUE
)

abline(a = 0, b = 1, lty = 2, col = "red")

legend(
  "bottomright",
  legend = c(
    paste("Low Impact AUC =", round(auc(roc_low), 4)),
    paste("Medium Impact AUC =", round(auc(roc_medium), 4)),
    paste("High Impact AUC =", round(auc(roc_high), 4))
  ),
  col = c("blue", "orange", "darkgreen"),
  lwd = 2
)

# ------------------------------------------------------------
# 17. Final Results Summary
# ------------------------------------------------------------

cat("\nMultinomial Logistic Regression Results\n")
cat("--------------------------------------\n")
cat("Accuracy        :", round(accuracy, 4), "\n")
cat("Macro Precision :", round(macro_precision, 4), "\n")
cat("Macro Recall    :", round(macro_recall, 4), "\n")
cat("Macro F1-score  :", round(macro_f1, 4), "\n")
cat("Multiclass AUC  :", round(as.numeric(multi_auc_value), 4), "\n")

cat("\nPer-Class Metrics:\n")
print(metrics_df)