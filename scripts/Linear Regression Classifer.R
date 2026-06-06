# ============================================================
# Logistic Regression Classification Model
# Fuel Subsidy Level Classification
# ============================================================

# ------------------------------------------------------------
# 1. Load required libraries
# ------------------------------------------------------------

library(dplyr)
library(caret)
library(pROC)
library(ggplot2)

# ------------------------------------------------------------
# 2. Load training and testing datasets
# ------------------------------------------------------------

# Check working directory
getwd()
list.files()

# Use this if the CSV files are in the same folder as your R script / R Markdown file
# train_data <- read.csv("train_data.csv")
# test_data <- read.csv("test_data.csv")

# If R cannot find the files, use this instead:
train_data <- read.csv(file.choose())
test_data <- read.csv(file.choose())

# ------------------------------------------------------------
# 3. Check the structure of the datasets
# ------------------------------------------------------------

str(train_data)
str(test_data)

# View column names
names(train_data)
names(test_data)

# ------------------------------------------------------------
# 4. Create binary target variable: subsidy_level
# ------------------------------------------------------------

# Logic:
# Low Tax  = High Subsidy
# High Tax = Low Subsidy
# Medium Tax is removed because Logistic Regression here is binary classification

train_lr <- train_data %>%
  filter(tax_category %in% c("Low Tax", "High Tax")) %>%
  mutate(
    subsidy_level = ifelse(
      tax_category == "Low Tax",
      "High Subsidy",
      "Low Subsidy"
    )
  )

test_lr <- test_data %>%
  filter(tax_category %in% c("Low Tax", "High Tax")) %>%
  mutate(
    subsidy_level = ifelse(
      tax_category == "Low Tax",
      "High Subsidy",
      "Low Subsidy"
    )
  )

# ------------------------------------------------------------
# 5. Convert target variable into factor
# ------------------------------------------------------------

train_lr$subsidy_level <- factor(
  train_lr$subsidy_level,
  levels = c("Low Subsidy", "High Subsidy")
)

test_lr$subsidy_level <- factor(
  test_lr$subsidy_level,
  levels = c("Low Subsidy", "High Subsidy")
)

# Check class distribution
table(train_lr$subsidy_level)
table(test_lr$subsidy_level)

# ------------------------------------------------------------
# 6. Convert categorical predictors into factors
# ------------------------------------------------------------

categorical_cols <- c("region", "income_level", "fuel_type")

for (col in categorical_cols) {
  train_lr[[col]] <- factor(train_lr[[col]])
  test_lr[[col]] <- factor(test_lr[[col]], levels = levels(train_lr[[col]]))
}

# ------------------------------------------------------------
# 7. Build Logistic Regression model
# ------------------------------------------------------------

log_model <- glm(
  subsidy_level ~ region + income_level + fuel_type +
    year + quarter +
    avg_retail_price + avg_crude_price +
    retail_pct_change + crude_pct_change,
  data = train_lr,
  family = binomial(link = "logit")
)

# View model summary
summary(log_model)

# ------------------------------------------------------------
# 8. Predict probability on test data
# ------------------------------------------------------------

pred_prob <- predict(
  log_model,
  newdata = test_lr,
  type = "response"
)

# View first few predicted probabilities
head(pred_prob)

# ------------------------------------------------------------
# 9. Convert predicted probability into class prediction
# ------------------------------------------------------------

pred_class <- ifelse(
  pred_prob >= 0.5,
  "High Subsidy",
  "Low Subsidy"
)

pred_class <- factor(
  pred_class,
  levels = c("Low Subsidy", "High Subsidy")
)

# View first few predicted classes
head(pred_class)

# ------------------------------------------------------------
# 10. Confusion Matrix
# ------------------------------------------------------------

conf_matrix <- confusionMatrix(
  pred_class,
  test_lr$subsidy_level,
  positive = "High Subsidy"
)

# Print full confusion matrix result
conf_matrix

# ------------------------------------------------------------
# 11. Extract evaluation metrics
# ------------------------------------------------------------

accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Recall"]
f1_score <- conf_matrix$byClass["F1"]

# Print metrics
accuracy
precision
recall
f1_score

# Put metrics into a table
performance_metrics <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1-score"),
  Value = c(accuracy, precision, recall, f1_score)
)

performance_metrics

# ------------------------------------------------------------
# 12. Visualise Confusion Matrix
# ------------------------------------------------------------

# Create confusion matrix table
cm <- table(
  Predicted = pred_class,
  Actual = test_lr$subsidy_level
)

cm

# Convert confusion matrix to dataframe
cm_df <- as.data.frame(cm)

# Plot confusion matrix as heatmap
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6, fontface = "bold") +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  labs(
    title = "Confusion Matrix for Logistic Regression",
    subtitle = "Classification of High vs Low Fuel Subsidy",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Count"
  ) +
  theme_minimal()

# ------------------------------------------------------------
# 13. ROC Curve and AUC
# ------------------------------------------------------------

roc_obj <- roc(
  test_lr$subsidy_level,
  pred_prob,
  levels = c("Low Subsidy", "High Subsidy")
)

# Calculate AUC
auc_value <- auc(roc_obj)
auc_value

# Plot ROC curve
plot(
  roc_obj,
  main = "ROC Curve for Logistic Regression Model",
  col = "blue",
  lwd = 2
)

abline(
  a = 0,
  b = 1,
  lty = 2,
  col = "red"
)

# ------------------------------------------------------------
# 14. Final Results Summary
# ------------------------------------------------------------

cat("Logistic Regression Classification Results\n")
cat("------------------------------------------\n")
cat("Accuracy :", round(accuracy, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall   :", round(recall, 4), "\n")
cat("F1-score :", round(f1_score, 4), "\n")
cat("AUC      :", round(auc_value, 4), "\n")