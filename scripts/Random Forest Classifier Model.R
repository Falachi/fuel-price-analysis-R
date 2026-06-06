# 1. Load libraries
library(randomForest)
library(caret)
library(ggplot2)
library(pROC)

# 2. Read train/test datasets
train_df <- read.csv("train_df.csv")
test_df  <- read.csv("test_df.csv")

# 3. Ensure factor levels match
train_df$impact_level <- trimws(train_df$impact_level)
test_df$impact_level  <- trimws(test_df$impact_level)

train_df$impact_level <- factor(train_df$impact_level,
                                levels = c("Low Impact","Medium Impact","High Impact"))
test_df$impact_level  <- factor(test_df$impact_level,
                                levels = c("Low Impact","Medium Impact","High Impact"))

# 4. Check class distribution
cat("Train class counts:\n")
print(table(train_df$impact_level))
cat("Test class counts:\n")
print(table(test_df$impact_level))

# 5. Train Random Forest model
set.seed(123)
rf_model <- randomForest(impact_level ~ quarter + country + fuel_type +
                           crude_pct_change,
                         data = train_df,
                         ntree = 300,
                         mtry = floor(sqrt(5)),
                         importance = TRUE)

# 6. Predict on test set
predictions <- predict(rf_model, newdata = test_df)

# 7. Confusion matrix
cm <- confusionMatrix(predictions, test_df$impact_level)
print(cm)

# 8. Per-class metrics
precision <- cm$byClass[, "Pos Pred Value"]
recall    <- cm$byClass[, "Sensitivity"]
f1        <- 2 * ((precision * recall) / (precision + recall))

metrics_df <- data.frame(
  Class     = rownames(cm$byClass),
  Precision = round(precision, 3),
  Recall    = round(recall, 3),
  F1_Score  = round(f1, 3),
  Balanced_Accuracy = round(cm$byClass[, "Balanced Accuracy"], 3)
)
print(metrics_df)

# 9. Variable importance plot
imp <- importance(rf_model)
imp_df <- data.frame(Variable = rownames(imp),
                     MeanDecreaseAccuracy = imp[, "MeanDecreaseAccuracy"])

ggplot(imp_df, aes(x = reorder(Variable, MeanDecreaseAccuracy),
                   y = MeanDecreaseAccuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance (Random Forest)",
       x = "Predictor",
       y = "Mean Decrease in Accuracy") +
  theme_minimal()

# 10. Class distribution plot
ggplot(train_df, aes(x = impact_level)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Class Distribution (Train Set)", x = "Impact Level", y = "Count") +
  theme_minimal()

ggplot(test_df, aes(x = impact_level)) +
  geom_bar(fill = "darkred") +
  labs(title = "Class Distribution (Test Set)", x = "Impact Level", y = "Count") +
  theme_minimal()

# 11. Confusion matrix heatmap
cm_table <- as.data.frame(cm$table)
ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix Heatmap") +
  theme_minimal()