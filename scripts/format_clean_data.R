# import the data
clean_data <- read.csv('data_clean/fuel_prices_cleaned_final.csv')

# changing data type
clean_data[['date']] <- as.POSIXct(clean_data[['date']], format = "%Y-%m-%d")
class(clean_data[['date']])

# change to default factor (nominal)
nominal_cols <- c('country', 'region')

for (col in nominal_cols) {
  clean_data[[col]] <- as.factor(clean_data[[col]])
  cat(col, 'new format:', class(clean_data[[col]]), '\n')
}

# change data type to ordinal  
clean_data$income_level <- factor(clean_data$income_level, levels = c("Low", "Middle","High"), ordered = TRUE)
class(clean_data$income_level)
levels(clean_data$income_level)

clean_data$subsidy_level <- factor(clean_data$subsidy_level, levels = c("Low", "Medium","High", "Very High"), ordered = TRUE)
class(clean_data$subsidy_level)
levels(clean_data$subsidy_level)

# clean up variables
rm(col)
rm(nominal_cols)

# overview of data
str(clean_data)
head(clean_data)
