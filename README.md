# Load required libraries
library(caret)
library(randomForest)
library(ggplot2)

# Load dataset
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
train_data <- read.csv(train_url, na.strings = c("NA", ""))
test_data <- read.csv("/mnt/data/pml-testing (2).csv", na.strings = c("NA", ""))

# Remove columns with too many missing values
train_data <- train_data[, colSums(is.na(train_data)) == 0]
test_data <- test_data[, colSums(is.na(test_data)) == 0]

# Remove non-predictive columns (e.g., ID, timestamps, user names)
nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
train_data <- train_data[, !nzv$nzv]
train_data <- train_data[, -c(1:6)] # Removing first few non-predictive columns

test_data <- test_data[, colnames(test_data) %in% colnames(train_data)]

# Convert target variable to factor
train_data$classe <- as.factor(train_data$classe)

# Split data into training and validation sets
set.seed(123)
trainIndex <- createDataPartition(train_data$classe, p = 0.7, list = FALSE)
training_set <- train_data[trainIndex, ]
validation_set <- train_data[-trainIndex, ]

# Train a Random Forest model
set.seed(123)
rf_model <- randomForest(classe ~ ., data = training_set, ntree = 100)

# Model evaluation
rf_predictions <- predict(rf_model, validation_set)
conf_matrix <- confusionMatrix(rf_predictions, validation_set$classe)
print(conf_matrix)

# Make predictions on test set
final_predictions <- predict(rf_model, test_data)
print(final_predictions)

# Save predictions
write.csv(final_predictions, "predictions.csv", row.names = FALSE)
