#* @apiTitle Diabetes Dataset API
#* @apiDescription Data tool


#Loading Libraries
library(plumber)
library(tidyverse)
library(tidymodels)
library(ranger)
library(parallel)
library(GGally)
library(leaflet)
library(caret)
library(ggplot2)

#Loading our data.
beetus_data<-read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
#Changing to factors and adding names.  
beetus_data <- beetus_data |>
  mutate(
    Diabetes_binary = factor(Diabetes_binary, labels = c("No Diabetes", "Pre/Diabetes")),
    HighBP = factor(HighBP, labels = c("No High BP", "High BP")),
    HighChol = factor(HighChol, labels = c("No High Cholesterol", "High Cholesterol")),
    Smoker = factor(Smoker, labels = c("Non-Smoker", "Smoker")),
    PhysActivity = factor(PhysActivity, labels = c("No Physical Activity", "Physical Activity")),
    
    Sex = factor(Sex, labels = c("Female", "Male")),
    GenHlth = factor(GenHlth, levels=1:5, labels = c("Excellent", "Very Good", "Good", "Fair", "Poor")),
    Education = factor(Education,levels=1:6, labels = c(
      "Never Attended School or only kindergarten",
      "Grades 1-8",
      "Grades 9-11",
      "Grade 12/GED",
      "Some College/Technical School",
      "College Graduate"
    )),
    Income = factor(Income,levels=1:8, labels = c(
      "Less than $10,000",
      "$10,000 to $15,000",
      "$15,000 to $20,000",
      "$20,000 to $25,000",
      "$25,000 to $35,000",
      "$35,000 to $50,000",
      "$50,000 to $75,000",
      "$75,000 or more"
    )),
    Age = factor(Age,levels=1:13, labels = c(
      "18-24", "25-29", "30-34", "35-39", "40-44",
      "45-49", "50-54", "55-59", "60-64", "65-69",
      "70-74", "75-79", "80 or older"
    ))
  )
#Variables we're interested in
beetus_filtered <- beetus_data |>
  select(Diabetes_binary,HighBP,HighChol,PhysActivity,BMI,Age,Sex,GenHlth)

set.seed(10)

#RF recipe
rf_rec <- recipe(Diabetes_binary ~., data = beetus_filtered) |>
  step_dummy(all_nominal_predictors(), -all_outcomes()) |>
  step_normalize(all_numeric(), -all_outcomes()) |>
  prep(training = beetus_filtered, retain=TRUE)



baked_data <- bake(rf_rec,new_data=beetus_filtered)

final_model <-rand_forest(mtry=5) |> #mtry 5 was the best model
  set_engine("ranger") |>
  set_mode("classification") |>
  fit(Diabetes_binary~.,data=baked_data) 
  
#Pred endpoint
#* Enter variables
#* @param HighBP 
#* @param HighChol
#* @param PhysActivity
#* @param BMI
#* @param Age
#* @param Sex
#* @param GenHlth
#* @get /pred

function(HighBP = "High BP",
         HighChol = "High Cholesterol",
         PhysActivity = "Physical Activity",
         BMI = "25",
         Age = "30-34",
         Sex = "Male",
         GenHlth = "Excellent") {
  
  # Create a tibble with the input values, ensuring factor levels match training data
  pred_model <- tibble(
    HighBP = factor(HighBP, levels = c("No High BP", "High BP")),
    HighChol = factor(HighChol, levels = c("No High Cholesterol", "High Cholesterol")),
    PhysActivity = factor(PhysActivity, levels = c("No Physical Activity", "Physical Activity")),
    BMI = as.numeric(BMI),
    Age = factor(Age, levels=c("18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80 or older")),
    Sex = factor(Sex, levels = c("Female","Male")),
    GenHlth = factor(GenHlth, levels = c("Excellent","Very Good","Good","Fair","Poor"))
  )
  
  # Apply the same transformations as during training
  pred_processed <- bake(rf_rec, new_data = pred_model)
  
  # Make prediction using the final model
  pred_fn <- predict(final_model, pred_processed)
  
  # Return the prediction
  list(prediction = pred_fn$.pred_class)
}
#Example calls:
#curl -X 'GET' \
#'http://127.0.0.1:7171/pred?HighBP=No%20High%20BP&HighChol=No%20High%20Cholesterol&PhysActivity=No%20Physical%20Activity&BMI=80&Age=18-24&Sex=Female&GenHlth=Good' \
#-H 'accept: */*'

##curl -X 'GET' \
##'http://127.0.0.1:7171/pred?HighBP=High%20BP&HighChol=High%20Cholesterol&PhysActivity=No%20Physical%20Activity&BMI=50&Age=45-49&Sex=Male&GenHlth=Poor' \
##-H 'accept: */*'
##curl -X 'GET' \
##'http://127.0.0.1:7171/pred?HighBP=No%20High%20BP&HighChol=No%20High%20Cholesterol&PhysActivity=No%20Physical%20Activity&BMI=80&Age=18-24&Sex=Female&GenHlth=Good' \
##-H 'accept: */*'

#Info Endpoint
#* Name Info
#* @get /info 
function() {
  list(name = "Eric Song",
      githubpage = "https://eks32.github.io/Final-Project/EDA.html"
  )
}

# Function to compute the confusion matrix and plot it
#* Compute and Plot Confusion Matrix
#* @serializer png
#* @get /confusion
function() {
  #get predictions
  predictions <- predict(final_model, baked_data)
  pred_classes <- predictions$.pred_class
  #convert to factors so they play nice
  actual <- as.factor(baked_data$Diabetes_binary)
  predicted <- as.factor(pred_classes)
  # Compute the confusion matrix using caret library
  cm <- confusionMatrix(data = predicted, reference = actual)
  cm
  # Convert confusion matrix to a data frame
  cm_df <- as.data.frame(cm$table)
  colnames(cm_df) <- c("Prediction", "Truth", "Freq")
  # Create the plot
  plot_cm <- ggplot(cm_df, aes(x = Prediction, y = Truth, fill = Freq)) + 
    geom_tile(aes(fill = Freq), color = "grey") +
    labs(title = "Confusion Matrix", x = "Predicted", y = "Actual")
  #print plot
  print(plot_cm)
}

# Example Call:
# /confusion