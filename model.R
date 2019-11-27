####################################################
# project: "Capstone Report - Product Classification by Product Name"
# author: "Kai Notté"
# date: "25.11.2019"
####################################################

####################################################
# Load Libraries

# Load package: caret
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# Load package: tidytext
if(!require(tidytext)) install.packages("tidytext", repos = "http://cran.us.r-project.org")

# Load package: tidyverse
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")

# Load package: tm
if(!require(tm)) install.packages("tm", repos = "http://cran.us.r-project.org")

####################################################
# Load Data sets

# Load file: parts.csv
filePath_parts <- "./data/parts.csv"

if(!file.exists(filePath_parts)){
  download.file("https://github.com/kai-notte/edx-PH125.9x-productclassification/raw/master/data/parts.csv",
                destfile="./data/parts.csv",
                method="auto")
}
parts <- read_csv(filePath_parts, col_names = TRUE)

# Load file: part_categories.csv
filePath_categories <- "./data/part_categories.csv"
if(!file.exists(filePath_parts)){
  download.file("https://github.com/kai-notte/edx-PH125.9x-productclassification/raw/master/data/part_categories.csv",
                destfile="./data/part_categories.csv",
                method="auto")
}
categories <- read_csv(filePath_categories, col_names = TRUE)

## Add category name to `part`
parts <- inner_join(parts, categories, by = c("part_cat_id" = "id")) %>% select(part_num, name.x, name.y) %>% rename(pid = part_num, pname = name.x, cname = name.y)

## Remove unused df `categories`
rm(categories)

# Load file: colors.csv
filePath_colors <- "./data/colors.csv"
if(!file.exists(filePath_parts)){
  download.file("https://github.com/kai-notte/edx-PH125.9x-productclassification/raw/master/data/colors.csv",
                destfile="./data/colors.csv",
                method="auto")
}

# Adjust to lower cases to normalize the spelling
colors <- read_csv(filePath_colors, col_names = TRUE) %>% mutate(lcName = tolower(name))


####################################################
# Separate train and validation data set

## Validation data set set will be 10% of original data set
set.seed(1, sample.kind="Rounding")
index <- createDataPartition(y = parts$cname, times = 1, p = 0.1, list = FALSE)
temp1 <- parts[-index,]
temp2 <- parts[index,]

## Make sure every category name `cname` is in the train data set
validation <- temp2 %>% semi_join(temp1, by = "cname")

## Add rows removed from validation set back into train set
removed <- anti_join(temp2, validation)
train <- rbind(temp1, removed)

## Remove unused dfs
rm(removed, temp1, temp2, index, parts)

####################################################
# Prepare the data set `train`

## Remove spaces in pattern: " x " to "x"
train <- as_tibble(lapply(train, function(x) {
  gsub(" x ", "x", x)
}))

## Create tidy text data frame
train_token <- train %>% 
  unnest_tokens(output = word, input = pname) %>%
### Remove stop words
  anti_join(stop_words, by=c("word" = "word")) %>%
### Remove color information
  anti_join(colors, by=c("word" = "lcName")) %>%
### Remove numbers
  filter(!str_detect(word, "^[0-9]*$"))

## Create Document-Term-Matrix
train_dtm <- train_token %>%
### Count each word (term) for each product
  count(pid, word) %>%
### Cast a document-term matrix
  cast_dtm(document = pid, term = word, value = n)

## Reduce sparsity
train_dtm <- removeSparseTerms(train_dtm, sparse = .99)

## Create `train_y`, containing category names`
train_y <- as.data.frame(train_dtm$dimnames$Docs) %>% 
  left_join(train, by = c("train_dtm$dimnames$Docs" = "pid")) %>% 
  select("cname")


####################################################
# Train model on data set `train`
# Using the results of the report evalutation: RF with mtry = 32

# Configuration of trControl
control <- trainControl(method = "cv", 
                        number = 2, 
                        p = .9, 
                        savePrediction = "final")

# Define tuneGrid for mtry
tunegrid <- data.frame(mtry = 32)

# Train rf model on the train data set
set.seed(1, sample.kind = "Rounding")
model_fit <- train(x = as.matrix(train_dtm),
                   y = factor(train_y$cname),
                   method = "rf",
                   trControl = control,
                   tuneGrid = tunegrid)


####################################################
# Prepare data set `validation`

## Remove spaces in pattern: " x " to "x"
validation <- as_tibble(lapply(validation, function(x) {
  gsub(" x ", "x", x)
}))

## Create tidy text data frame
validation_token <- validation %>% 
  unnest_tokens(output = word, input = pname) %>%
### Remove stop words
  anti_join(stop_words, by=c("word" = "word")) %>%
### Remove color information
  anti_join(colors, by=c("word" = "lcName")) %>%
### Remove numbers
  filter(!str_detect(word, "^[0-9]*$"))

## Create Document-Term-Matrix
validation_dtm <- validation_token %>%
  # Count each word (term) for each product
  count(pid, word) %>%
  # Cast a document-term matrix
  cast_dtm(document = pid, term = word, value = n)

## Generate validation target vector
validation_y <- as.data.frame(validation_dtm$dimnames$Docs) %>% 
  left_join(validation, by = c("validation_dtm$dimnames$Docs" = "pid")) %>% 
  select("cname")

####################################################
# Evaluate model by using the data set `validation`

# Compute prediction by validaion data set
prediction <- predict(model_fit, validation_dtm)

# Evaluate prediction
mean(prediction == validation_y$cname)
