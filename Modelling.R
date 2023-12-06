
library(readr)
car_data <- read_csv("car data.csv")
car_data <-car_data[,-1]
selling_price=car_data$Selling_Price
lm0 <- lm(selling_price~., data = car_data)
summary(lm0)
smp_size <- floor(0.8 * nrow(car_data))
set.seed(123)
train_ind <- sample(seq_len(nrow(car_data)), size = smp_size)
# get training and testing data
train <- car_data[train_ind, ]
test <-car_data[-train_ind, ]
str(train)
selling_price<-train$Selling_Price
lm0_train <- lm(selling_price ~ ., data = train)
fitted <- predict(lm0_train, test)
mse(test$Selling_Price, fitted)
X <- model.matrix(selling_price~ ., train)
lm1 <- glmnet(X, train$Selling_Price, family = 'gaussian', alpha = 0)
plot(lm1, xvar = 'lambda')
# check the first 10 lambdas used 
head(lm1$lambda, 10)
# regularized linear regression: LASSO (alpha = 1)
lm2 <- glmnet(X, train$Selling_Price, family = 'gaussian', alpha = 1)
# plot the coefficients by lambda
plot(lm2, xvar = 'lambda')

# check the first 10 lambdas used 
head(lm2$lambda, 10)
# regularized linear regression: ELASTIC NET
lm3 <- glmnet(X, train$Selling_Price, family = 'gaussian', alpha = 0.5)

# plot the coefficients by lambda
plot(lm3, xvar = 'lambda')
# check the first 10 lambdas used 
head(lm3$lambda, 10)
lm1_cv <- cv.glmnet(X, train$Selling_Price, family = 'gaussian', alpha = 0, nfolds = 5)
plot(lm1_cv)
lambda_opt <- lm1_cv$lambda.min

lm1_1 <- glmnet(X, train$Selling_Price, family = 'gaussian', alpha = 0,
                lambda = lambda_opt)
X_test <- model.matrix(Selling_Price~ ., test)
ncol(X_test)
names(X_test)
str(X_test)
fitted1_1 <- predict(lm1_1, X_test)

mse(test$Selling_Price, fitted1_1)
#By default, cv.glmnet pick lambda.1se instead of lambda.min. let's try prediction use that.
fitted1_1se <- predict(lm1_cv, X_test)
mse(test$Selling_Price, fitted1_1se)

# 3. elastic net 
# 5-fold cross validation with alpha = 0.5 (exercise)
lm3_cv <- cv.glmnet(X, train$Selling_Price, family = 'gaussian', alpha = 0.5, nfolds = 5)

plot(lm3_cv)

# get optimal lambda with min mean squared error
lambda_opt <- lm3_cv$lambda.min
# re-run glmnet but use lambda_opt as lambda
lm3_1 <- glmnet(X, train$Selling_Price, family = 'gaussian', alpha = 0.5,lambda = lambda_opt)
# get predicted values and mean squared errors
fitted3_1 <- predict(lm3_1, X_test)
mse(test$Selling_Price, fitted3_1)
# again, let's use lambda.1se to predict, directly use results from cv.glmnet (exercise)
fitted3_1se <- predict(lm3_cv, X_test)
lambda_opt
