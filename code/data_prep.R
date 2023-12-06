
library(tidymodels)
library(zoo)
library(forecast)


## Importing the data
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))  # set current working directory as directory holding this file
data <- read.table('../data/met_mlo_insitu_1_obop_hour_2000.txt')

for (i in 2001:2019){
  path = paste0('../data/met_mlo_insitu_1_obop_hour_', i, '.txt')
  temp = read.table(path)
  data = rbind(data, temp)
}

colnames(data) <- c("site_code", "year", "month", "day", "hour", "wind_direction", "wind_speed", "wind_steadiness_factor", "barometric_pressure", "temp_at_2", "temp_at_10", "temp_at_top", "relative_humidity", "precipitation_intensity")

data$wind_steadiness_factor[data$wind_steadiness_factor == -9] = NA
data[data == -999] = NA
data[data == -999.9] = NA
data[data == -99] = NA
data[data == -99.9] = NA


# Imputing missing values

cols_to_impute <- colnames(data)[! colnames(data) %in% c("site_code","year","month","day","hour")]

for (i in cols_to_impute) {
  data[i] <- data.frame(na.approx(data[i],rule=2))
}


# Feature engineering

data$date = paste0(data$year, "-", data$month, "-", data$day)
data$date = as.Date(data$date)

data$ah <- ((0.000002*data$temp_at_2^4)+(0.0002*data$temp_at_2^3)+(0.0095*data$temp_at_2^2)+(0.337*data$temp_at_2)+4.9034)*(data$relative_humidity/100) #calculating absolute humidity


# Looking at the total rainfall by month.

barchart <- data %>% group_by(month) %>% summarize(rainfall = sum(precipitation_intensity, na.rm=TRUE))

barplot(barchart$rainfall, names = c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"), xlab = "Month", ylab = "millimeters", main = "Total rainfall in Mauna Loa, Hawaii (by month)")


# Looking at the number of extreme precipitation events

sum(data$precipitation_intensity)/240 # Monthly precipitation

quantile((data %>% group_by(date) %>% summarise(pi = sum(precipitation_intensity)))$pi,c(.86, .90, .95, .99)) # top quantiles for precipitation.


# Looking at the difference in X values for these days

cor(drop_na(data[,c("wind_direction", "wind_speed","wind_steadiness_factor","barometric_pressure","temp_at_2","temp_at_10","temp_at_top","relative_humidity","precipitation_intensity","ah")]))



# Creating new_df which includes the predictors and the response

new_df <- data[,c("year","month","day","hour","date")]
new_df$pi_24 <- rollsum(data$precipitation_intensity, 24, align = 'left', fill=NA)

temp <- data.frame(rollmean(data[, c('wind_direction','wind_speed',"wind_steadiness_factor",
                                       "barometric_pressure","temp_at_2","temp_at_10","temp_at_top",
                                       "relative_humidity","ah", "precipitation_intensity")], 
                              3,
                              align="right",
                              fill = NA))

colnames(temp) <- c("wd","ws","wsf","bp","t2","t10","tt","rh","ah","pi")

new_df <- cbind(new_df, temp)


summary(new_df[new_df$pi_24>=15,][,c("pi","ws","wsf","bp","t2","t10","tt","rh","ah")])


summary(new_df[new_df$p_24<15,][,c("pi","ws","wsf","bp","t2","t10","tt","rh","ah")])

# Normalizing variables


new_df[, c("wd","ws","wsf","bp","t2","t10","tt","rh","ah","pi")] <- scale(new_df[, c("wd","ws","wsf","bp","t2","t10","tt","rh","ah","pi")])




# Also adding lagged variables

for (i in 1:48) {
  lagged_temp <- lag(new_df[,c("wd","ws","wsf","bp","t2","t10","tt","rh","ah","pi")],i)
  colnames(lagged_temp) <- paste0(c("wd","ws","wsf","bp","t2","t10","tt","rh","ah","pi"),i)

  new_df <- cbind(new_df, lagged_temp)
}

new_df <- drop_na(new_df)

new_df$pi_cat <- ifelse(new_df$pi_24 >15, 1, 0)

new_df$pi_cat <- as.factor(new_df$pi_cat)

# save df to csv
write.csv(new_df, "../data/new_df2.csv")
