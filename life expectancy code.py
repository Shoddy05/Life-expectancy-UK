# Imports needed libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#Makes and sets the size of the figure W,H in inches
plt.rcParams["figure.figsize"] = [15, 6]
# Automatically adjusts subplot params so that the subplot fits in to the figure area. 
plt.rcParams["figure.autolayout"] = True
#Uses pandas to read life_expectancy csv and sets it to df
df = pd.read_csv("C:/Users/carls/Downloads/Life_expectancy_in_the_UK.csv")
# Sets the collum with the "Males"/"Females" heading to male_col/female_col
male_col= df["Males"]
female_col= df["Females"]
# Sets the collum with the "Years" heading to year_col then converts to a NumPy array
year_col = df["Years"].to_numpy()
df.head()
#Changes the shape of the array 
year_col= year_col.reshape(-1,1)
#Array of future years
future_years = [2022,2024, 2026, 2028, 2030, 2032, 2034, 2036, 2038, 2040]
#----------------------------------------------------------------
# Male and female 1980 - 2022 graph
#----------------------------------------------------------------
#Sets the index to the "Years" collum and plots points on the graph
df.set_index('Years').plot()
#Sets title of graph
plt.title("Life expectancy at birth for males and females, UK, between 1980 to 1982 and 2020 to 2022")
#Labels y-axis
plt.ylabel("Age")
#Sets the tick locations and labels of the x-axis
plt.xticks(np.arange(1980, 2024, step=2))
#Shows the graph
plt.show()
#----------------------------------------------------------------
#Future graph Male 2022- 2040
#----------------------------------------------------------------
male_predictions =[]
#Makes Linear Regression model   
regression_model = LinearRegression()
#Uses model to predict future male_col values
regression_model.fit(year_col,male_col)
#Add prediction to the predictions array for every future year
for year in future_years:
    year = [[year]]
    male_predictions.append(regression_model.predict(year))
#Plots line lables it "Males"
plt.plot(future_years,male_predictions , label= "Males")
#----------------------------------------------------------------
#Future graph Female 2022- 2040
#----------------------------------------------------------------
female_predictions =[]
regression_model = LinearRegression()
regression_model.fit(year_col,female_col)
for year in future_years:
    year = [[year]]
    female_predictions.append(regression_model.predict(year))
plt.plot(future_years,female_predictions, label= "Females")
#----------------------------------------------------------------
plt.xticks(np.arange(2022, 2042, step=2))
plt.title("Predicted life expectancy at birth for males and females, UK, 2022 to 2040")
plt.ylabel("Age")
#Labels x-axis
plt.xlabel("Year")
#Makes legend show up in top left of graph
plt.legend(loc="upper left")
plt.show()