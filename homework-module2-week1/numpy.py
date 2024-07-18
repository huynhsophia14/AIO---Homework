import numpy as np
import matplotlib.image as mpimg
import pandas as pd
#Question 12, 13, 14: Convert color picture to grey picture by different methods
#Download image
%pip install gdown
!gdown 1i9dqan21DjQoG5Q_VEvm0LrVwAlXD0vB
#Q12: Lightness
img = mpimg.imread('/workspaces/Homework-1/Module 2 - HW1/dog.jpeg')
gray_img_01 = np.zeros((img.shape[0], img.shape[1]))

for i in range(img.shape[0]):
  for j in range(img.shape[1]):
    r, g, b = img[i, j][0], img[i, j][1], img[i, j][2]
    gray_img_01[i, j] = (max(r, g, b) + min(r, g, b)) / 2

#get the grayscale value of the first pixel
gray_img_01[0, 0]
#Q13: Average
gray_img_02 = np.zeros((img.shape[0], img.shape[1]))

for i in range(img.shape[0]):
  for j in range(img.shape[1]):
    gray_img_02[i, j] = np.mean(img[i, j])

#get the grayscale value of the first pixel
gray_img_02[0, 0]
#Q14: Luminosity
gray_img_03 = np.zeros((img.shape[0], img.shape[1]))

for i in range(img.shape[0]):
  for j in range(img.shape[1]):
    r, g, b = img[i, j][0], img[i, j][1], img[i, j][2]
    gray_img_03[i, j] = 0.21*r + 0.72*g + 0.07*b

#get the grayscale value of the first pixel
gray_img_03[0, 0]
#Question 15 - 21: Work with data table
#Download data
!gdown 1iA0WmVfW88HyJvTBSQDI5vesf-pgKabq
df = pd.read_csv('/workspaces/Homework-1/Module 2 - HW1/advertising.csv')

#convert data frame to munpy array
data = df.to_numpy()
#Q15: Get the maximum value and its index on Sales column
# Find the maximum value in column 'Sales'
sales_max_value = df['Sales'].max()
# Find its index
sales_max_index = df['Sales'].idxmax()

sales_max_value, sales_max_index
#Q16: Average of column 'TV'
avg_TV = df['TV'].mean()
avg_TV
#Q17: count the number of records that have sale value >= 20
# Define condition: sale values >= 20 in column 'sales'
condition = df['Sales'] >= 20

# Count the number of values that satisfy the condition
count = condition.sum()
count
#Q18: average of all values in column 'Radio' that have the corresponding sale value >= 15
# Define condition
condition_2 = df['Sales'] >= 15

# Filter the DataFrame based on the condition and calculate the mean
avg_radio = df.loc[condition_2, 'Radio'].mean()
avg_radio
#Q19: sum of all values in column 'Sales' that have the corresponding Newspaper value greater than its average
# Define condition
condition_3 = df['Newspaper'] >= df['Newspaper'].mean()

# Filter the DataFrame based on the condition and calculate the sum
avg_radio = df.loc[condition_3, 'Sales'].sum()
avg_radio
#Q20:
A = df['Sales'].mean()

#Create the scores array
def classify_sales(sale, average):
    if sale > average:
        return 'Good'
    elif sale < average:
        return 'Bad'
    else:
        return 'Average'

scores = df['Sales'].apply(lambda x: classify_sales(x, A)).to_numpy()
scores[7:10]
#Q21:
#Find the nearest value to average sales value
def residual(sale, average):
        return abs(sale - average)

residual_arr = df['Sales'].apply(lambda x: residual(x, A)).to_numpy()
nearest_avg_value = df['Sales'][np.argmin(residual_arr)]

#Create the scores array
def classify_sales(sale, nearest_value):
    if sale > nearest_value:
        return 'Good'
    elif sale < nearest_value:
        return 'Bad'
    else:
        return 'Average'

scores = df['Sales'].apply(lambda x: classify_sales(x, nearest_avg_value)).to_numpy()
scores[7:10]
