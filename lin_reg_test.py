import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x_data = np.array([115, 117, 140, 115, 116, 127, 121, 121, 115, 103,
                114, 90, 24, 98, 110, 119, 121, 108, 141, 78, 93, 98, 88, 141, 84])
y_data = np.array([168, 164, 192, 136, 146, 142, 132, 199, 127, 131,
                157, 115, 0, 162, 168, 131, 112, 183, 119, 59, 176, 187, 93, 226, 154])

# create a Linear Regression model and fit it to the data
model = LinearRegression()
model.fit(x_data.reshape(-1, 1), y_data)

# predict y-values for the given x-values
y_pred = model.predict(x_data.reshape(-1, 1))

print('Regression equation: y = {:.3f}x + {:.3f}'.format(model.coef_[0], model.intercept_))

# plot the data points and the regression line
plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred, color='red')
plt.xlabel('x_data')
plt.ylabel('y_data')
plt.title('Linear Regression')
plt.show()
