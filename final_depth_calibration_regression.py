import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load your collected data
data = np.load("depth_pairs.npy")
X = data[:, 0].reshape(-1, 1)  # DA values (inverted normalized)
y = data[:, 1]                 # Sensor depth in meters


lin_reg = LinearRegression()
lin_reg.fit(X, y)
a = lin_reg.coef_[0]
b = lin_reg.intercept_
r2 = lin_reg.score(X, y)
print(f"Linear fit: Depth = {a:.3f} * DA + {b:.3f}")
print(f"R^2 = {r2:.3f}")


plt.scatter(X, y, s=10, label="Data")
plt.plot(X, lin_reg.predict(X), color='red', label="Linear Fit")
plt.xlabel("Depth Anything (inverted normalized)")
plt.ylabel("Sensor Depth (m)")
plt.legend()
plt.grid(True)
plt.title("Depth Calibration")


textstr = f"$Depth = {a:.3f} \\times DA + {b:.3f}$\n$R^2 = {r2:.3f}$"
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.show()
