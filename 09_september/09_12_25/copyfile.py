import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pie chart data
countries = ["USA", "Canada", "Mexico"]
values = [45, 30, 25]

# Line graph data
time_points = [1, 2, 3, 4, 5]
counts = [5, 12, 18, 24, 33]

# Bar chart data
categories = ["A", "B", "C", "D"]
bar_values = [10, 23, 17, 5]

# Scatter plot data
x_scatter = np.linspace(-5, 5, 20)
y_scatter = x_scatter ** 2 + np.random.normal(scale=3, size=len(x_scatter))

# Contour plot data
x_contour = np.linspace(-3, 3, 100)
y_contour = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_contour, y_contour)
Z = np.sin(X) ** 2 + np.cos(Y) ** 2

plt.figure(figsize=(6,6))
plt.pie(values, labels=countries, autopct="%1.1f%%", startangle=90)
plt.title("Pie Chart: Countries")
plt.savefig(os.path.join(BASE_DIR, "plots/pie_chart.png"), dpi=300)
plt.close()

plt.figure(figsize=(7,5))
plt.plot(time_points, counts, marker="o", linestyle="-", color="b")
plt.title("Line Graph: Counts Over Time")
plt.xlabel("Time")
plt.ylabel("Counts")
plt.grid(True)
plt.savefig(os.path.join(BASE_DIR, "plots/line_graph.png"), dpi=300)
plt.close()

plt.figure(figsize=(7,5))
sns.barplot(x=categories, y=bar_values, palette="viridis")
plt.title("Bar Chart: Category Values")
plt.xlabel("Category")
plt.ylabel("Value")
plt.savefig(os.path.join(BASE_DIR, "plots/bar_chart.png"), dpi=300)
plt.close()

plt.figure(figsize=(7,5))
plt.scatter(x_scatter, y_scatter, c="red", edgecolor="black")
plt.title("Scatter Plot: Quadratic Trend")
plt.xlabel("X")
plt.ylabel("Y = X^2 (with noise)")
plt.savefig(os.path.join(BASE_DIR, "plots/scatter_plot.png"), dpi=300)
plt.close()

plt.figure(figsize=(7,6))
contour = plt.contour(X, Y, Z, cmap="plasma")
plt.colorbar(contour, label="Z value")
plt.title("Contour Plot: sin^2(X) + cos^2(Y)")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig(os.path.join(BASE_DIR, "plots/countour_plot.png"), dpi=300)
plt.close()

print("All plots saved")