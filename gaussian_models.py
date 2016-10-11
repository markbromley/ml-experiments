import numpy as np
import matplotlib.pyplot as plt

data1 = np.random.rand(200)
data2 = np.random.rand(200)

colors = ['red', 'green']

#area = np.pi * (15 * np.random.rand(200))**2  # 0 to 15 point radiuses

plt.scatter(data1, data2, c=colors, alpha=0.5)
plt.show()