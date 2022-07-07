import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np
y1 = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1])
y2 = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

plt.subplot(2,1,1)
plt.plot(y1)
plt.title("y1")

plt.subplot(2,1,2)
plt.plot(y2)
plt.title("y2")

plt.show()