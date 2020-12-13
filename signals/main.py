import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, irfft

from math import cos, sin, pi
from random import random

signals = 10
domain = np.linspace(0, 10 * pi, 1000)

f = lambda x: 1 if sin(x) == 0 else sin(signals * x) / sin(x) * cos(x) / signals

outputs = np.array([f(x) for x in domain])

transformed = rfft(outputs)

fig, axs = plt.subplots(4, 2)
axs[0][0].plot(domain, outputs)
axs[1][0].plot(transformed.imag)
axs[2][0].plot(transformed.real)
axs[3][0].plot(domain, irfft(transformed))

noisy = [x if random() > .3 else 4*(random()-.5) for x in outputs]

transformed = rfft(noisy)

axs[0][1].plot(domain, noisy)

cutoff_frequency = 45
threshold = 40

transformed = np.array([(x.imag*1j + x.real if x.real > threshold else 0) for x in transformed])
transformed = np.array([transformed[n] if n < cutoff_frequency else 0 for n in range(len(transformed))])


axs[1][1].plot(transformed.imag)
axs[2][1].plot(transformed.real)

axs[3][1].plot(domain, irfft(transformed))

plt.show()
