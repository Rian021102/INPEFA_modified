import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyNPEFAmain import inpefa
import lasio
import tempfile

y=lasio.read('/Users/rianrachmanto/pypro/project/PyNPEFA/data/1051308423.las').df().GR.dropna()
x = np.array(y.index)
inpefa_log = inpefa(y, x)


plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

plt.subplot(151)
plt.plot(inpefa_log['OG'],-x) # Original signal
plt.grid(True)
plt.xlabel('GR (API)')
plt.ylabel('Depth (ft)')
plt.xlim((0,150))
plt.title('Original GR Curve')

plt.subplot(152)
plt.plot(inpefa_log['1'],-x) # Long term INPEFA
plt.grid(True)
plt.xlim((-1,1))
plt.title('Long Term INPEFA')

plt.subplot(153)
plt.plot(inpefa_log['2'],-x) # Mid term INPEFA
plt.grid(True)
plt.xlim((-1,1))
plt.title('Mid Term INPEFA')

plt.subplot(154)
plt.plot(inpefa_log['3'],-x) # Short term INPEFA
plt.grid(True)
plt.xlim((-1,1))
plt.title('Short Term INPEFA')

plt.subplot(155)
plt.plot(inpefa_log['4'],-x) # Shorter term INPEFA
plt.grid(True)
plt.xlim((-1,1))
plt.title('Shorter Term INPEFA')

plt.show()