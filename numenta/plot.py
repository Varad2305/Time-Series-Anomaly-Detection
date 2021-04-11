import sys
import matplotlib.pyplot as plt
import pandas as pd

file = sys.argv[1]

df = pd.read_csv(file)
values = df['value']
plt.plot(values)
plt.xlabel('Timestamp',fontsize=18)
plt.ylabel('Value',fontsize=18)
# plt.savefig('1',format='eps')
plt.grid(True)
plt.show()