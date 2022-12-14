import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
sns.lineplot(x = [(i/10)*150000 for i in range(11)], y = [0, 18, 29, 37, 46, 53, 60, 65, 69, 75, 79])
sns.lineplot(x = [(i/10)*150000 for i in range(11)], y = [0, 72, 79, 81, 85, 88, 89, 90, 92, 92, 92])
plt.title('Number of resources that can choose their activity \n after 10 simulations per type')
plt.legend(['Max','Ale'])
plt.ylabel('% of resources')
plt.xlabel('Number of generated solutions')
plt.ylim(0,100)



import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme()
z = np.random.geometric(p=0.11, size=100000)
sns.histplot(z)
plt.title('Geometric distribution of idx_repl_res')
plt.xlim(0,80)

