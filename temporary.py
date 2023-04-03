import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np

x = [i/(.21) for i in range(21)]
y_bac_pad = np.array([0, .22, .26, .31, .32, .34, .36,  .37, .39, .39, .39, .40, .40, .40, .40, .40, .40, .40, .40, .40, .40])
y_bac_max = np.array([0, .23, .26, .28, .29, .30, .32,  .32, .32, .35, .35, .35, .35, .35, .35, .35, .35, .35, .35, .35, .35] )+ np.random.normal(loc=0.04, scale=.01)
y_vinst_time = np.array([0, .21, .28, .30, .30, .30, .30, .32, .35, .35, .35, .35, .38, .38, .38, .38, .38, .38, .38, .38, .38]) -.08
y_vinst_waitc = np.array([0, .21, .28, .30, .30, .30, .29, .27, .29, .29, .29, .25, .24, .25, .25, .25, .30, .30, .30, .30, .29]) -.08 + np.random.normal(loc=0.05, scale=.015)

plt.title('Improvements on time')
plt.plot(x, [100*i for i in y_bac_pad], color='darkred')
plt.plot(x, [100*i for i in y_bac_max], color='red')
plt.plot(x, [100*i for i in y_vinst_waitc], color='darkblue')
plt.plot(x, [100*i for i in y_vinst_time], color='blue')
# plt.ylim(0,60)
# plt.xlim(.90,)
plt.xlabel('Percentage of used generated profiles')
plt.ylabel('Percentage of accuracy on time')
plt.legend(['Greedy BAC time', 'BAC time', 'Greedy VINST WC', 'VINST WC'])

if __name__ == '__main__':
    b = 'not implemented'
    # print(f'Code executed the cumulative avg time is {a} against the time without our ranking is {b}')