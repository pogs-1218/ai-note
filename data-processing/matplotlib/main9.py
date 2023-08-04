import matplotlib.pyplot as plt

days = [1, 2, 3, 4, 5, 6, 7]
max_t = [50, 51, 52, 48, 47, 49, 46]
min_t = [43, 42, 40, 44, 33, 35, 37]
avg_t = [45, 48, 48, 46, 40, 42, 41]

# plt.plot(x, y, 'bo', alpha=0.2)
plt.plot(days, max_t, label='max')
plt.plot(days, min_t, label='min')
plt.plot(days, avg_t, label='avg')

plt.grid()
plt.title('Weather')
plt.xlabel('Days')
plt.ylabel('Temperature')
plt.legend(loc='best', shadow=True)
plt.show()