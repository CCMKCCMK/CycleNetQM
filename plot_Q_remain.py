import matplotlib.pyplot as plt
import pandas as pd

# load csv file
Q = pd.read_csv(r'test_results\Electricity_720_720_CycleNet_custom_ftM_sl720_pl720_cycle168_linear_seed1024\Q.csv')
total = pd.read_csv(r'dataset/electricity.csv')

# remove date column
total = total.drop(columns=['date'])

# Align Q with the original data
len_total = len(total)
len_Q = len(Q)
roll =  -720 % len_Q

# roll Q and match the original data length
Q = pd.concat([Q.iloc[-roll:], Q.iloc[:-roll]]).reset_index(drop=True)

# repeat Q to match the original data length
Q = pd.concat([Q] * (len_total // len_Q), ignore_index=True)

# Ensure Q has the same columns as total
Q.columns = total.columns
remaining = total - Q


Q = Q['OT']
total = total['OT']
remaining = remaining['OT']


x=500
y=1000
# plot Q and the original data
plt.plot(Q[x:y], label='Q')
plt.plot(total[x:y], label='OT')
plt.plot(remaining[x:y], label='Remaining')
plt.legend()
plt.show()

# save the remaining data
remaining.to_csv(r'test_results\Electricity_720_720_CycleNet_custom_ftM_sl720_pl720_cycle168_linear_seed1024\remaining.csv', index=False)
