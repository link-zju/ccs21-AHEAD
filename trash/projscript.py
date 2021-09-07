# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import numpy as np
aa = np.load('./dataset/Two_Laplace01-Set_10_5-Domain_8_8-Data.npz')
# print(aa['dataset'])

with open('./dataset/Two_Laplace01-Set_10_5-Domain_8_8-Data.txt',"w") as f: 
    for items in aa['dataset']:
        print(items)
        # f.writelines((str(items)))
        f.write(str(items[0]) + ' ' + str(items[1])+ '\n')

# %%
import numpy as np
aa = np.load('./dataset/Normal_Skew00-Set_10_5-Domain_10-Data.npy')
# print(aa['dataset'])

with open('./dataset/Normal_Skew00-Set_10_5-Domain_10-Data.txt',"w") as f: 
    for items in aa:
        print(items)
        # f.writelines((str(items)))
        f.write(str(int(items)) + '\n')
# %%
import numpy as np
aa = np.load('./query_table/Rand_QueryTable_Domain10_Attribute1.npy')
print(aa[:10])

with open('./query_table/Rand_QueryTable_Domain10_Attribute1.txt',"w") as f: 
    for items in aa:
        print(items)
        # f.writelines((str(items)))
        for item in items:
            f.write(str(int(item)) + ' ')
        f.write('\n')
# %%

