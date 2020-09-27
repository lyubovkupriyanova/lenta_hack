import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#group to predict
category_1 = 'NONFOOD d645920e395fedad7bbbed0eca3fe2e0 bd3ef5c19067fe179f71c8b86ea4b39b 5bf563e8f99ed820f322704e4898df6b'

clients = pd.read_csv('clients.csv')
materials = pd.read_csv('materials.csv')
plants = pd.read_csv('plants.csv')
transaction = pd.read_parquet('transactions.parquet')

materials_merged = materials.copy()
materials_merged['merged'] = materials_merged['hier_level_1'] + ' ' + materials_merged['hier_level_2'] + \
                             ' ' + materials_merged['hier_level_3'] + ' ' + materials_merged['hier_level_4']
materials_merged = materials_merged.drop(['hier_level_1','hier_level_2','hier_level_3','hier_level_4'], axis = 1)

X = clients.copy()
tmp = transaction[['chq_id', 'client_id', 'sales_sum']]
tmp = tmp.groupby(['client_id','chq_id']).sum()

clients_to_use = []
prev = None
prev_ind = []
save_ind = []
count = 0
number = 0
for itr, i in enumerate(tmp.index):
    if i[0] == prev:
        count += 1
        prev_ind.append(itr)
    else:
        #print(prev,': ',count)
        if count >= 6:
            number += 1
            clients_to_use.append(prev)
            save_ind += prev_ind
        prev = i[0]
        count = 0
        prev_ind = [itr]
        
tmp = tmp.iloc[save_ind,:]
tmp = tmp.reset_index().drop(['chq_id'], axis = 1).groupby('client_id').mean()
tmp.columns = ['avg_chq']

X = X.set_index('client_id')
X = pd.concat([X,tmp], join = 'inner', axis = 1)

tmp = transaction[['client_id', 'material']]
tmp2 = tmp.loc[tmp['client_id'].isin(X.index), :]
tmp = tmp2.merge(materials_merged, on='material', how='left')
tmp = tmp.groupby(['client_id','merged']).count()
tmp = tmp.drop(['vendor', 'is_private_label', 'is_alco'], axis=1)
tmp.columns = ['count']
tmp_count_sum = tmp.groupby('client_id').sum()
tmp_count_sum.columns = ['count_sum']
tmp = tmp.reset_index()
tmp = tmp.merge(tmp_count_sum, on='client_id', how='inner')
tmp['rel_count'] = tmp['count']/tmp['count_sum']
tmp = tmp.drop(['count', 'count_sum'], axis = 1)
tmp2 = tmp.groupby('client_id').apply(lambda x: x.sort_values('rel_count', ascending = False))
tmp = tmp2.drop('client_id', axis=1).reset_index().drop(['level_1'],
                                                        axis=1).groupby('client_id').head(3)

tmp3 = tmp.groupby('client_id').count()
tmp3.reset_index(inplace=True)
tmp3.drop(np.where(tmp3['merged'] < 3)[0], inplace=True)
tmp2 = tmp.merge(tmp3, on = 'client_id', how = 'inner')
tmp2.drop(['merged_y', 'rel_count_y'], axis = 1, inplace = True)
tmp2.columns = ['client_id', 'merged', 'rel_count']
tmp2 = np.array(tmp2.set_index('client_id'))
a,b = tmp2.shape
indices = tmp3['client_id']
indices.drop_duplicates(inplace = True)
tmp2 = pd.DataFrame(tmp2.reshape((a//3, 6)), index = indices)
tmp2.columns = ['top_1_group', 'top_1_value', 'top_2_group', 'top_2_value', 'top_3_group', 'top_3_value']
X = pd.concat([X, tmp2], join = 'inner', axis = 1)

trans = transaction[['client_id', 'material', 'sales_count', 'sales_sum', 'is_promo']]
trans_categ = trans.merge(materials_merged, on='material', how='left')
trans_categ = trans_categ[trans_categ.merged == category_1]
trans_categ = trans_categ.groupby(['client_id', 'material']).count()
trans_categ.reset_index(inplace=True)
trans_max = trans_categ.groupby('client_id').max()
trans_top = trans_categ.groupby('client_id').apply(lambda x: x.sort_values('sales_count',ascending = False))
tmp = trans_top.drop('client_id', axis = 1).reset_index().drop(['level_1', 'sales_sum', 'is_promo',
                                                                'vendor','is_private_label', 'is_alco', 'merged'], axis = 1)
tmp21 = tmp.groupby('client_id').count()
tmp21.reset_index(inplace = True)
tmp21 = tmp21[tmp21.material!=1]

tmp = tmp.merge(tmp21, on = 'client_id', how = 'inner').drop(['material_y','sales_count_y'], axis = 1)
tmp.columns = ['client_id', 'material', 'sales_count']
tmp4 = tmp.groupby('client_id').head(2)
tmp5 = np.array(tmp4)
tmp5 = tmp5.reshape((tmp5.shape[0]//2,6))
tmp4 = pd.DataFrame(tmp5)
tmp4.columns = ['client_id', 'top1', 'top1_sum', 'client_id_ext', 'target', 'target_sum']
tmp4.drop(['top1_sum', 'client_id_ext', 'target_sum'], axis = 1, inplace = True)
trans_max = tmp4.set_index('client_id').drop('target', axis = 1)
y = tmp4.set_index('client_id').drop('top1', axis = 1)

X = pd.concat([X,y], join = 'inner', axis = 1)
trans_max.columns = ['top_1']
X = pd.concat([X, trans_max], join = 'inner', axis = 1)
X = X.reset_index()
X = X[(X.top_1_group == category_1) | (X.top_2_group == category_1) | (X.top_3_group == category_1)]

tmp = transaction[['material', 'sales_count', 'sales_sum', 'is_promo']]
tmp = pd.concat([tmp,tmp['sales_sum']/tmp['sales_count']],axis = 1)
tmp.columns = list(tmp.columns[:-1])+['price']
tmp.drop('sales_sum', axis = 1, inplace = True)
tmp = tmp.groupby('material').mean()

tmp = tmp.reset_index()
tmp.columns = ['top_1'] + list(tmp.columns[1:])
X = X.merge(tmp, on = 'top_1', how = 'inner')

y = X['target']
X = X.drop('target', axis=1)

enc = LabelEncoder()
y = enc.fit_transform(y)

X['gender'] = np.where(X['gender'].isnull(), '', X['gender'])
X['birthyear'].fillna((X['birthyear'].mean()), inplace=True)

X.to_parquet('X.parquet')
np.save('y.npy', y)
