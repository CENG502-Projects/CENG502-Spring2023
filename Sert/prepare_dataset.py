import numpy as np
import pandas as pd
from tqdm import tqdm

FRAC = 0.01
chunksize = 1000000
steps = int(np.ceil(6685902 / chunksize))
chunks = pd.read_csv('data/review.csv', chunksize=chunksize)
slices = []
for i, df in enumerate(tqdm(chunks, total=steps)):
    s = df[['user_id', 'business_id']]
    # s.to_parquet(f'data/chunk_{i}.parquet')
    slices.append(s)

df = pd.concat(slices)


users =list(df.user_id.unique())
allowed_users =  users[:int(FRAC * len(users))]

df = df[df.user_id.isin(allowed_users)]

user_deg = df.user_id.value_counts().to_frame().reset_index().rename(columns={'user_id': 'user_deg', 'index': 'user_id'})
business_deg = df.business_id.value_counts().to_frame().reset_index().rename(columns={'business_id': 'business_deg', 'index': 'business_id'})
mdf = df.merge(user_deg, how='left', on='user_id')\
        .merge(business_deg, how='left', on='business_id')

sub_mdf = mdf[(mdf.user_deg >= 10) & (mdf.business_deg >= 10)]
sub_mdf['user_id'] = sub_mdf['user_id']
sub_mdf['business_id'] = sub_mdf['business_id']



train = []

test = []
for user_id, particle in sub_mdf.groupby('user_id'):
    cutoff = int(len(particle) * 0.8)
    tr = particle.head(cutoff)
    te = particle.tail(len(particle) - cutoff)
    train.append(tr)
    test.append(te)

pd.concat(train).to_parquet('data/train.parquet', index=False)
pd.concat(test).to_parquet('data/test.parquet', index=False)