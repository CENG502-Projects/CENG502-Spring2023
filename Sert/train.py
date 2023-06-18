import uuid
import json
import click
import torch
import numpy as np
import pandas as pd
import torch.optim as optim

from torch import nn
from vins import VINS
from tqdm import tqdm
from model import MatrixFactorization
from torch.utils.data import DataLoader
from dataset import UserBusinessDataset
from sklearn.preprocessing import OrdinalEncoder


@click.command()
@click.option('--embedding_size', default=128)
@click.option('--beta', default=0.9)
@click.option('--margin', default=1)
@click.option('--batch_size', default=256)
@click.option('--max_step', default=64)
@click.option('--epochs', default=150)
@click.option('--use_vins', default=1)
def main(embedding_size, beta, margin, batch_size, max_step, epochs, use_vins):
    train = pd.read_parquet('data/train.parquet')
    test = pd.read_parquet('data/test.parquet')
    test = test[test.business_id.isin(train.business_id.unique()) & 
                test.user_id.isin(train.user_id.unique())]
    print(f'#Train: {len(train)}, #Test: {len(test)}')

    user_encoder = OrdinalEncoder()
    business_encoder = OrdinalEncoder()
    train['user_id'] = user_encoder.fit_transform(train['user_id'].values.reshape(-1, 1))[:, 0].astype(int)
    train['business_id'] = business_encoder.fit_transform(train['business_id'].values.reshape(-1, 1))[:, 0].astype(int)

    test['user_id'] = user_encoder.transform(test['user_id'].values.reshape(-1, 1))[:, 0].astype(int)
    test['business_id'] = business_encoder.transform(test['business_id'].values.reshape(-1, 1))[:, 0].astype(int)


    vins = VINS(train, max_step, margin, beta)
    model = MatrixFactorization(
        n_users=train['user_id'].max() + 1,
        n_items=train['business_id'].max() + 1,
        n_factors=embedding_size
    )


    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.LogSigmoid()
    best_f1 = 0

    losses = []
    f1s = []
    ndcgs = []

    run_id = str(uuid.uuid4())
    with open(f'checkpoints/{run_id}.details', 'w') as f:
        f.write(json.dumps({
            "Embedding Size": embedding_size, 
            "Beta": beta, 
            "Margin": margin, 
            "Batch Size": batch_size, 
            "Max Step": max_step, 
            "Epochs": epochs, 
            "Use VINS": use_vins
        }))


    for epoch in range(epochs):

        ####### VINS Implementation #######
        
        df_epoch = vins.generate_dataset(model, use_vins=use_vins) # this line

        ####### VINS Implementation #######

        df_epoch.to_parquet(f'{run_id}_{epoch}.parquet')
        avg_iv = df_epoch['iv'].mean()
        max_iv = df_epoch['iv'].max()
        dataloader = DataLoader(UserBusinessDataset(df_epoch), batch_size=batch_size, shuffle=True)

        model.train()

        losses = []
        for user, positive, negative, weight in dataloader:
            optimizer.zero_grad()
            p = model(user, positive)
            n = model(user, negative)

            loss = -(weight * criterion(p - n)).sum()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        
        f1, ndcg = model.evaluate(test['user_id'].values,
                        test['business_id'].values)
        print(f1, ndcg)
        

        f1_str = '%.2f' % (100*f1)
        ndcg_str = '%.2f' % (100 * ndcg)
        epoch_str = '%.3d' % epoch
        loss_str = '%.4f' % np.array(losses).mean()
        max_iv_str = '%.2f' % max_iv
        avg_iv_str = '%.2f' % avg_iv

        if f1 > best_f1:
            beta_str = '%.1f' % beta
            bbb = beta_str.replace('.', '_')
            fff = f1_str.replace('.', '_')
            torch.save(model, f'checkpoints/f1_{fff}_{run_id}_epoch_{epoch}.pt')
            losses.append(loss)
            ndcgs.append(ndcg)
            f1s.append(f1)


        print(f'[Epoch {epoch_str}] Loss={loss_str} | F1@10={f1_str}% | NDCG@10={ndcg_str} | avg IV={avg_iv_str} | max IV={max_iv_str}')
        torch.save(losses, f'checkpoints/{run_id}.loss')
        torch.save(ndcgs, f'checkpoints/{run_id}.ndcg')
        torch.save(f1s, f'checkpoints/{run_id}.f1')

if __name__ == '__main__':
    

    main()