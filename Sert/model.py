import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):

    def __init__(self, n_users, n_items, n_factors=20):
        ''' this is a simple matrix factorization model using embeddings. '''
        super().__init__()
        print('Num users:', n_users)
        print('Num items:', n_items)
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.business_factors = nn.Embedding(n_items, n_factors)

        nn.init.xavier_uniform_(self.user_factors.weight)
        nn.init.xavier_uniform_(self.business_factors.weight)

    def forward(self, user, business):
        return (self.user_factors(user) * self.business_factors(business)).sum(1)
    
    def get_score(self, user_id, business_id):
        ''' this function is utilized by VINS to get the score
            of supplied user and item (business) id.
        '''
        with torch.no_grad():
            user_id = torch.LongTensor([user_id])
            business_id = torch.LongTensor([business_id])

            return self.forward(user_id, business_id).item()
            
    def recommend_businesses(self, user, k=10):
        ''' this function recommends top 10 items to users '''
        with torch.no_grad():
            user_embedding = self.user_factors(user)
            item_embeddings = self.business_factors.weight
            scores = torch.mm(user_embedding, item_embeddings.t())
            _, top_k_items = torch.topk(scores, k, dim=1)
            return top_k_items
        
    def evaluate(self, user_ids, bus_ids, k=10):
        ''' function to calculate F1@10 and NDCG@10 on test set. '''
        user_ids = torch.LongTensor(user_ids)
        bus_ids = torch.LongTensor(bus_ids)
        recommendations = self.recommend_businesses(user_ids)

        mask = recommendations == bus_ids.view(-1, 1)
        relevant = mask.any(1).float()
        precision = relevant.mean().item()
        recall = (relevant.sum() / bus_ids.size(0)).item()
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        gains = mask.float()
        discounts = torch.log2(torch.arange(2, k + 2))
        dcgs = (gains / discounts).sum(1)
        idcgs = (torch.ones(gains.size(0)) / discounts[0]).sum().unsqueeze(0)  # ideal DCG
        ndcg = (dcgs / idcgs).mean().item()
        return f1, ndcg




