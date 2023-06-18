import numpy as np
from tqdm import tqdm

class VINS:
    def __init__(self, df, max_step, margin, beta=0.9) -> None:
        '''
            VINS, takes a dataset with positive user-item interactions
            and then samples a negative item for each user-item interaction
            in the dataset.
        '''
        self.df = df.copy()
        self.beta = beta
        self.max_step = max_step
        self.margin=margin
        self.edges = len(df)
        self.Z = float(df['business_deg'].apply(lambda x: x ** self.beta).sum())        

        vc = df['business_id'].value_counts()
        self.degrees = {rid: occ for rid, occ in zip(vc.index, vc.values)}
        self.pi_all = sum([d ** beta for d in self.degrees.values()])


        self.all_businesss = set(df['business_id'].unique())
        self.user_history = df[['user_id', 'business_id']].groupby('user_id').business_id.agg(set)


    def generate_dataset(self, model, use_vins=True):
        ''' generates dataset with negative examples given the model '''
        negatives = []
        weights = []
        ivs = []
        for user_id, positive_id in zip(self.df.user_id, self.df.business_id):
            negative_id, weight, iv = self.sample_negative(model, user_id, positive_id, use_vins=use_vins)

            ivs.append(iv)
            negatives.append(negative_id)
            weights.append(weight)
        
        new_df = self.df.copy()
        new_df['negative_id'] = negatives
        new_df['weight'] = weights
        new_df['iv'] = ivs

        
        return new_df.dropna()

    def sample_negative(self, model, user_id, positive_id, max_shot=5, use_vins=True):
        ''' samples a negative example given user - business interaction
            Implementation of VINS (Appendix C, Algorithm 1) 
        '''
        x_pos = self.score(model, user_id, positive_id)
        best_score = -1
        best_negative = None
        for k in range(self.max_step):
            negative_id = self.reject_sampler(user_id, positive_id, max_shot, use_vins=use_vins)
            if negative_id == None:
                return None, None, None
            
            if not use_vins:
                best_negative = negative_id
                break

            x_neg = self.score(model, user_id, negative_id)
            score = x_neg + self.margin - x_pos
            if score > best_score:
                best_score = score
                best_negative = negative_id

            if score > 0:
                best_negative = negative_id
                break
        
        r = np.ceil(self.Z / min(k + 1, self.max_step))
        
        numerator = 1 + 0.5 * (np.log2(r + 1) - 1)
        denominator = 1 + 0.5 * (np.log2(self.Z + 1) - 1)
        weight = numerator / denominator

        iv = None
        if best_negative in self.degrees:
            deg = self.degrees[best_negative]
            n = deg ** (1 - self.beta) * self.pi_all
            d = self.edges - deg
            iv = n / d

        return best_negative, weight, iv


    def reject_sampler(self, user_id, positive_id, max_shot, use_vins):
        ''' Implementation of Reject Sampler (Appendix C, Algorithm 2)'''
        candidate_set = list(self.all_businesss - self.user_history[user_id])
        for _ in range(max_shot):
            business_id = np.random.choice(candidate_set)
            if not use_vins:
                return business_id

            reject_ratio = min(1 - self.pi(business_id) / self.pi(positive_id), 1)
            if np.random.uniform() > reject_ratio:
                return business_id
        

    def pi(self, business_id):
        ''' this is the implementation of pi function in the paper. '''
        deg = self.degrees[business_id]
        return deg ** self.beta
    
    def score(self, model,  user_id, business_id):
        ''' call to the supplied model to get the score of user - item interaction '''
        return model.get_score(user_id, business_id)