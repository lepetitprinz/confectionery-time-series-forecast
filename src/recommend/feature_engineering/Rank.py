
class Rank(object):
    def __init__(self, item_indices: dict, similarity, top_n: int):
        self.item_col = 'sku_cd'
        self.meta_col = 'bom_cd'
        self.item_indices = item_indices
        self.item_indices_rev = {val: key for key, val in item_indices.items()}
        self.similarity = similarity
        self.top_n = top_n

    # Rank similar items for each item
    def ranking(self, item_cd: str):
        rank = self.get_rank(item_cd=item_cd)
        ranks = self.get_rank_item_score(rank=rank)

        return ranks

    def get_rank(self, item_cd: str):
        idx = self.item_indices[item_cd]
        sim_rank = list(enumerate(self.similarity[idx]))
        del sim_rank[idx]    # remove itself
        sim_rank = sorted(sim_rank, key=lambda x: x[1], reverse=True)

        # get scores of the top n most similar items
        sim_rank = sim_rank[:self.top_n]

        return sim_rank

    def get_rank_item_score(self, rank):
        result = [(self.item_indices_rev[sim[0]], round(sim[1], 3)) for sim in rank]

        return result
