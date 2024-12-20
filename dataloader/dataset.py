import torch.utils.data

def custom_collate_fn(batch):
    v_f_seq, t_f, a_f, label, item_id = zip(*batch)
    v_f_seq = torch.tensor(v_f_seq, dtype=torch.float)
    t_f = torch.tensor(t_f, dtype=torch.float)
    a_f = torch.tensor(a_f, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.float)
    return v_f_seq, t_f, a_f, label, item_id


class MyData(torch.utils.data.Dataset):

    def __init__(self, dataframe):
        super().__init__()
        self.t_f_list = dataframe['textual_feature_embedding'].tolist()
        self.v_f_list = dataframe['visual_feature_embedding_cls'].tolist()
        self.a_f_list = dataframe['audio_feature_embedding'].tolist()
        self.label_list = dataframe['label'].tolist()
        self.id_list = dataframe['item_id'].tolist()

    def __getitem__(self, index):
        v_f_seq = self.v_f_list[index]
        t_f = self.t_f_list[index]
        a_f = self.a_f_list[index]
        label = self.label_list[index]
        item_id = self.id_list[index]
        return v_f_seq, t_f, a_f, label, item_id

    def __len__(self):
        return len(self.t_f_list)
