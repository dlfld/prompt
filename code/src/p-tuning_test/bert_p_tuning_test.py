from typing import Dict

import torch


def generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
        使用随机初始化的soft prompt embedding 替换原来的embedding
    """
    input_ids = batch['input_ids']
    bz = batch['input_ids'].shape[0]
    block_flag = batch["block_flag"]
    model = self.model.module if hasattr(self.model, 'module') else self.model

    if self.config.model_type == "albert":
        raw_embeds = model.model.albert.embeddings.word_embeddings(input_ids)
    elif self.config.model_type == "bert":
        raw_embeds = model.model.bert.embeddings.word_embeddings(input_ids)
    elif self.config.model_type == "roberta":
        raw_embeds = model.model.roberta.embeddings.word_embeddings(input_ids)

    replace_embeds = model.prompt_embeddings(
        # 在这儿随机初始化的一个embeds，创新点是是否能够有效的初始化
        torch.LongTensor(list(range(model.prompt_length))).cuda()
    )

    replace_embeds = replace_embeds.unsqueeze(0)  # [batch_size, prompt_length, embed_size]

    if self.config.prompt_encoder_type == "lstm":
        replace_embeds = model.lstm_head(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
        if model.prompt_length == 1:
            replace_embeds = model.mlp_head(replace_embeds)
        else:
            replace_embeds = model.mlp_head(replace_embeds).squeeze()

    elif self.config.prompt_encoder_type == "mlp":
        replace_embeds = model.mlp(replace_embeds)
    else:
        raise ValueError("unknown prompt_encoder_type.")

    blocked_indices = (block_flag == 1).nonzero().reshape((bz, model.prompt_length, 2))[:, :, 1]

    for bidx in range(bz):
        for i in range(blocked_indices.shape[1]):
            raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

    inputs = {'inputs_embeds': raw_embeds, 'attention_mask': batch['attention_mask']}

    if self.config.model_type in ['bert']:
        inputs['token_type_ids'] = batch['token_type_ids']

    return inputs
