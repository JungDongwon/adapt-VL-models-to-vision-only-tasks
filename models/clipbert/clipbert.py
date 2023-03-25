from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional, Union, List, Tuple, Dict, Any
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertEmbeddings

class BertImageEmbeddings(BertEmbeddings):
    """
    Patched version of BertEmbeddings where no positional nor token_type embeddings are added where token_type_ids is -1
    """
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(torch.maximum(token_type_ids, torch.tensor(0, dtype=torch.long)))
        token_type_embeddings[token_type_ids == -1] = 0

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            position_embeddings = position_embeddings.repeat(token_type_ids.shape[0], 1, 1)
            position_embeddings[token_type_ids == -1] = 0
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertImageModel(BertModel):
    """
    Extends BertModel class to add image features to input (along with a projection layer to match transformer dim)
    It does so by using the `inputs_embeds` argument to the `forward` method
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.embeddings = BertImageEmbeddings(config)
        self.img_projection = torch.nn.Linear(config.img_feature_dim, self.config.hidden_size, bias=True)
        #logger.info('BertImgModel Image Dimension: {}'.format(config.img_feature_dim))

    def forward(self, 
                input_ids,              # [batch, seq_len]
                img_feats=None,         # [batch, num_img_features, img_feature_dim]
                attention_mask=None,    # [batch, seq_len]
                token_type_ids=None,    # [batch, seq_len]
                inputs_embeds=None,
                **kwargs):

        device = input_ids.device
        inputs_embeds = self.embeddings.word_embeddings(input_ids)  # [batch, seq_len, hidden_size]

        # Image features
        if img_feats is not None:

            # Patch token_type_ids by adding -1 columns for image features
            if token_type_ids is None:
                token_type_ids = torch.zeros(inputs_embeds.size()[:-1], dtype=torch.long, device=device)
            minus_ones = -torch.ones((token_type_ids.shape[0], img_feats.shape[1]), 
                                        dtype=token_type_ids.dtype,
                                        device=device)
            token_type_ids = torch.cat((token_type_ids, minus_ones), dim=1)

            # Patch attention_mask
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=device)
            attention_mask = torch.cat((attention_mask, torch.ones(attention_mask.shape[0], 1, dtype=torch.long, device=device)), dim=1)

            proj_img_feats = self.img_projection(img_feats)
            inputs_embeds = torch.cat((inputs_embeds, proj_img_feats), dim=1)


        return super().forward(input_ids=None,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               inputs_embeds=inputs_embeds,
                               **kwargs)

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

"""
def get_clipbert_batch(batch, visual_feats=None, use_imagined_visual_feats=False):
    # align the visual inputs
    if use_imagined_visual_feats:
        assert "img_feats" in batch, "With 'use_imagined_visual_feats'=True, visual features should already be present in batch"
        return batch
    elif visual_feats is not None:
        batch_size = len(batch)
        img_feats = visual_feats.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1)
        batch.update(
            {
                "img_feats": img_feats
            }
        )

    return batch
"""

class ClipBertForImageClassification(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        config.img_feature_dim = CLIP_EMBED_DIM
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertImageModel(config, add_pooling_layer=False)
        self.pooler = BertPooler(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        img_feats: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            img_feats=img_feats,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # Take only text token logits
        #sequence_output = sequence_output[:, :input_ids.shape[1], :].contiguous()
        sequence_output = sequence_output[:, input_ids.shape[1]: , :].contiguous()

        # We do our own pooling to avoid pooling over image features
        pooled_output = self.pooler(sequence_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[3] + outputs[5]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )