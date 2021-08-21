import torch
from torch.nn import MSELoss, CrossEntropyLoss
from tqdm import tqdm
from transformers import BertModel, AutoModelForSequenceClassification, BertPreTrainedModel
import random
import numpy as np
from transformers import AutoTokenizer
from src.settings import config


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.bert = AutoModelForSequenceClassification.from_pretrained(config.model_name, config.num_labels)
        self.cls_dropout = torch.nn.Dropout(0.3)  # dropout on CLS transformed token embedding
        self.ent_dropout = torch.nn.Dropout(0.3)  # dropout on average entity embedding

        self.classifier = torch.nn.Linear(config.hidden_size * 3, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, e1_mask=None, e2_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # for details, see https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        pooled_output = outputs[1]  # sequence of hidden-states at the output of the last layer of the model
        sequence_output = outputs[
            0]  # last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function.

        def extract_entity(sequence_output, e_mask):
            # print('extract_entity')
            extended_e_mask = e_mask.unsqueeze(1)
            extended_e_mask = torch.bmm(
                extended_e_mask.float(), sequence_output).squeeze(1)
            return extended_e_mask.float()

        e1_h = self.ent_dropout(extract_entity(sequence_output, e1_mask))
        e2_h = self.ent_dropout(extract_entity(sequence_output, e2_mask))
        context = self.cls_dropout(pooled_output)
        pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)

        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # print('loss ottt')
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def fit(self, epochs, dataloader_train, device, optimizer, scheduler):

        model = self
        seed_val = 17
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        training_stats = []

        for epoch in tqdm(range(1, epochs + 1)):

            model.train()

            loss_train_total = 0

            progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
            for batch in progress_bar:
                model.zero_grad()

                batch = tuple(b.to(device) for b in batch)

                # inputs = {'input_ids':      batch[0],
                #           'attention_mask': batch[1],
                #           'labels':         batch[2],
                #          }

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          # XLM and RoBERTa don't use segment_ids
                          # 'token_type_ids': batch[2],
                          'e1_mask': batch[2],
                          'e2_mask': batch[3],
                          'labels': batch[4]
                          }

                outputs = model(**inputs)

                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

            torch.save(model.state_dict(), f'./drive/MyDrive/data_sets/PERLEX/models/hoshvare_persian_{epoch}.model')

            tqdm.write(f'\nEpoch {epoch}')

            loss_train_avg = loss_train_total / len(dataloader_train)
            tqdm.write(f'Training loss: {loss_train_avg}')

    def evaluate(self, dataloader_val, device):
        model = self
        model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in dataloader_val:
            batch = tuple(b.to(device) for b in batch)

            # inputs = {'input_ids':      batch[0],
            #           'attention_mask': batch[1],
            #           'labels':         batch[2],
            #          }

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      # XLM and RoBERTa don't use segment_ids
                      # 'token_type_ids': batch[2],
                      'e1_mask': batch[2],
                      'e2_mask': batch[3],
                      'labels': batch[4],

                      }
            with torch.no_grad():
                outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(dataloader_val)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals


def r_bert_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    special_tokens_dict = {'additional_special_tokens': ['[E11]', '[E12]', '[E21]', '[E22]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer, special_tokens_dict, num_added_toks
