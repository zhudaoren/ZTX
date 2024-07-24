import os
import pandas as pd
import pdfplumber
import pickle
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class MagazineDataset(Dataset):
    def __init__(self, data_folder, tokenizer, max_length=512, cache_file='data/references.pkl'):
        self.data_folder = data_folder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_file = cache_file
        self.pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]

        # 如果缓存文件存在则加载，否则进行预处理并保存
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.texts = pickle.load(f)
            print(f'Loaded cached references from {self.cache_file}')
        else:
            self.texts = self._preprocess_pdfs()
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.texts, f)
            print(f'Saved preprocessed references to {self.cache_file}')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

    def _preprocess_pdfs(self):
        texts = []
        for pdf_file in tqdm(self.pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(self.data_folder, pdf_file)
            with pdfplumber.open(pdf_path) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text()
            texts.append(text)
        return texts


class TextClassifier(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', max_len=128, batch_size=8, num_epochs=6,
                 model_save_path='result',reference_num=23):
        super(TextClassifier, self).__init__()
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = model_save_path

        # 初始化tokenizer和模型
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        # 定义两个独立的DistilBERT模型，分别处理问题文本和参考文献文本
        self.bert_model_question = DistilBertModel.from_pretrained(self.model_name)
        self.bert_model_reference = DistilBertModel.from_pretrained(self.model_name)
        # 参考文献信息作为一个知识库的形式给到每个问题的输入
        # 参考文献的输入需要再过一个全连接成得到一个固定大小的输出然后再和问题的模型输出拼接
        self.reference_fc1=nn.Linear(self.bert_model_question.config.hidden_size,1)
        self.reference_fc2=nn.Linear(reference_num,self.bert_model_question.config.hidden_size)

        # 全连接层用于分类
        # self.fc = nn.Linear(self.bert_model_question.config.hidden_size * 2, 2)  # *2是因为拼接了两个输入
        self.fc = nn.Linear(self.bert_model_question.config.hidden_size, 2)  # *2是因为拼接了两个输入
        #直接简单的加和吧
        self.dropout = nn.Dropout(0.1)

        self.to(self.device)

    def count_parameters(self,model):
        print("model paramenters count: ",sum(p.numel() for p in model.parameters() if p.requires_grad))


    class CustomDataset(Dataset):
        def __init__(self, questions, references, labels, tokenizer, max_len):
            self.questions = questions
            self.references = references
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.questions)

        def __getitem__(self, idx):
            question = str(self.questions[idx])
            reference = str(self.references[idx])
            label = self.labels[idx]

            # 编码问题文本
            question_encoding = self.tokenizer.encode_plus(
                question,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            # 编码参考文献文本
            reference_encoding = self.tokenizer.encode_plus(
                reference,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            return {
                'question_input_ids': question_encoding['input_ids'].flatten(),
                'question_attention_mask': question_encoding['attention_mask'].flatten(),
                'reference_input_ids': reference_encoding['input_ids'].flatten(),
                'reference_attention_mask': reference_encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    def trainModel(self, train_questions, train_references, train_labels):
        train_dataset = self.CustomDataset(train_questions, train_references, train_labels, self.tokenizer,
                                           self.max_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        criterion = torch.nn.CrossEntropyLoss()
        #单独处理参考文献信息
        encoded_references = self.tokenizer(references, padding=True, truncation=True, max_length=self.max_len,
                                            return_tensors='pt')
        reference_input_ids = encoded_references['input_ids'].to(self.device)
        reference_attention_mask = encoded_references['attention_mask'].to(self.device)

        for epoch in range(self.num_epochs):
            self.train()
            train_loss = 0.0



            train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
            for batch in train_loader_tqdm:
                question_input_ids = batch['question_input_ids'].to(self.device)
                question_attention_mask = batch['question_attention_mask'].to(self.device)
                # reference_input_ids = batch['reference_input_ids'].to(self.device)
                # reference_attention_mask = batch['reference_attention_mask'].to(self.device)reference_input_ids = batch['reference_input_ids'].to(self.device)
                # reference_attention_mask = batch['reference_attention_mask'].to(self.device)

                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                # 使用两个独立的DistilBERT模型分别处理问题和参考文献
                question_outputs = self.bert_model_question(input_ids=question_input_ids,
                                                            attention_mask=question_attention_mask)
                reference_outputs = self.bert_model_reference(input_ids=reference_input_ids,
                                                              attention_mask=reference_attention_mask)

                reference_output_fc1=self.reference_fc1(reference_outputs.last_hidden_state[:,0])
                reference_output_fc2=self.reference_fc2(reference_output_fc1.T)

                # 获取池化后的输出
                question_pooled_output = question_outputs.last_hidden_state[:, 0]
                # reference_pooled_output = reference_outputs.last_hidden_state[:, 0]


                # 将两个模型的输出拼接起来
                # combined_pooled_output = torch.cat((question_pooled_output, reference_pooled_output), dim=1)
                combined_pooled_output=question_pooled_output+reference_output_fc2
                logits = self.fc(self.dropout(combined_pooled_output))

                loss = criterion(logits, labels)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()

                train_loader_tqdm.set_postfix({'loss': loss.item()})

            avg_train_loss = train_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {avg_train_loss:.4f}')

            # 定期保存模型参数
            save_path = os.path.join(self.model_save_path, f'epoch_{epoch + 1}.pt')
            torch.save(self.state_dict(), save_path)
            print(f'Model saved to {save_path}')

    def predict(self, questions, references,model_path):
        # 加载模型参数
        self.load_state_dict(torch.load(model_path))
        self.eval()

        encoded_questions = self.tokenizer(questions, padding=True, truncation=True, max_length=self.max_len,
                                           return_tensors='pt')
        encoded_references = self.tokenizer(references, padding=True, truncation=True, max_length=self.max_len,
                                            return_tensors='pt')

        question_input_ids = encoded_questions['input_ids'].to(self.device)
        question_attention_mask = encoded_questions['attention_mask'].to(self.device)
        reference_input_ids = encoded_references['input_ids'].to(self.device)
        reference_attention_mask = encoded_references['attention_mask'].to(self.device)

        with torch.no_grad():
            question_outputs = self.bert_model_question(input_ids=question_input_ids,
                                                        attention_mask=question_attention_mask)
            reference_outputs = self.bert_model_reference(input_ids=reference_input_ids,
                                                          attention_mask=reference_attention_mask)

            question_pooled_output = question_outputs.last_hidden_state[:, 0]
            # reference_pooled_output = reference_outputs.last_hidden_state[:, 0]
            reference_output_fc1 = self.reference_fc1(reference_outputs.last_hidden_state[:, 0])
            reference_output_fc2 = self.reference_fc2(reference_output_fc1.T)

            # combined_pooled_output = torch.cat((question_pooled_output, reference_pooled_output), dim=1)
            combined_pooled_output = question_pooled_output + reference_output_fc2

            logits = self.fc(combined_pooled_output)#在0维上凭借起来。

            preds = torch.argmax(logits, dim=-1)
            return preds.cpu().numpy()

    def save_predictions_to_csv(self, questions, references, result_df, output_file='result.csv',model_path='data/result/epoch_6.pt'):
        predictions = self.predict(questions, references,model_path)
        print('加载模型参数: ',model_path)
        # 更新result_df中的answer列
        result_df['answer'] = ['T' if pred == 1 else 'F' for pred in predictions]
        result_df.to_csv(output_file, index=False)
        print(f'Predictions saved to {output_file}')


# 示例用法
if __name__ == '__main__':
    # 数据路径和模型参数
    data_folder = 'data/参考文献'
    model_name = 'data/pred'#/distilbert-base-uncased
    max_len = 128
    batch_size = 8
    num_epochs = 2
    model_save_path = 'data/result'

    # 加载问题和参考文献数据
    questions_df = pd.read_csv('data/test_A.csv')
    result_df = pd.read_csv('data/result.csv')

    questions = questions_df['question'].tolist()
    references_dataset = MagazineDataset(data_folder, model_name)
    references = [ref for ref in references_dataset]

    # 只使用前4个样本进行训练
    train_questions = questions[:4]
    train_references = references[:4]
    train_labels = [1 if ans == 'T' else 0 for ans in result_df['answer'][:4]]

    # 初始化和训练模型
    classifier = TextClassifier(model_name=model_name, max_len=max_len, batch_size=batch_size, num_epochs=num_epochs,
                                model_save_path=model_save_path)
    #输出模型的参数量
    classifier.count_parameters(classifier)
    classifier.trainModel(train_questions, train_references, train_labels)

    # 保存预测结果到csv文件
    classifier.save_predictions_to_csv(questions, references, result_df, output_file='data/result/result.csv',model_path='data/result/epoch_'+str(num_epochs)+'.pt')
