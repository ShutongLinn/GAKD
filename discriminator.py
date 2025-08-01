import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from torch.nn.utils.parametrizations import spectral_norm

import numpy as np


class Discriminator(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Discriminator, self).__init__()
        self.input_proj = nn.Linear(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, 1)
        
        self.to(torch.bfloat16)
        
    def forward(self, logits):
        if logits.dtype != torch.bfloat16:
            logits = logits.to(torch.bfloat16)
            
        x = torch.mean(logits, dim=1, dtype=torch.bfloat16)
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = F.relu(layer(x))
            
        return torch.sigmoid(self.output(x))
    
class SimpleMLPDiscriminator(nn.Module):
    """
    简单的3层MLP判别器，替代原来的LLM架构
    保持原有的单任务设计（只做Real/Fake判别）
    """
    def __init__(self, vocab_size, hidden_dim=512, seq_len=None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # 序列处理：将 [B, L, V] 转换为 [B, V]
        # 使用注意力池化机制
        self.attention_pool = nn.Linear(vocab_size, 1)
        
        # 输入批量归一化
        self.input_bn = nn.BatchNorm1d(vocab_size)
        
        # 3层MLP主体网络
        self.mlp = nn.Sequential(
            # 第一层
            nn.Linear(vocab_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 第二层
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 第三层
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 输出层（保持原有的单任务设计）
        self.output_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, input_logits, attention_mask=None):
        """
        前向传播
        Args:
            input_logits: [batch_size, seq_len, vocab_size] 或 [batch_size, vocab_size]
            attention_mask: 可选的注意力掩码
        Returns:
            output: [batch_size, 1] - Real/Fake分数（无sigmoid，适合WGAN）
        """
        # 处理输入维度
        input_logits = input_logits.to(torch.bfloat16)
        if input_logits.dim() == 3:
            # 输入是序列: [B, L, V] → [B, V]
            batch_size, seq_len, vocab_size = input_logits.shape
            
            # 使用注意力池化
            attn_scores = self.attention_pool(input_logits)  # [B, L, 1]
            attn_weights = F.softmax(attn_scores, dim=1)     # [B, L, 1]
            
            # 如果有attention_mask，应用掩码
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
                attn_weights = attn_weights * mask
                attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)
            
            # 加权平均池化
            pooled = torch.sum(input_logits * attn_weights, dim=1)  # [B, V]
            
        elif input_logits.dim() == 2:
            # 输入已经是向量: [B, V]
            pooled = input_logits
        else:
            raise ValueError(f"Expected input_logits to be 2D or 3D, got {input_logits.dim()}D")
        
        # 批量归一化
        x = self.input_bn(pooled)
        
        # 通过3层MLP
        features = self.mlp(x)
        
        # 输出Real/Fake分数
        output = self.output_head(features)
        
        return output


class MLPEmbeddingLLMDiscriminator(nn.Module):
    def __init__(self, model_name):
        """
        :param new_hidden_dim: 新 embedding 层输出的维度 D，必须与预训练模型的 hidden_size 相同
        :param model_name: 使用的预训练模型名称（例如 "gpt2"）
        """
        super().__init__()

        
        self.config = AutoConfig.from_pretrained(model_name)
        vocab_size = self.config.vocab_size  
        

        new_hidden_dim  = self.config.hidden_size
           

    
        self.model = AutoModelForCausalLM.from_pretrained(model_name, config=self.config)
        
     
        self.new_embedding = nn.Linear(vocab_size, new_hidden_dim, bias=False)

        self.pred_head = nn.Sequential(
            nn.Linear(new_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input_one_hot, attention_mask=None):
        """
        :param input_one_hot: 输入 one-hot 编码张量，形状为 (B, L, V)
        :param attention_mask: 注意力 mask（可选）
        :return: 返回最后一个 token 的 hidden embedding，形状为 (B, new_hidden_dim)
        """
        # 将 one-hot 输入通过线性层映射到 (B, L, new_hidden_dim)
        input_one_hot = input_one_hot.to(torch.bfloat16)
        inputs_embeds = self.new_embedding(input_one_hot)
        
        # 将映射后的 embeddings 传递给预训练模型，
        # 注意这里用的是 inputs_embeds 参数，从而跳过了模型内部的 token embedding
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,output_hidden_states=True)
        
        # 获取模型最后一层的 hidden state，形状为 (B, L, new_hidden_dim)
        last_hidden_state = outputs.hidden_states[-1]
        
        # 取出每个序列中最后一个 token 的 hidden embedding
        final_token_embedding = last_hidden_state[:, -1, :]
        output = self.pred_head(final_token_embedding)
        return output

class WGANLLMDiscriminator(nn.Module):
    def __init__(self, model_name):
        """
        :param new_hidden_dim: 新 embedding 层输出的维度 D，必须与预训练模型的 hidden_size 相同
        :param model_name: 使用的预训练模型名称（例如 "gpt2"）
        """
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        vocab_size = self.config.vocab_size  

        new_hidden_dim  = self.config.hidden_size
    
        self.model = AutoModelForCausalLM.from_pretrained(model_name, config=self.config)
        
        self.new_embedding = nn.Linear(vocab_size, new_hidden_dim, bias=False)

        self.pred_head = nn.Sequential(
            nn.Linear(new_hidden_dim, 1)
        )

    def forward(self, input_one_hot, attention_mask=None):
        """
        :param input_one_hot: 输入 one-hot 编码张量，形状为 (B, L, V)
        :param attention_mask: 注意力 mask（可选）
        :return: 返回最后一个 token 的 hidden embedding，形状为 (B, new_hidden_dim)
        """
        # 将 one-hot 输入通过线性层映射到 (B, L, new_hidden_dim)
        input_one_hot = input_one_hot.to(torch.bfloat16)
        inputs_embeds = self.new_embedding(input_one_hot)
        
        # 将映射后的 embeddings 传递给预训练模型，
        # 注意这里用的是 inputs_embeds 参数，从而跳过了模型内部的 token embedding
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,output_hidden_states=True)
        
        # 获取模型最后一层的 hidden state，形状为 (B, L, new_hidden_dim)
        last_hidden_state = outputs.hidden_states[-1]
        
        # 取出每个序列中最后一个 token 的 hidden embedding
        final_token_embedding = last_hidden_state[:, -1, :]
        output = self.pred_head(final_token_embedding)
        return output