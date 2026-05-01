import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import time
import os

# ==========================================
# 1. 真・因果的FFTレゾナンス (God-Speed Causal FFT)
# ==========================================
class GodSpeedCausalFFT(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # 過去方向にのみ減衰するカーネル（これが因果律の正体）
        # 指数的に減衰する重みを学習可能なパラメータとして持つ
        self.decay_kernel = nn.Parameter(torch.exp(-torch.linspace(0, 10, 128)))
        self.dim = dim

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = torch.chunk(self.qkv_proj(x), 3, dim=-1)
        
        # KとVを結合（情報の蓄積）
        # 未来の情報が混ざらないよう、ここをベースにする
        src = k * v 
        
        # 畳み込み定理を利用した高速計算: y = src * kernel
        # シーケンス長Lとカーネル長を考慮したパディング
        K_len = self.decay_kernel.size(0)
        N = L + K_len - 1
        
        # FFTによる周波数ドメインへの変換
        src_f = torch.fft.rfft(src, n=N, dim=1)
        
        # カーネルの準備（過去へのみ伸びるように配置）
        # チャンネルごとに異なる減衰を適用できるよう拡張
        kernel = self.decay_kernel.view(1, -1, 1).expand(-1, -1, D)
        kernel_f = torch.fft.rfft(kernel, n=N, dim=1)
        
        # 複素空間での乗算（これが時間軸での「因果的畳み込み」になる）
        res_f = src_f * kernel_f
        res = torch.fft.irfft(res_f, n=N, dim=1)
        
        # 必要な長さ（L）だけ取り出し、現在のQと作用させる
        # src_fに未来が入っていないため、resのどの地点も未来の影響を10ミリも受けない
        causal_res = res[:, :L, :]
        output = q * causal_res
        
        return self.out_proj(self.dropout(output))

# ==========================================
# 2. アーキテクチャ (ORE-Apex 本体)
# ==========================================
class ApexCell(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.resonance = GodSpeedCausalFFT(dim, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, dim * 4, bias=False)
        self.w2 = nn.Linear(dim, dim * 4, bias=False)
        self.w3 = nn.Linear(dim * 4, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 残差接続と再帰
        x = x + self.resonance(self.norm1(x))
        res = self.norm2(x)
        gate = F.silu(self.w1(res)) * self.w2(res)
        x = x + self.dropout(self.w3(gate))
        return x

class ORE_Apex(nn.Module):
    def __init__(self, vocab_size, dim=384, max_loops=12, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.cell = ApexCell(dim, dropout)
        self.max_loops = max_loops
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight 

    def forward(self, x):
        x = self.token_emb(x)
        for _ in range(self.max_loops):
            x = self.cell(x)
        return self.head(self.norm(x))

# ==========================================
# 3. 学習・保存・生成 (RTX 4060 最適化版)
# ==========================================
def run_ultimate_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Preparing Full Wikitext-2 Dataset...")
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    train_data = raw["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
    train_loader = DataLoader(train_data.with_format("torch"), batch_size=32, shuffle=True)

    # モデル構築
    model = ORE_Apex(tokenizer.vocab_size, dim=384, max_loops=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    print(f"--- ORE-Apex: Perfect Causal FFT Mode ---")
    
    for epoch in range(10):
        model.train()
        t_loss = 0
        start = time.time()
        
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # 次単語予測（ラベルを1つずらす）
            logits = model(ids[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size), 
                ids[:, 1:].reshape(-1), 
                label_smoothing=0.05
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            
        avg_loss = t_loss/len(train_loader)
        print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Time: {time.time()-start:.1f}s")
        
        # チェックポイントの保存
        torch.save(model.state_dict(), f"ore_apex_perfect_epoch_{epoch}.pt")

    # ==========================================
    # 4. 生成 (知的な文章の出力)
    # ==========================================
    print("\n--- Generating Intelligent Insight ---")
    prompt = "The fundamental nature of intelligence is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0].tolist()
    
    model.eval()
    for _ in range(100):
        with torch.no_grad():
            output = model(input_ids)
            logits = output[:, -1, :] / 1.0
            
            # 強力な単語重複ペナルティ
            for token in set(generated[-40:]):
                if len(tokenizer.decode([token])) > 3:
                    logits[0, token] -= 30.0
            
            # サンプリング
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id: break

    print(f"\nFinal Result: {tokenizer.decode(generated, skip_special_tokens=True)}")

if __name__ == "__main__":
    run_ultimate_training()