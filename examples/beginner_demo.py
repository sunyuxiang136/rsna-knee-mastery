"""RSNA Knee 小白演示：不需要 GPU，只用 numpy。

演示四件事（对应小白版文档第 4~7 节）：
  1. 按毫米裁剪：为什么不同分辨率的图能裁出"同样大小的膝盖"
  2. softmax 注意力：模型怎么给每张切片打"重要度"
  3. 每个疾病各用一套注意力（RaptorClassifier.head 的缩小版）
  4. 排名融合：多个模型的分数怎么合成一个，以及某个模型失败时怎么办

运行：python3 beginner_demo.py
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. 按毫米裁剪
print("=" * 60)
print("1. 按毫米裁剪（CROP_MM = 140）")
CROP_MM = 140.0
for pixel_spacing in (0.3, 0.5):  # 每个像素代表多少毫米
    crop_px = round(CROP_MM / pixel_spacing)  # 要裁多少个像素才等于 140 毫米
    print(f"  像素间距 {pixel_spacing} mm -> 裁 {crop_px} 像素 -> 再缩放到 224x224")
print("  结论：两张图最后都是 224x224，且都正好覆盖真实的 140mm，膝盖大小一致。")


# ---------------------------------------------------------------- 2. softmax
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


print("=" * 60)
print("2. softmax：把任意分数变成'加起来等于 1 的权重'")
scores = np.array([0.2, 3.0, 0.1, 0.5])  # 4 个窗口的"原始重要度"
w = softmax(scores)
print("  原始分数:", scores)
print("  softmax 权重:", w, " 总和 =", w.sum().round(3))
print("  第 2 个窗口分数最高，所以它拿到最大权重。")

# ---------------------------------------------------------------- 3. 每类一套注意力
print("=" * 60)
print("3. 每个疾病一套注意力（缩小版 RaptorClassifier.head）")
K, F, N = 4, 6, 3  # 4 个窗口、每个窗口 6 维特征、3 个疾病(真实是 12 个)
labels = ["ACL", "Effusion", "Fracture"]

feats = rng.normal(size=(K, F))  # backbone 给每个窗口算出的特征
att_W = rng.normal(size=(F, N))  # 注意力层：每个疾病一列
cls_W = rng.normal(size=(N, F))  # 每个疾病一个分类向量
cls_b = np.zeros(N)

att_score = feats @ att_W  # [K, N] 每个窗口对每个疾病的"重要度"
att = softmax(att_score, axis=0)  # 在"窗口"维度做 softmax
print("  注意力权重 [窗口 x 疾病]（每一列和为 1）:")
print(att)

pooled = np.einsum("kn,kf->nf", att, feats)  # 每个疾病自己的加权平均特征 [N, F]
logits = (pooled * cls_W).sum(-1) + cls_b  # 每个疾病一个 logit
prob = 1 / (1 + np.exp(-logits))  # sigmoid -> 概率
for name, p in zip(labels, prob):
    print(f"  {name:9s} 概率 = {p:.3f}")
print("  不同疾病的注意力列不一样 -> 它们各自'看'不同的切片。")


# ---------------------------------------------------------------- 4. 排名融合
def pct_rank(v):
    """百分位排名：最小 -> 接近 0，最大 -> 1。只保留'谁排在前面'，丢掉分数大小。"""
    order = v.argsort().argsort().astype(float)
    return (order + 1) / len(v)


print("=" * 60)
print("4. 排名融合（5 个病人，某疾病的预测）")
model_a = np.array([0.90, 0.80, 0.30, 0.20, 0.10])  # 模型 A 很'自信'
model_b = np.array([0.55, 0.60, 0.52, 0.51, 0.40])  # 模型 B 很'保守'，但排序一致
print("  A 原始概率:", model_a)
print("  B 原始概率:", model_b)
print("  A 排名    :", pct_rank(model_a))
print("  B 排名    :", pct_rank(model_b))
print("  若直接平均概率，B 几乎没作用；用排名就公平了（AUC 只看排序）。")

w_raptor = 0.8  # 比如 Synovitis 的 0.8
raptor_half = 0.65 * pct_rank(model_a) + 0.35 * pct_rank(model_b)
r6 = pct_rank(np.array([0.7, 0.9, 0.2, 0.3, 0.1]))
final = w_raptor * raptor_half + (1 - w_raptor) * r6
print("  最终分数  :", final)

print("\n  --- 模型失败的处理：失败的病人在 A 里是整行 0.5（哨兵值）---")
a_fail = np.array([False, False, True, False, False])  # 第 3 个病人 A 失败
b_rank_ok = pct_rank(model_b)
a_scores = np.where(a_fail, 0.5, model_a)
# 只在"成功的行"里排名，失败行不参与，避免 0.5 把别人的排名挤歪
a_rank = np.full(5, np.nan)
a_rank[~a_fail] = pct_rank(a_scores[~a_fail])
combined = np.where(a_fail, b_rank_ok, 0.65 * a_rank + 0.35 * b_rank_ok)
print("  A 的排名(失败为 nan):", a_rank)
print("  失败的那位直接用 B 的排名:", combined)
