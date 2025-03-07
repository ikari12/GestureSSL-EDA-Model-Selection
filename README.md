# 局所的な時間変動を含むジェスチャーに適したTNCとその転移学習による精度向上

## ジェスチャー認識，表現学習のため参考のサーベイ

2020年以降のArxivの参照に基づくSelf-Supervisedの傾向

UWaveとHARのEDAより，時間局所性がキーポイントとなる可能性が高く，TNCを採用

| 手法名 | 概要 | 発表年月 | 精度 | 計算コスト | 解釈性 | 引用 |
|---|---|---|---|---|---|---|
| TSDE | Diffusionモデルを用いた時系列データの表現学習手法で，データの欠損補完や予測を通じて特徴を学習する． | 2024年5月 | 高い | 中程度 | 高い | arXiv:2405.05959 |
| Series2Vec | 時系列データの類似性に基づく自己教師あり表現学習手法で，時間領域と周波数領域の両方で類似性を学習する。 | 2023年12月 | 中程度 | 高い | 高い | arXiv:2312.03998 |
| TimesURL | 普遍的な時系列表現（長短期変動に頑強）学習のための自己教師ありコントラスト学習フレームワークで，時間再構成とコントラスト学習を組み合わせている． | 2023年12月 | 中程度 | 中程度 | 高い | arXiv:2312.15709 |
| TNC（採用） | 時系列データの近傍情報を活用し，局所的な時間的整合性を考慮した自己教師あり学習手法． | 2022年8月 | 高い | 低い | 高い | arXiv:2208.08393 |
| CoST | 対比学習と時間的変換を組み合わせた時系列データの表現学習手法． | 2022年5月 | 高い | 低い | 中程度 | arXiv:2205.09101 |
| TSTCC | 時間的・周波数的コントラスト学習を活用した時系列表現学習手法． | 2021年11月 | 高い | 中程度 | 中程度 | arXiv:2111.08418 |
| TS2Vec | 階層的時系列データのコントラスト学習を行うフレームワーク． | 2021年10月 | 高い | 中程度 | 中程度 | arXiv:2110.08266 |
| CPC | Contrastive Predictive Coding を用いた時系列データの表現学習． | 2020年1月 | 中程度 | 中程度 | 高い | arXiv:1905.09272 |

## 以下の手順をまとめたNotebook

下記の手順は，「ジェスチャー識別のための時系列自己教師あり学習：EDAとモデル選定とその結果報告.ipynb」でも再現できる．

## 計算機環境の構築

Python3.11が推奨，OSはUbuntu22.04が推奨．

```
pip install -r requirements.txt
```

## データの作成と前処理

HARデータをTNC用に変換する．

```
!python tnc/har.py
```

UwaveデータをTNC用に変換する．

```
!python tnc/uwave.py
```

## HARの表現学習と識別を合わせた学習とその評価

ここで転移の元となるHARを学習したTNCも同時に作成される．

```
!python -m tnc.tnc --data har --train --w 0.00001
```

## HARの表現学習と識別を合わせた学習とその評価

ここで転移の先となるUwaveを学習したTNCの成績がベースラインとして求められる．

```
!python -m tnc.tnc --data waveform --train --w 0.00001
```

## HARの表現学習をUwaveに転移させた際の評価

これにより，TNCの優位性が示される．

```
!python -m tnc.tnc --data waveform --train --transfer --w 0.00001
```

# 参考と謝辞

## References

1. Tonekaboni, S., Eytan, D., & Goldenberg, A. (2021). Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding. International Conference on Learning Representations. https://openreview.net/forum?id=8qDwejCuCN
2. https://openreview.net/forum?id=8qDwejCuCN

## Acknowledgements

- https://github.com/sanatonek/TNC_representation_learning?tab=readme-ov-file
- https://seunghan96.github.io/cl/ts/(CL_code3)TNC/



