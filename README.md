# GlobalPointer
全局指针统一处理嵌套与非嵌套NER。

## 介绍

- 博客：https://kexue.fm/archives/8373

## 效果

### 人民日报NER

| | 验证集F1 | 测试集F1 | 训练速度 | 预测速度 |
| :-: | :-: | :-: | :-: | :-: |
| CRF | 96.39% | 95.46% | 1x | 1x |
| GlobalPointer (w/o RoPE) | 54.35% | 62.59% | 1.61x | 1.13x |
| GlobalPointer (w/ RoPE) | 96.25% | 95.51% | 1.56x | 1.11x |

### CLUENER

| | 验证集F1 | 测试集F1 | 训练速度 | 预测速度 |
| :-: | :-: | :-: | :-: | :-: |
| CRF | 79.51% | 78.70% | 1x | 1x |
| GlobalPointer | 80.03% | 79.44% | 1.22x | 1x |

### CMeEE

| | 验证集F1 | 测试集F1 | 训练速度 | 预测速度 |
| :-: | :-: | :-: | :-: | :-: |
| CRF | 63.81% | 64.39% | 1x | 1x |
| GlobalPointer | 64.84% | 65.98% | 1.52x | 1.13x |

## 环境

需要`bert4keras >= 0.10.6`。个人实验环境是tensorflow 1.14 + keras 2.3.1 + bert4keras 0.10.6。

## 交流
QQ交流群：808623966，微信群请加机器人微信号spaces_ac_cn
