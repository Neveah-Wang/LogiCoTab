## ✅Adult
IR = 多数类样本/少数类样本

训练集：
 - 样本数量：26048
 - 多数类样本数（salary is less than 50k）：19775
 - 少数类样本数（salary is greater than or equal to 50k）：6273
 - IR = 3.15

数据来源：https://archive.ics.uci.edu/dataset/2/adult

## ✅shopper

训练集：
 - 样本数量：9247
 - 多数类样本数（False）：7839
 - 少数类样本数（True）：1408
 - IR = 5.57

数据来源：https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset 或者 https://www.kaggle.com/datasets/henrysue/online-shoppers-intention

## ❌convertype
1. **训练的效果比较差。但是在最新的那篇论文里面效果很好**
2. 这个里面的离散数据使用了one-hot编码。我使用的时候把它调整成Order。

数据来源：
https://archive.ics.uci.edu/dataset/31/covertype 或者 
https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset/data


## ✅Buddy
1. TabDDPM使用了
2. 没有明显的逻辑关系
数据来源：https://www.kaggle.com/datasets/akash14/adopt-a-buddy

## ✅Obesity
1. CTSyn使用了
2. 没有明显的逻辑关系

数据来源：https://github.com/shaecodes/Obesity-Level-Prediction

## ✅magic
1. 只有连续数据， 
2. 对于一个椭圆：Length >= Width 
3. TabSyn中使用了
4. 问题：目前我的方法生成的数据中包含逻辑错误数据！！！

训练集：
 - 样本数量：13313
 - 多数类样本数（False）：8636
 - 少数类样本数（True）：4677
 - IR = 1.85

数据来源： https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

## ✅churn
1. 没有明显的逻辑关系
2. label列为0时，NumOfProduct不存在4
3. TabDDPM 使用了

训练集：
 - 样本数量：7000
 - 多数类样本数（0）：5577
 - 少数类样本数（1）：1423
 - IR = 3.92

数据来源：https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling

## ✅Bean
1. Perimeter > MajorAxisLength > MinorAxisLength
2. AspectRation = MajorAxisLength / MinorAxisLength
3. Area < ConvexArea
3. 在 Systematic Assessment of Tabular Data Synthesis Algorithms 中使用了

数据来源：https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

## ✅page
1. 多分类
2. 有逻辑关系

数据来源： https://archive.ics.uci.edu/dataset/78/page+blocks+classification

## fetal
1. 多分类
2. 有逻辑关系

数据来源：https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

## ✅Abalone
1. 回归数据

数据来源：https://archive.ics.uci.edu/dataset/1/abalone

## ✅Bike Sharing
1. 回归数据
2. 季节和月份有对应关系
3. casual + registered = cnt

数据来源：https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

## ✅Insurance
1. 回归数据

数据来源：https://www.kaggle.com/datasets/mirichoi0218/insurance

## 🟩Productivity
1. 回归数据

数据来源：https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees


## 🟩Insurance（另一个同名的数据集）
1. 回归数据
2. 有癌症病史的人 至少做过1次手术

数据来源：https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction


## Rice (Cammeo and Osmancik)
1. 二分类
2. 跟bean数据集有点相似，可以一试


数据来源：https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik

## occurpancy
1. 二分类
2. 没有明显的逻辑关系

数据来源：https://archive.ics.uci.edu/dataset/357/occupancy+detection


## Statlog 
1. 二分类
2. 没有明显的逻辑关系

数据来源：https://archive.ics.uci.edu/dataset/148/statlog+shuttle

## HTRU2
1. 二分类
2. 没有明显的逻辑关系

数据来源：https://archive.ics.uci.edu/dataset/372/htru2

## acceleromater
1. 二分类
2. 没有明显的逻辑关系

数据来源：https://archive.ics.uci.edu/dataset/755/accelerometer+gyro+mobile+phone+dataset


## Productivity Prediction of Garment Employees
1. 回归数据
2. 可以一试（有逻辑关系）

数据来源：https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees

## heart
1. 没有明显的逻辑关系

数据来源：https://archive.ics.uci.edu/dataset/145/statlog+heart

分析讲解：https://blog.csdn.net/m0_51732188/article/details/116131636

## Robot
1. 没有明显的逻辑关系

## CMC
1. 没有明显的逻辑关系

数据来源：https://archive.ics.uci.edu/dataset/30/contraceptive+method+choice

## Nursery
1. 没有明显的逻辑关系

数据来源：https://archive.ics.uci.edu/dataset/76/nursery

数据来源：https://www.kaggle.com/datasets/uciml/wall-following-robot

## gesture
1. TabDDPM用了
2. 没有明显的逻辑关系

数据来源：https://archive.ics.uci.edu/dataset/302/gesture+phase+segmentation

## credit
1. 只有num数据
2. 没有明显的逻辑关系


数据来源
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## default
数据来源：https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients 或者 https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset


