毕业论文《基于宽度&深度学习的离网预测模型》的实验代码

## 代码依赖关系

在完成实验的过程中，主要使用了以下工具包：

* `Keras`: 搭建神经网络模型，后端使用TensorFlow
* `Matplotlib`: 画模型的损失曲线
* `Pandas`: 原始数据预处理
* `H5py`: 保存预处理过后的数据，加速读数据
* `Scikit-learn`: 计算评价指标

此外，在涉及到矩阵操作的地方，均使用`Numpy`来计算。

## 目录结构

以下是对实验代码目录结构的简要描述：

* `data/`: 模型的数据集，原始训练集和测试集的格式见`docs/xxx`
  * `train_all.csv`: 原始训练集
  * `test.csv`: 原始测试集
  * `true`: 测试集标签
* `config/`: 训练不同模型和预测不同模型使用的配置文件
  * `env.sh`: 配置文件通用的环境变量
  * `train/`: 训练阶段使用的配置文件
  * `predict/`: 预测阶段使用的配置文件
* `docs/`: 各种文档和毕业论文
  * `数据说明图.pdf`: 原始数据集的格式
  * `字段说明表.xlsx`: 原始数据集的字段含义
  * `答辩.pptx`: 答辩ppt
  * `毕业论文.docx`: 毕业论文doc版
  * `毕业论文.pdf`: 毕业论文pdf版
* `src/`: 所有代码文件
  * `batch/`: 生成模型需要的mini-batch输入
  * `csv2hdf5/`: 将原始数据集转换为hdf5格式数据集
  * `model/`: 各种网络模型
  * `shuffle.py`: 打乱训练集并划分验证集
  * `min_max_scaler.py`: 数据集归一化，效果不好，此文件废弃
  * `utils.py`: 包括参数解析、获取评价指标报告、画损失曲线3个函数
  * `tf_backend.py`: Keras后端使用TensorFlow时tf的参数设置
  * `ensemble.py`: 取平均值法融合模型
  * `main.py`: 模型训练和预测的入口
* `train.sh`: 模型训练脚本
* `predict.sh`: 模型预测脚本
* `choose_high_score.sh`: 无限循环训练模型，自动保存Top 25000的Precision最高的模型

除了以上目录，模型在训练过程中会在以下目录保存文件：

* `saved_model/`: 训练好的模型
* `report/`: 模型的评价指标报告
* `predict_proba/`: 模型在测试集上的预测概率
* `curve/`: 模型训练过程的损失曲线
* `struct_pic/`: 模型网络结构图

## 代码使用方法

#### 数据预处理

使用以下步骤来将已有的原始数据转换为模型需要的输入（如已有中间步骤的数据文件，可以酌情跳过）：

1. 进入 `10.10.64.206` 服务器的 `/home/shihongbin/projects/churn` 目录，我的账号的密码为 `shb` ;
2. 将原始训练集和测试集放到 `data/` 目录下;
3. 运行 `python3 src/shuffle.py data/train_all.csv data/train.csv data/valid.csv` 来分割原始训练集，得到新训练集和验证集;
4. 运行 `python3 src/csv2hdf5/base.py data/train.csv data/valid.csv data/test.csv data/shuffle.hdf5` 来将csv文件转换为hdf5文件。

#### 模型训练

共有 `train.sh` 和 `choose_high_score.sh` 两种方式，区别见目录结构一节中的介绍。模型训练过程结束后会同时得到验证损失最小的模型在测试集上的预测结果。以下为训练模型的具体步骤：

1. 在 `src/model/` 中定义模型结构，定义方式参考该目录下的文件;
2. 在 `src/batch/` 中定义数据输入，定义方式参考该目录下的文件;
3. 在 `config/train/` 中写配置文件，配置参数参考该目录下的文件;
4. 运行 `mkdir saved_model report predict_proba curve struct_pic tmp` 来创建训练过程中需要使用的目录;
5. 对于 `train.sh` ，运行 `./train.sh config/train/cnn_9_month.sh 0` 来训练模型。其中 `cnn_9_month.sh` 可以替换成该目录下任意配置文件， `0` 表示gpu的id，可选id有0、1、2，使用 `nvidia-smi` 查看gpu使用情况后选择空闲的gpu。对于 `choose_high_score.sh` ，运行 `./choose_high_score.sh config/train/cnn_9_month.sh 0` 来训练模型，在 `tmp/` 下会生成与配置文件同名的日志文件，使用 `Ctrl+C` 终止训练脚本。

#### 模型预测

如果已有训练好的模型，想直接得到在测试集上的预测结果，可以运行 `./predict.sh config/predict/cnn_9_month.sh saved_model/xxx.hdf5 predict_proba/xxx 0` 来使用第二个参数中训练好的模型对测试集进行预测，结果保存到第三个参数的文件中。

#### 模型融合

模型融合脚本目前支持对多个预测的概率值进行取平均值融合，融合时对多个预测结果求所有可能的组合，找出分数最高的组合。使用方式如下：

1. 创建文件 `xxx` ，将需要进行融合的预测结果的文件名（相对或绝对路径）写入这个文件，每行一个文件名；
2. 运行 `cat xxx | xargs python3 src/ensemble.py data/true` 开始融合，融合日志打印在屏幕上，如需保存，可以在管道中加上 `| tee log_name` 。
