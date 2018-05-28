# swc_cnn
本代码运用了卷积神经网络训练神经元swc数据<br>

## 库安装
建议使用anaconda运行，需要安装pyzem包。<br>
https://github.com/janelia-flyem/pyzem　<br>
依次安装requirements中的关键库。其余库若需要根据错误提示安装。<br>

## 代码运行
首先解压数据集到celldata文件夹下<br> 
cd code 执行python run.py 进行数据预处理，得到test.tfrecords和train.tfrecords<br> 
python cifar10.py 进行卷积神经网络训练<br> 
训练得到不同迭代次数的模型和准确率

## 其余代码说明
runsia.py　为运行siamese network进行训练的函数，其中网络结构课可以在inference.py中改变，运行需要预处理并且添加一个model2的文件夹。<br>
net_out.py　为加载现有模型，输入数据（png）输出csv格式的输出数据的函数，运行这个函数需要检查路径是否正确和各个类型数据的个数<br>
computea.py 为计算检索准确率的函数，利用net_out.py 输出训练集和测试集的网络输出，输入进行计算可得siamese network的检索准确率，运行前检查路径以及各类神经元个数
