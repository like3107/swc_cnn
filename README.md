# swc_cnn
本代码运用了卷积神经网络训练神经元swc数据<br>
建议使用anaconda运行，需要安装pyzem包。<br>
https://github.com/janelia-flyem/pyzem　<br>
首先解压数据集到celldata文件夹下<br> 
cd code 执行python run.py 进行数据预处理，得到test.tfrecords和train.tfrecords<br> 
python cifar10.py 进行卷积神经网络训练<br> 
训练得到不同迭代次数的模型和准确率
