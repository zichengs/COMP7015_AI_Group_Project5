#set par(spacing: 2em)

我们利用 PlantVillage Dataset 训练了三个不同的模型

*Dataset Info:*
- 数据集中包含了不同种类的植物的不同病态或者健康的叶子的照片。
- 数据集中，对于同一片叶子，可能拍摄了几张略有不同的照片。而每张照片在数据集中，分别有color, grayscale和segmented三个不同的版本。
- 每个文件，根据它的叶子的物种，病态和版本，被保存在了相应的文件夹内。例如 a color version of Tomato leaf with Early blight 保存在`color/Tomato___Early_blight`文件夹中。 因此根据每个照片所在的文件夹的名字，我们就能判断出它的物种和病态。
- `leaf-map.json` 文件中表示了图片与叶子ID的对应关系，例如图片`rs_late.b 4894`对应的叶子ID为`Tomato___Late_blight:::12.0`。但是需要注意的是，文件中的数据并不完整，不是每一张照片在文件中都有所记录。所以，有些照片找不到官方登记的leaf ID，所以我们无法得知该照片与其他的一些照片是否属于同一叶子。


*Data split:*
- 训练模型需要把数据集分成training, validation 和 testing 三部分。 在这个项目的训练数据集中，training, validation, testing 数据分割比例为（70%， 15%， 15%）


*Image preprocessing:*
- 


*2D CNN models:*

Basic experimentation:

Evaluation: