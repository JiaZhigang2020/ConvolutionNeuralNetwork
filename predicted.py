import os
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
import glob
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QTextBrowser, QHBoxLayout


class Fc(torch.nn.Module):
    def __init__(self, in_features):
        super(Fc, self).__init__()
        self.linear_1 = torch.nn.Linear(in_features=in_features, out_features=256)
        self.linear_2 = torch.nn.Linear(in_features=256, out_features=64)
        self.linear_3 = torch.nn.Linear(in_features=64, out_features=3)
        self.dropout_1 = torch.nn.Dropout2d(p=0.7)
        self.dropout_2 = torch.nn.Dropout2d(p=0.5)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        x = self.dropout_1(F.relu(self.linear_1(input)))
        x = self.dropout_2(F.relu(self.linear_2(x)))
        logits = self.linear_3(x)
        return logits


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        resnet_model = torchvision.models.resnet18(pretrained=True)
        in_features = resnet_model.fc.in_features
        resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
        self.resnet_model = resnet_model
        self.fc = Fc(in_features)

    def forward(self, input):
        x = self.resnet_model(input)
        logits = self.fc(x)
        return logits


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path):
        self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.Resize((343, 1000)),
                            torchvision.transforms.ToTensor()
        ])
        self.imgs_path = imgs_path

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil)
        if len(img_np.shape) == 2:
            img_np = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)
            img_pil = Image.fromarray(img_np)
        img = self.transform(img_pil)
        return img.type(torch.float32)

    def __len__(self):
        return len(self.imgs_path)


class Predicted:
    def __init__(self, model):
        self.predicted_folder = './predicted_folder/'
        self.imgs_file_path = glob.glob(self.predicted_folder + '*.jpg')
        self.model = model
        self.index_to_label = {0: '水稻白叶枯病（Bacterial leaf blight）', 1: '水稻褐斑病（Brown spot）', 2: '水稻叶黑粉病（Leaf smut）'}

    def get_imgs_dataset(self):
        return MyDataset(self.imgs_file_path)

    def pred(self, imgs_dataset):
        pred_result = ""
        with torch.no_grad():
            for i, img in enumerate(imgs_dataset):
                pred = model(img.reshape(1, *img.shape)).argmax(1).item()
                pred = self.index_to_label[pred]
                pred_result += "<h2>" + (self.imgs_file_path[i].split('\\')[-1] + ":" + pred + "\n") + "</h2>"
            print(pred_result)
        return pred_result

    def main(self):
        imgs_dataset = Predicted.get_imgs_dataset(self)
        pred = Predicted.pred(self, imgs_dataset)
        return pred


class Demo(QWidget):
    def __init__(self, model):
        super(Demo, self).__init__()
        self.resize(500, 300)
        self.model = model
        self.setWindowTitle('水稻病害识别系统')
        self.text_edit = QTextBrowser(self)
        self.text_edit.setText('<h1>使用方法：</h3>'
                               '<h2>1.单击“导入病害图片”按钮，并在弹出的文件夹内放入需要判别的病害图片；</h2>'
                               '<h2>2.单击“开始预测”，获取预测结果；</h2>'
                               '<h2>3.对结果存疑或有其它疑难问题欢迎点击"人工辅助"，加群与我们交流。</h2>')

        self.pre_btn = QPushButton('开始预测', self)
        self.mkdir_btn = QPushButton('导入病害图片', self)
        self.help_btn = QPushButton('帮助信息', self)
        self.prevent_measures_btn = QPushButton('人工辅助', self)
        self.pre_btn.clicked.connect(lambda: self.predicted())
        self.mkdir_btn.clicked.connect(lambda: self.mkdir())
        self.prevent_measures_btn.clicked.connect(lambda: self.artificial_auxiliary())
        self.help_btn.clicked.connect(lambda: self.help())

        self.h_layout_1 = QHBoxLayout()
        self.h_layout_2 = QHBoxLayout()
        self.v_layout = QVBoxLayout()
        self.h_layout_1.addWidget(self.pre_btn)
        self.h_layout_1.addWidget(self.mkdir_btn)
        self.h_layout_2.addWidget(self.prevent_measures_btn)
        self.h_layout_2.addWidget(self.help_btn)
        self.v_layout.addWidget(self.text_edit)
        self.v_layout.addLayout(self.h_layout_1)
        self.v_layout.addLayout(self.h_layout_2)
        self.setLayout(self.v_layout)

    def mkdir(self):
        if not os.path.exists('./predicted_folder'):
            os.mkdir('./predicted_folder')
            self.text_edit.setText('<h3>已创建,并打开文件夹，请放入需要判别的病害图片后单击“开始预测”获取结果。</h3>')
        else:
            self.text_edit.setText('<h3>已打开文件夹，请放入需要判别的病害图片后单击“开始预测”获取结果。</h3>')
        os.system("explorer .\predicted_folder\\")

    def predicted(self):
        # print('正在识别')
        pred = Predicted(self.model).main()
        # print(pred)
        # os.sys.exit()
        self.text_edit.setText(pred)

    def help(self):
        self.text_edit.setText('<h1>使用方法：</h3>'
                               '<h2>1.单击“导入病害图片”按钮，并在弹出的文件夹内放入需要判别的病害图片；</h2>'
                               '<h2>2.单击“开始预测”，获取预测结果；</h2>'
                               '<h2>3.对结果存疑或有其它疑难问题欢迎点击"人工辅助"，加群与我们交流。</h2>')

    def artificial_auxiliary(self):
        img = Image.open('./qq.jpg')
        img.show()


if __name__ == '__main__':
    model = Model()
    model_file = 'best_model/0_flod.pth'
    model.load_state_dict(torch.load(model_file))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    app = QApplication(sys.argv)
    demo = Demo(model)
    demo.show()
    sys.exit(app.exec_())
