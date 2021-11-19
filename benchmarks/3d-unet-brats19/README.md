
# Code description
```
.
├── README.md
├── accuracy-brats.py
├── bm_SUT.py
├── bm_run.py  入口代码
├── brats_QSL.py
├── compile
│   └── torch_to_fp32bmodel.sh
├── env.sh  当前任务依赖的一些包
├── nnUnet  依赖的第三方库
├── run.sh  执行脚本
└── user.conf
```

# How run
执行前，现执行根目录下的`env.sh`安装load_gen，然后执行当前目录下的`env.sh`和`run.sh`
```
# Under the root dir
source env.sh
# Under current dir
git submodule update --init --recursive
source env.sh
bash run.sh
```

# Current problem（20211118）
1. 目前只测试了fp32的模型，int8模型转换的时候有一个deconv3d算子BM1684不支持，所以未测试
2. 等BM1686出片后，在BM1686上测试