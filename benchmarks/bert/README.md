
# Code description
```
.
├── README.md
├── accuracy-squad.py
├── bm_SUT.py
├── bm_run.py  入口函数
├── compile
│   └── tf_to_fp32bmodel.sh  将pb文件转换成fp32bmodel
├── evaluate-v1.1.py
├── run.sh  执行脚本
├── squad_QSL.py
├── tokenization.py
└── user.conf
```

# How run
执行前，现执行根目录下的`env.sh`安装load_gen，然后执行当前目录下的`run.sh`
```
# Under the root dir
source env.sh
# Under current dir
bash run.sh
```

# Current problem（20211118）
1. 目前只测试了fp32的模型，int8模型之前同事分析过在bm1684上的性能（性能较差），所以未测试
2. 等BM1686出片后，在BM1686上测试