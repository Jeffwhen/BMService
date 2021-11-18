
# Code description
- bm_run.py 需要替换run.py
- bm_SUT.py 模型使用bm芯片，接入了loadgen
- run.sh 启动脚本
- env.sh 环境脚本，安装loadgen和其他需要的包
- compile/tf_to_fp32bmodel.sh 把tf的pb文件转成fp32bmodel


# 阶段问题（20211118）
1. 目前只测试了fp32的模型，int8模型之前同事分析过在bm1684上的性能（性能较差），所以未测试
2. 等BM1686出片后，在BM1686上测试