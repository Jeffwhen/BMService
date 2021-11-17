
# code description
- bm_run.py 需要替换run.py
- bm_SUT.py 模型使用bm芯片，接入了loadgen
- run.sh 启动脚本
- env.sh 环境脚本，安装loadgen和其他需要的包


# Summary
## SingleStream
- batch_size = 1
- SingleStream
- PerformanceOnly
- 三个三芯卡

```
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : SingleStream
Mode     : PerformanceOnly
90th percentile latency (ns) : 609750011
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
```


## Offline
- batch_size = 1
- Offline
- PerformanceOnly
- 三个三芯卡


```
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 14.7862
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
```



- batch_size = 2
- Offline
- PerformanceOnly
- 三个三芯卡


```
ERROR
```