
# Code description
- bm_run.py 需要替换run.py
- bm_SUT.py 模型使用bm芯片，接入了loadgen
- run.sh 启动脚本
- env.sh 环境脚本，安装loadgen和其他需要的包


# Summary
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
Samples per second: 1.21305
Result is : INVALID
  Min duration satisfied : NO
  Min queries satisfied : Yes
Recommendations:
 * Increase expected QPS so the loadgen pre-generates a larger (coalesced) query.
```
