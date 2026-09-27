# 本机训练恢复模拟闭环

状态：SUCCEEDED
版本：0.3.5
源码指纹：`48e7bb77ba360fb7885c8ea2c5e25553efce38c3e79cc9d38db4c5770ea4a1ca`

| 门槛 | 退出码 | 原始输出 |
| --- | ---: | --- |
| static | 0 | [static](static.txt) |
| tests | 0 | [tests](tests.txt) |
| cpu-acceptance | 0 | [cpu-acceptance](cpu-acceptance.txt) |
| package | 0 | [package](package.txt) |
| wheel | 0 | [wheel](wheel.txt) |
| fresh-install | 0 | [fresh-install](fresh-install.txt) |
| upgrade-boundary | 0 | [upgrade-boundary](upgrade-boundary.txt) |

[测试故障与恢复原始目录](test-artifacts/)保留在本次结果目录中。

真实 CPU 验收：SUCCEEDED；10/10 个故障/负控案例通过。
[验收原始记录](acceptance/acceptance-6d2db60966d2/acceptance.json)。

本机矩阵包含真实 CPU 双进程恢复、双启动器拓扑、提交切点硬退出、
异步协调与事件耐久顺序，以及对象条件提交和 epoch 接管的协议模拟。
真实 CUDA、跨主机隔离、远程对象服务和主机断电仍需对应环境实测。
