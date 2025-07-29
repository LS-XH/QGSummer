# 制卡

在<a src="http://www.orangepi.org/">香橙派官网</a>下载ubuntu镜像

可在Ascend-产品-Atlas开发者套件-（往下翻）快速入门-下载制卡工具

在<a src="http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-AIpro.htmlOrangePi ">OrangePi AIPro（8T）</a>中-下载-下载Ubuntu镜像（Desktop）

# SSH连接

#### 在OrangePi中

- 使用**nmtui**：Edit a connection-Wired connection1-Edit-IPV4将**Automatic**改成**Manual**
     - 设置地址Addresses
     - Gateway是XXX.XXX.XXX.1
     - DNS随便8.8.8.8

#### 在电脑中

- 控制面板-网络和Internet-以太网-IPV4，设置自己的IP，网段要一样
- 子网掩码255.255.255.0
- SSH连接到OrangePi的IP，==端口为22==

# Vscode连接

<a src="https://www.hiascend.com/forum/thread-0243177209352216074-1-1.html">参考</a>

- 安装Remote SSH插件

- 点击SSH右边的+号新建连接，在顶部框中输入ssh连接命令

     ```shell
     ssh <username>@<ipv4>:<port>
     
     #example:
     ssh root@192.168.1.110:22
     ```

## Vscode+Conda运行python

在右下角，点击一个类似版本的东西，可以改变py解释器

# 安装CANN

<a src="https://www.mindspore.cn/docs/zh-CN/r2.4.10/orange_pi/environment_setup.html">参考</a>

- #### 软件包升级

     - *卸载旧版本CANN：在/usr/local/Ascend/ascend-toolkit/，卸载7.0.0的

          ```
          sudo ./cann_uninstall.sh
          ```

     - <a src="https://www.hiascend.com/developer/download/community/result">官网下载CANN</a>，选择arm架构的8.0.0beta的**toolkit开发包**`Ascend-cann-toolkit_8.0.0_linux-aarch64.run`

     - 添加可执行权限

          ```shell
          chmod +x ./Ascend-cann-toolkit_8.0.0_linux-aarch64.run
          ```

     - 安装

          ```shell
          ./Ascend-cann-toolkit_8.0.0_linux-aarch64.run --install
          ```

- #### 二进制算子包kernel升级

     - 获取npu型号，香橙派是310B4，下载与toolkit同版本的算子包`Ascend-cann-kernels-310b_8.0.0_linux-aarch64.run`

          ```shell
          npu-smi info
          ```

     - 添加可执行权限

          ```shell
          chmod +x ./Ascend-cann-kernels-310b_8.0.0_linux-aarch64.run
          ```

     - 安装

          ```shell
          ./Ascend-cann-kernels-310b_8.0.0_linux-aarch64.run --install
          ```

          

- #### Mindspore升级


