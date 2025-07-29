# Git使用方法

## 安装及环境配置

## 指令

- ### SSH配置

     - 用电脑本地的ssh生成公钥私钥

          ```cmd
          ssh-keygen -t <生成算法>
          
          #example:
          
          #1):rsa
          #输入之后，连按三个回车键（什么都不需要输入）。
          #然后就会生成两个文件 – id_rsa 和 id_rsa.pub。其中id_rsa是私钥，id_rsa.pub是公钥。
          ssh-keygen -t rsa
          ```
          
     - 由于github的端口被封禁，需要绕过，在`C:\Users\XH\.ssh\config`中配置

          ```
          Host github.com
            Hostname ssh.github.com
            Port 443
            User git
          ```

     - 验证ssh是否连接成功

          ```cmd
          ssh -T git@github.com
          ```

     - 在GitHub中，个人账号界面-Setting-SSH and GPG keys-SSH keys-New SSH keys，放入公钥

- ### Git代理配置

     校园网连不上Github喜欢吗

     cmd中配置git代理

     ```cmd
     git config --global http.proxy http://proxy_address:port
     git config --global https.proxy https://proxy_address:port
     
     #example:
     git config --global http.proxy http://127.0.0.1:7890
     git config --global https.proxy https://127.0.0.1:7890
     ```

     

- ### 下载

     - ##### 法1：克隆远程仓库

          ```cmd
          git clone <远程仓库链接*.git> <要放入的本地文件夹>
          ```

     - ##### 法2：拉取仓库
          
          - 初始化仓库
          
               ```cmd
               git init
               ```
          - 和远程仓库连接
          
               ```cmd
               git	remote add <远程主机名（自己起）> <远程仓库地址>
               
               #example
               git	remote add origin git@github.com:lilia1204/test.git
               ```
          
               

- ### 上传文件

     - ##### 添加文件至暂存区

          ```cmd
          git add <文件路径>
          git add .   //添加所有变动的文件
          ```

     - ##### 提交文件

          ```cmd
          git commit <文件路径> -m <备注>
          ```

     - 推送文件至远程仓库

          ```cmd
          git push <远程主机名（自己起的那个）> <远程分支名>
          git push 				#简洁方法
          ```

          



- ### 文件操作

     - ##### 查看当前所有文件状态

          ```cmd
          git status
          ```

     - ##### 忽略文件

          在主目录下建立".gitignore"文件，语法如下

          ```cmd
          *.txt	#忽略所有txt文件
          !lib.txt	#但lib.txt除外
          /temp	#仅忽略根目录的temp文件夹以及子文件
          build/	#忽略build目录下的所有文件
          doc/*.txt	#仅忽略doc目录下的txt文件,不包括子目录
          ```

          