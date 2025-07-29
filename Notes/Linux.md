# 指令大全

- ### su/sudo

     切换用户





- ### apt

     包管理器

     - #### 安装vim

          ```shell
          sudo apt update
          sudo apt install vim
          ```

     - #### 换源

          1. 文件在\etc\apt\sources.list

               ```
               cd \etc\apt
               ```

          2. 清空sources.list文件

               ```shell
               >sources.list
               ```

          3. 写入sources.list文件

               ```shell
               vim sources.list
               
               
               #阿里源
             deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse
               deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse
               deb http://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse
               deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse
             ```
          
          4. 更新apt
             
               ```shell
               apt-get update
               ```

- ### vim

     编辑器

     - ##### 打开一个文件

          ```shell
          vim sources.list
          ```

     - ##### 普通模式（Normal Mode）‌：按 `Esc` 进入，可执行命令（如 `:wq`）。

     - ##### **插入模式（Insert Mode）**‌：按 `i` 或 `a` 进入，用于编辑文本。

     - ##### ‌**可视模式（Visual Mode）**‌：按 `v` 进入，用于文本选择。

     - ##### 退出

          先按Esc退出Insert模式

          | 操作           | 快捷键        |
          | -------------- | ------------- |
          | 保存并退出     | `:wq` 或 `ZZ` |
          | 强制退出不保存 | `:q!`         |
          | 仅保存         | `:w`          |
          | 另存为其他文件 | `:w <新路径>` |


- ### chmod

     修改文件或目录的权限

     - 

