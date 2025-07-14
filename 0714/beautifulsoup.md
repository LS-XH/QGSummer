# 环境安装

- Requests

     ```shell
     conda install requests
     ```

- bs4

     ```shell
     conda install beautifulsoup4 
     ```

- lxml解析器

     ```py
     conda install lxml 
     ```

     

# Requests

## 请求网页

- 请求网页

     ```py
     import requests
     requests.get(url)
     
     rqg = requests.get("https://www.baidu.com")
     rqg.status_code # 查看状态码
     rqg.encoding # 查看响应的编码
     rqg.text # 查看网页的内容
     rqg.content #以二进制方式获取http请求的内容
     ```

# BeautifulSoup

## 基础

### 创建一个soup并初始化

- 创建一个bs实例

     ```py
     soup = BeautifulSoup(text, "lxml")
     ```

- 整理成有序的html文档

     ```py
     print(soup) #原始的
     print(soup.prettify()) #整理后的
     ```

### 架构

bs主要有四种对象

- Tag：html中的一个标签
- NavigableString：字符串，主要是标签之间的字符串
- BeautifulSoup：最先生成的bs
- Comment：注释部分

## 查询

- find：查找匹配到的第一个标签

     ```py
     #name: 标签名字
     #attrs: 标签属性的字典
     def find(self,name,attrs,recursive: bool = True,string,**kwargs) -> Tag | NavigableString | None
      
     soup.find()
     ```

## Tag

### 获取信息

```py
#此处以soup.a标签举例

#标签名称
soup.<tag>.name -> str
soup.a.name

#标签属性
soup.<tag>.attrs -> dict
soup.a.attrs

#标签文本内容
soup.<tag>.string -> str
soup.a.string

```

### 节点选择

- #### 嵌套选择

     ```py
     #获取标签的子标签
     soup.<tag>.<tag> -> tag
     soup.head.title
     ```

- #### 关联选择

     - 子

          ```py
          #获取所有子标签(迭代器)
          soup.<tag>.children -> Iterator(tag)
          
          
          #标签所有子内容(列表)
          soup.<tag>.contents -> list(tag)
          soup.a.contents
          
          #标签所有子孙标签
          soup.<tag>.descendants -> Iterator(tag)
          ```

          

     - 父

          ```py
          #获取标签的父标签
          soup.<tag>.parent -> tag
          soup.head.title.parent      (head)
          
          
          #获取标签的祖先标签，返回从父标签（1）到最外层标签（n）的所有祖先标签
          soup.<tag>.parents -> Iterator(tag)
          ```

          

     - 兄弟

          ```py
          #获取同级的后一个标签
          soup.<tag>.next_sibling -> tag
          
          #获取同级后面的所有标签
          soup.<tag>.next_siblings -> Iterator(tag)
          
          
          #获取同级的前一个标签
          soup.<tag>.previous_sibling -> tag
          
          #获取同级后面的所有标签
          soup.<tag>.previous_siblings -> Iterator(tag)
          ```

          

     