- 本实验主要实现了王健师兄的sparse coding方法
- 算法流程
    - ExtractPatches.py 主要是用来提取patch,　保存成.mat格式,
    - 然后交给python3 去学习得到字典，
    - 然后再执行main.py, 得到最终的分类结果
－　python3 是指需要安装了python3的环境，并且安装了ksvd库