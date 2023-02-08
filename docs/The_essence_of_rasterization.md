### 写在前面
    本文的目的是深入浅出地讲解一个软渲染器从最底层如何实现的基本流程，并且分享一些我学习的心路历程。
    我始终认为，亲自动手实现一个一门被广泛运用的项目，始终是学习这门学科最好的途径。
    在编程实现的过程中，思路是尤其宝贵的东西，因为唯有思路顺畅，条理清晰，代码才能有迹可循，才能像一个个小故事一样写出来，
    千千万万个小故事的汇合，最终会如同千千万万条小溪一样，汇流成滔滔不绝的江水。
    缘此，本文在讲解原理的同时，会更加注重思路的开拓与启发。

- Q：为什么会想要开始写渲染器？

其实是因为知乎的一篇问题:
[如何开始用 C++ 写一个光栅化渲染器？](https://www.zhihu.com/question/24786878)
当时觉得能用代码写一个显示各种各样的3D物体简直是太酷了,之后又接触到了闫老师的图形学入门基础:
[GAMES101-现代计算机图形学入门-闫令琪](https://www.bilibili.com/video/BV1X7411F744/?spm_id_from=333.1007.top_right_bar_window_default_collection.content.click&vd_source=081eadd5e0d99615751efaf94759c6ab)
可以说是把渲染器的工作原理（包括光栅化（Rasterization）、几何图形处理（Geometry）、光线追踪（Ray Tracing）、动画（Animation））讲透了，所以我后续的理论基础是根据这门课来的。
另外在 [如何开始用 C++ 写一个光栅化渲染器？](https://www.zhihu.com/question/24786878) 这个问题中我也受到了其中一些答案的启发，尤其值得一提的是 **arrayJY** 提出的 **“‘渐进式’实现渲染管线”**：

> 
> 我推荐“渐进式”实现渲染管线。意思就是先快速搭建一个非常简单的管线，然后再逐步往其中添加内容。最基础的先从模型文件读入顶点，随便设置一个摄像机进行变换，组装三角形，然后光栅化，随便着个色
> 然后加上个环绕相机，再用深度值进行着色，一个简单的Depth Shader就完成了。
> 以上就是从顶点坐标可以得到的内容，接下来对顶点的其他属性进行处理（法线、纹理坐标）。当然也别忘了多换几个模型来测试一下。
> 完成了纹理映射的效果处理完就可以加入光照了。可以先做简单一点的Flat Shading或Gouraud Shading，而我是一步跳到了Phong Shading。
> 这样一来，图形渲染管线的大部分内容都实现完了。接下来再对原来有错误的地方进行修正，比如不正确的插值（透视校正）、超出边界的三角形（齐次裁剪）等等。

给了我一个从0开始写渲染器的大致思路，另外在一些文件读取的代码上，我参考了 [tinyrenderer](https://github.com/ssloy/tinyrenderer) 的一些代码。

- Q：大致打算如何实现，打算做的事？

实现的是一个可以实现键盘交互的光栅化软渲染器，走一遍渲染管线流程，实现光栅化一些基本的东西（投影变换、纹理映射、点光源处理、阴影处理）。日后有时间的话，会研究一下光线追踪的算法。

## 目录

。。。


### 1.线性代数部分

线性代数可以说是图形学的基础，包括了本渲染器实现所依赖的一些基本运算。为了实现这个渲染器，我们至少要实现两个对象：

1. 支持`1 * 3`，`1 * 4`的向量类
2. 支持`3 * 3`, `4 * 4`的矩阵类

> 有些同学可能会和我一样疑惑：既然是实现一个3D世界，理应只用得上一个三维的矩阵和三维的向量，但为什么这儿又写了`1 * 4`的向量、`4 * 4`的矩阵呢？不急，先卖个关子，之后在讲解透视变换时会说。

对于向量类，我们需要至少实现如下方法：

- 向量的初始化
- 获取向量的维度
- 根据索引获取/设置向量中每一个元素的值
- 向量的单位化

我的向量类：

``` cpp
        class Vector
        {

        public:
            // Initialization
            explicit Vector(size_t n);
            // ...

            // Get the value of the Vector by index
            Utils::Result<VectorElemType> operator[](int n) const;

            // Get without result
            VectorElemType GetFast(int n) const;

            // setter
            Utils::DefaultResult Set(size_t index, VectorElemType value) const;
            // ...

            Utils::DefaultResult Unitization();

        private:
            // The pointer indicated to vector.
            Utils::P<VectorElemType[]> _V;
            // The length of the vector.
            std::size_t _N = Vector$::NOT_INITIALIZED_N;
        };
```


对于矩阵类，我们需要至少实现如下方法：

- 矩阵的初始化
- 获取矩阵的维度
- 根据索引获取/设置矩阵中每一个元素的值
- 矩阵与矩阵的乘法
- 矩阵与向量的乘法
- 求转置矩阵
- 求逆矩阵
- 求伴随矩阵
- 求行列式
- 求代数余子式


我的矩阵类：

``` cpp
/**
         * @brief  n * n Square Matrix
         *
         */
        class SMatrix
        {
        public:
            // Initialization
            SMatrix(std::initializer_list<SMatrixElemType> list);

            // Get the size of the Matrix.
            Utils::Result<std::size_t> N() const;

            // Get the value of the Vector by index
            Utils::Result<SMatrixElemType> Get(size_t row, size_t col) const;
            // Setter
            Utils::DefaultResult Set(size_t row, size_t col, SMatrixElemType value) const;

            // Matrix multiplication with matrix
            Utils::DefaultResult operator*=(SMatrix const &sm);

            // Matrix multiplication with vector
            Utils::DefaultResult operator*(Vector &v) const;

            // Transpose matrix
            Utils::Result<SMatrix> operator+() const;
            // Inverse matrix
            Utils::Result<SMatrix> operator-() const;
            // Adjoint matrix
            Utils::Result<SMatrix> operator*() const;

            // The determinant
            Utils::Result<SMatrixElemType> Determinant(std::vector<std::size_t> row_list, std::vector<std::size_t> col_list) const;
            Utils::Result<SMatrixElemType> Determinant() const;

            // algebraic complement
            Utils::Result<SMatrixElemType> AComplement(size_t row, size_t col) const;

        private:
            // The pointer indicated to square matrix.
            Utils::P<SMatrixElemType[]> _SM;
            // The length of the square amtrix.
            std::size_t _N = SMatrix$::NOT_INITIALIZED_N;
        };

```

### 2. 图形接口部分
