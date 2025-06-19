# 🚀 Quant Factor Pipeline UI - 快速启动指南

## 🎯 新功能：实时进度显示

### ⚡ 实时更新特性
- **实时步骤显示**: 每个pipeline步骤完成后立即显示结果
- **动态代码展示**: 代码生成和调试过程实时可见
- **即时数据预览**: 数据获取完成后立即显示表格
- **进度条追踪**: 可视化显示整个pipeline的执行进度

## 1️⃣ 环境准备

### 安装依赖
```bash
pip install streamlit pandas numpy pykx
```

### 检查Python路径
确保项目根目录在Python路径中：
```bash
export PYTHONPATH="${PYTHONPATH}:."
# 或 Windows PowerShell:
$env:PYTHONPATH = "$env:PYTHONPATH;."
```

### 测试环境（可选）
```bash
cd src/raw_version/ui
python test_ui.py
```

## 2️⃣ 启动UI

### 方法1：使用启动脚本（推荐）
```bash
cd src/raw_version/ui
python run_ui.py
```

### 方法2：直接使用Streamlit
```bash
cd src/raw_version/ui
streamlit run factor_pipeline_ui.py
```

## 3️⃣ 访问界面

打开浏览器访问：http://localhost:8501

## 4️⃣ 实时体验流程

### 输入示例因子请求：
```
Create a momentum factor based on BTCUSDT 5-minute price changes, 
smoothed with a 20-period rolling average
```

### 点击"🚀 Run Pipeline"

### 观察实时执行过程：

1. **🔄 Step 1: 需求分析** 
   - 实时显示: 因子描述的分析结果
   - 可展开查看: 详细的特征描述

2. **🔄 Step 2: 数据获取**
   - 实时显示: 生成的查询语句
   - 立即展示: 获取的数据表格预览
   - 实时更新: 数据统计信息

3. **🔄 Step 3: 特征构建**
   - 实时显示: 初始代码生成
   - 动态更新: 每次代码调试过程
   - 即时反馈: 验证结果和错误信息
   - 最终展示: 成功的特征表格

### 查看结果标签页：
- **🔍 Query**: 生成的查询详情和参数
- **📊 Data**: 获取的KDB数据表格（支持多表切换）
- **💻 Code Evolution**: 代码演化的完整历史
- **📈 Feature Table**: 最终的特征表格和统计

## 🔧 故障排除

### 导入错误
如果出现`ModuleNotFoundError`：
1. 运行测试脚本: `python test_ui.py`
2. 检查PYTHONPATH设置
3. 确保在正确的目录运行命令
4. 安装缺失的依赖包

### 实时更新不工作
1. 确保使用最新版本的Streamlit
2. 检查浏览器控制台是否有JavaScript错误
3. 尝试刷新页面

### 端口占用
如果8501端口被占用：
```bash
streamlit run factor_pipeline_ui.py --server.port 8502
```

### 权限问题
Windows用户可能需要以管理员身份运行PowerShell

## 📱 界面预览

### 实时执行过程
启动后你会看到：
- **左侧边栏**: 输入框、控制按钮、实时步骤状态
- **执行区域**: 动态进度条和步骤详情
- **结果标签页**: 四个主要结果展示区域

### 左侧边栏状态指示器
- ✅ **已完成**: 绿色勾号表示步骤完成
- 🔄 **进行中**: 旋转图标表示正在执行
- ⏳ **等待中**: 时钟图标表示尚未开始

## 💡 实时功能小贴士

### 最佳体验建议：
- **保持页面激活**: 确保浏览器标签页处于前台
- **避免刷新**: 执行过程中不要刷新页面
- **观察进度**: 注意左侧边栏的实时状态更新
- **查看详情**: 点击展开按钮查看每步的详细信息

### 调试技巧：
- **实时错误**: 代码错误会立即显示在执行区域
- **步骤追踪**: 可在"详细步骤状态"中查看所有步骤
- **代码历史**: "Code Evolution"标签记录了所有代码版本

### 性能优化：
- 大型数据集加载可能需要时间，请耐心等待
- 复杂因子的代码调试可能需要多次迭代
- 可以随时点击"🗑️ Clear Results"重新开始

## 🎯 新功能总结

与之前版本相比，新的UI提供了：

1. **📊 四个专门的结果标签页**，更好地组织信息
2. **⚡ 实时进度更新**，可以看到每个步骤的执行过程
3. **🔍 查询详情展示**，了解数据获取的具体参数
4. **🔄 动态代码演化**，观察AI如何逐步完善代码
5. **📈 增强的状态追踪**，清晰显示每个组件的执行状态

这些改进让整个量化因子开发过程变得透明和可控！ 