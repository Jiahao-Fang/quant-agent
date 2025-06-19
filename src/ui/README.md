# Quant Factor Pipeline UI

这是一个交互式的Web界面，用于运行和监控量化因子构建管道。

## 功能特点

### 🎯 主要功能
1. **📝 因子请求输入**: 用自然语言描述你想要的量化因子
2. **📊 数据展示**: 交互式查看获取的市场数据
3. **💻 代码演化追踪**: 观察特征生成代码在调试周期中的演进
4. **📈 结果分析**: 检查最终的特征表格及统计信息

### 🖼️ 界面布局

#### 左侧边栏
- **因子请求输入框**: 描述你想要的因子
- **运行管道按钮**: 启动整个处理流程
- **清除结果按钮**: 清空所有结果
- **管道状态显示**: 实时显示各阶段的进度

#### 主要内容区域

##### 1. 📊 数据获取结果窗口
- 显示通过查询获取的KDB数据库表格
- **多表格支持**: 如果有多个数据源，每个key对应一个标签页
- **数据预览**: 显示每个表格的前10行
- **列信息**: 可展开查看列的详细信息（数据类型、空值统计等）

##### 2. 💻 特征代码演化窗口
- **多标签显示**: 
  - `🔧 Original`: 最初生成的代码
  - `✅ Retry X`: 成功的重试版本
  - `❌ Retry X`: 失败的重试版本
- **代码展示**: 语法高亮的Python代码
- **调试信息**: 
  - 执行状态（成功/失败/处理中）
  - 错误信息
  - 执行步骤
  - 变量信息

##### 3. 📈 最终特征表格窗口
- **基本统计**: 行数、列数、特征数量
- **数据预览**: 前20行数据
- **列统计**: 可展开的统计信息（均值、标准差等）
- **数据类型**: 显示每列的数据类型和空值比例

## 🚀 使用方法

### 1. 启动UI
```bash
cd src/raw_version/ui
python run_ui.py
```

或者直接使用Streamlit：
```bash
streamlit run factor_pipeline_ui.py
```

### 2. 访问界面
打开浏览器访问: `http://localhost:8501`

### 3. 使用流程
1. 在左侧边栏的文本框中输入因子描述
2. 点击"🚀 Run Pipeline"按钮
3. 观察左侧状态栏显示的进度
4. 查看主要区域的三个窗口中的结果

## 📝 示例因子请求

### 动量因子
```
Create a momentum factor based on BTCUSDT 5-minute price changes, 
smoothed with a 20-period rolling average
```

### 波动率因子
```
Build a volatility factor using ETHUSDT hourly returns 
with 14-period standard deviation
```

### 均值回归因子
```
Generate a mean reversion factor for BTCUSDT using 
price deviation from 30-period moving average
```

## 🔧 技术细节

### 依赖项
- `streamlit`: Web界面框架
- `pandas`: 数据处理
- `pykx`: KDB+数据库接口
- `numpy`: 数值计算

### 状态管理
UI使用Streamlit的session state来管理：
- `pipeline_state`: 管道各阶段的状态
- `pipeline_running`: 管道运行状态
- `pipeline_results`: 最终结果
- `code_history`: 代码演化历史

### 数据流
1. 用户输入 → 管道执行
2. 实时状态更新 → 界面刷新
3. 结果展示 → 交互式分析

## 🎨 自定义样式

界面使用自定义CSS样式，包括：
- 专业的配色方案
- 响应式布局
- 状态指示器
- 交互式组件

## 🐛 故障排除

### 常见问题

1. **导入错误**: 确保所有依赖项已安装
2. **数据显示错误**: 检查KDB+连接
3. **代码执行失败**: 查看调试信息中的错误详情

### 日志查看
运行时的详细日志会显示在终端中，包括：
- 管道执行进度
- 代码验证结果
- 错误信息

## 📊 性能优化

- 数据表格只显示前10-20行以提高加载速度
- 使用Streamlit的缓存机制减少重复计算
- 异步执行避免界面卡顿 