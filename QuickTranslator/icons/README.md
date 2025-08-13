# 图标文件说明

此目录包含QuickTranslator插件的图标文件。

## 图标转换说明

由于浏览器插件需要不同尺寸的PNG图标文件，您需要将`icon.svg`转换为以下尺寸的PNG文件：

- icon16.png (16x16像素)
- icon48.png (48x48像素)
- icon128.png (128x128像素)

## 转换方法

您可以使用以下方法将SVG转换为PNG：

### 方法1：使用在线转换工具
1. 访问在线SVG转PNG工具（如 https://convertio.co/svg-png/ 或 https://cloudconvert.com/svg-to-png）
2. 上传icon.svg文件
3. 选择需要的尺寸（16x16, 48x48, 128x128）
4. 下载转换后的PNG文件并重命名为相应的名称

### 方法2：使用图像编辑软件
1. 使用Adobe Illustrator、Inkscape或其他支持SVG的图像编辑软件打开icon.svg
2. 导出为PNG格式，分别设置尺寸为16x16、48x48和128x128
3. 保存为相应的文件名

### 方法3：使用命令行工具（如ImageMagick）
如果您安装了ImageMagick，可以使用以下命令：

```bash
# 转换为16x16
convert icon.svg -resize 16x16 icon16.png

# 转换为48x48
convert icon.svg -resize 48x48 icon48.png

# 转换为128x128
convert icon.svg -resize 128x128 icon128.png
```

## 图标设计说明

图标设计采用了以下元素：
- 蓝色圆形背景（#4285f4，Google蓝色）
- 两个对话气泡，表示翻译功能
- 气泡中的文本线条，表示文本内容

这个设计简洁明了，能够直观地表达翻译插件的功能。