"""
使用示例脚本

展示如何使用图像处理实验框架进行基本的图像处理实验。
"""

import cv2
import numpy as np
from loaders import load_image, load_images
import processors
from pipeline import ImageLab
from visualizer import show_results, show_grid



def analyze_connected_components(img: np.ndarray, lab=None) -> np.ndarray:
    """
    连通组件分析复合函数

    对二值图像进行连通组件分析，返回彩色标签图像用于可视化
    每个组件用不同颜色显示，同时打印统计信息
    """
    try:
        labels, stats, centroids = processors.connectedComponentsWithStats(img)

        # 保存连通域信息供后续使用
        if lab is not None:
            lab.save_intermediate_result('cc_labels', labels)
            lab.save_intermediate_result('cc_stats', stats)
            lab.save_intermediate_result('cc_centroids', centroids)
            lab.save_intermediate_result('cc_binary_img', img)

        # 打印统计信息
        num_components = stats.shape[0]
        print(f"  找到 {num_components} 个连通组件")

        # 打印前几个组件的信息（如果有的话）
        if num_components > 1:  # 第一个通常是背景
            for i in range(1, min(4, num_components)):  # 显示前3个组件
                area = stats[i, cv2.CC_STAT_AREA]
                left = stats[i, cv2.CC_STAT_LEFT]
                top = stats[i, cv2.CC_STAT_TOP]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]

                print(f"    组件{i}: 面积={area}, 位置=({left},{top}), "
                      f"尺寸={width}x{height}, 质心=({cx:.1f},{cy:.1f})")

        # 创建彩色可视化图像
        if num_components <= 1:
            # 只有一个组件（背景），返回黑色图像
            colored_labels = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
        else:
            # 为每个组件分配随机颜色
            colored_labels = np.zeros((*labels.shape, 3), dtype=np.uint8)

            # 为背景（标签0）设置黑色
            colored_labels[labels == 0] = [0, 0, 0]

            # 为其他组件分配随机颜色
            np.random.seed(42)  # 保证颜色一致性
            colors = np.random.randint(50, 255, (num_components, 3), dtype=np.uint8)
            # 确保背景是黑色
            colors[0] = [0, 0, 0]

            for i in range(1, num_components):
                colored_labels[labels == i] = colors[i]

        return colored_labels

    except Exception as e:
        print(f"  连通组件分析失败: {e}")
        return img.copy()


def main():
    """
    主函数：演示基本使用流程
    """
    print("=" * 60)
    print("图像处理实验框架 - 使用示例")
    print("=" * 60)
    
    # # 示例1: 单张图片处理
    # print("\n示例1: 单张图片加载和处理")
    # print("-" * 60)
    
    # try:
    #     # 注意：这里使用示例路径，实际使用时请替换为您的图片路径
    #     # img_path = "test.jpg"  # 取消注释并替换为实际路径
    #     # img = load_image(img_path)
        
    #     # 为了演示，我们创建一个示例图片
    #     print("创建示例图片...")
    #     img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    #     print(f"图片形状: {img.shape}")
        
    #     # 创建ImageLab实例
    #     lab = ImageLab()
        
    #     # 定义处理步骤
    #     funcs = [
    #         ("Grayscale", simple_grayscale, {}),
    #         ("Blur", simple_blur, {"ksize": 5}),
    #     ]
        
    #     # 运行流水线
    #     print("运行处理流水线...")
    #     lab.run_pipeline(img, funcs)
        
    #     # 获取结果并可视化
    #     results = lab.get_results()
    #     print(f"流水线包含 {len(results)} 个步骤")
    #     print("步骤列表:", [name for name, _ in results])
        
    #     # 显示结果
    #     print("显示处理结果...")
    #     show_results(results, title="图像处理流水线示例")
        
    # except Exception as e:
    #     print(f"错误: {e}")
    #     print("\n提示: 如果您有实际的图片文件，可以取消注释代码中的图片加载部分")
    
    # # 示例2: 参数扫描（Grid Search）
    # print("\n\n示例2: 参数扫描对比")
    # print("-" * 60)
    
    # try:
    #     # 创建示例灰度图
    #     test_img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        
    #     # 测试不同模糊参数
    #     results = [("Original", test_img)]
    #     for ksize in [3, 5, 7, 9]:
    #         blurred = simple_blur(test_img, ksize=ksize)
    #         results.append((f"Blur ksize={ksize}", blurred))
        
    #     # 使用网格显示
    #     print("显示参数扫描结果（2x3网格）...")
    #     show_grid(results, rows=2, cols=3, title="模糊参数对比")
        
    # except Exception as e:
    #     print(f"错误: {e}")

    # 示例3: 文件夹批量处理
    print("\n\n示例3: 文件夹批量处理")
    print("-" * 60)

    try:
        # 注意：这里使用示例路径，实际使用时请替换为您的图片文件夹路径
        folder_path = r"C:\Users\13538\Desktop\pcba\alg\roi_pic"  

        print(f"加载文件夹: {folder_path}")
        image_generator = load_images(folder_path)

        # 处理前几张图片作为示例
        processed_count = 0
        max_examples = 10  # 最多处理3张图片作为示例

        for filename, img in image_generator:
            if processed_count >= max_examples:
                break

            print(f"\n处理图片: {filename}, 形状: {img.shape}")

            # 创建新的ImageLab实例处理每张图片
            lab = ImageLab()

            # 创建包装函数来传递lab参数
            def analyze_cc(img):
                return analyze_connected_components(img, lab)

            def filter_cc(img):
                return processors.filter_connected_components(img, lab,
                    filter_type="most_circular", top_n=3)

            # 定义处理步骤（连通域筛选演示）
            funcs = [
                ("Grayscale", processors.to_grayscale, {}),
                ("Blur", processors.gaussian_blur, {"ksize": 5}),
                ("Otsu", processors.otsu_threshold, {}),
                ("Connected Components", analyze_cc, {}),
                ("Filter Most Circular", filter_cc, {}),
            ]

            # 运行流水线
            lab.run_pipeline(img, funcs)
            results = lab.get_results()

            # 显示结果
            print(f"  流水线包含 {len(results)} 个步骤")
            show_results(results, title=f"处理结果 - {filename}")
            processed_count += 1

        print(f"\n共处理了 {processed_count} 张图片")

    except Exception as e:
        print(f"错误: {e}")
        print("\n提示: 请确保文件夹路径正确，且包含支持的图片格式(.jpg, .png, .bmp等)")
        print("您可以通过修改 folder_path 变量来指定实际的图片文件夹路径")

    print("\n" + "=" * 60)
    print("示例演示完成！")
    print("=" * 60)
    print("\n使用提示:")
    print("1. 在 processors.py 中添加您的算法函数")
    print("2. 使用 load_image() 加载单张图片")
    print("3. 使用 load_images() 批量处理文件夹中的图片（生成器模式，支持大批量处理）")
    print("4. 使用 ImageLab 类管理处理流水线")
    print("5. 使用 show_results() 或 show_grid() 可视化结果")
    print("6. 要加载实际文件夹，只需修改示例3中的 folder_path 变量")


if __name__ == "__main__":
    main()
