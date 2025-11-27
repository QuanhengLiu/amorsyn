import os
from ruamel.yaml import YAML


def single_param_modify():
    yaml = YAML()
    yaml.preserve_quotes = True  # 保持引号格式
    yaml.width = 4096  # 避免长行换行

    base_path = "/home/liu/桌面/amorsyn/data/carbon-fiber/"
    input_file = os.path.join(base_path, "00000_ground_59.yaml")

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在！")
        return

    # 存储生成的文件路径
    generated_files = []

    # 参数配置：参数名、文件前缀、起始值、步长、数量
    param_configs = [
        {
            'param_name': 'ropeSpringStiffness',
            'file_prefix': 'stiff',
            'start_value': 15.0,
            'step': 0.1,
            'count': 50,
            'description': '绳索弹簧刚度'
        },
        {
            'param_name': 'printerFrictionCoefficient',
            'file_prefix': 'fric',
            'start_value': 1.2,
            'step': 0.004,
            'count': 100,
            'description': '打印机摩擦系数'
        },
        {
            'param_name': 'printerMass',
            'file_prefix': 'mass',
            'start_value': 800,
            'step': 4,
            'count': 100,
            'description': '打印机质量'
        },
        {
            'param_name': 'ropeIndividualMass',
            'file_prefix': 'indmass',
            'start_value': 0.9,
            'step': 0.01,
            'count': 20,
            'description': '绳索单位质量'
        },
        {
            'param_name': 'groundFrictionCoefficient',
            'file_prefix': 'ground',
            'start_value': 80.0,
            'step': 0.5,
            'count': 80,
            'description': '地面摩擦系数'
        }
    ]

    print(f"开始从 {input_file} 生成配置文件...")

    # 读取基础配置
    try:
        with open(input_file, 'r') as f:
            base_config = yaml.load(f)
        print(f"成功读取基础配置文件")
    except Exception as e:
        print(f"读取基础配置文件失败: {e}")
        return

    # 为每个参数生成配置文件
    for param_config in param_configs:
        param_name = param_config['param_name']
        file_prefix = param_config['file_prefix']
        start_value = param_config['start_value']
        step = param_config['step']
        count = param_config['count']
        description = param_config['description']

        print(f"\n正在生成 {description} ({param_name}) 的配置文件...")

        for i in range(count):
            # 计算实际参数值
            param_value = start_value + i * step

            # 生成输出文件名
            output_file = os.path.join(base_path, f"00000_{file_prefix}_{i}.yaml")

            # 复制基础配置并修改参数
            config = dict(base_config)  # 深拷贝配置
            config[param_name] = param_value

            # 写入新的配置文件
            try:
                with open(output_file, 'w') as f:
                    yaml.dump(config, f)

                # 生成对应的txt文件路径
                txt_file = output_file.replace('.yaml', '.txt')
                generated_files.append((output_file, txt_file))

                # 显示进度（每10个文件显示一次）
                if i % 10 == 0 or i == count - 1:
                    print(f"  已生成 {i + 1}/{count}: {os.path.basename(output_file)} (值: {param_value:.4f})")

            except Exception as e:
                print(f"  错误：生成文件 {output_file} 失败: {e}")

    # 生成list文件
    list_path = os.path.join(base_path, 'list')
    try:
        with open(list_path, 'w') as f:
            for yaml_path, txt_path in generated_files:
                f.write(f"{yaml_path},{txt_path}\n")
        print(f"\n成功生成list文件: {list_path}")
        print(f"包含 {len(generated_files)} 个配置文件对")
    except Exception as e:
        print(f"生成list文件失败: {e}")

    # 输出统计信息
    print(f"\n=== 生成统计 ===")
    print(f"总共生成配置文件: {len(generated_files)} 个")

    # 按参数类型统计
    param_stats = {}
    for param_config in param_configs:
        prefix = param_config['file_prefix']
        count = sum(1 for yaml_path, _ in generated_files if f"_{prefix}_" in yaml_path)
        param_stats[param_config['description']] = count

    for desc, count in param_stats.items():
        print(f"  {desc}: {count} 个文件")


def cleanup_generated_files():
    """清理生成的文件（可选功能）"""
    base_path = "/home/liu/桌面/amorsyn/data/carbon-fiber/"

    # 定义要清理的文件模式
    patterns = ['*_stiff_*.yaml', '*_fric_*.yaml', '*_mass_*.yaml',
                '*_indmass_*.yaml', '*_ground_*.yaml']

    import glob

    total_deleted = 0
    for pattern in patterns:
        files = glob.glob(os.path.join(base_path, pattern))
        for file in files:
            try:
                os.remove(file)
                total_deleted += 1
            except Exception as e:
                print(f"删除文件 {file} 失败: {e}")

    # 删除list文件
    list_file = os.path.join(base_path, 'list')
    if os.path.exists(list_file):
        try:
            os.remove(list_file)
            total_deleted += 1
        except Exception as e:
            print(f"删除list文件失败: {e}")

    print(f"已清理 {total_deleted} 个文件")


def verify_generated_files():
    """验证生成的文件（可选功能）"""
    base_path = "/home/liu/桌面/amorsyn/data/carbon-fiber/"
    list_path = os.path.join(base_path, 'list')

    if not os.path.exists(list_path):
        print("list文件不存在，无法验证")
        return

    yaml = YAML()
    valid_files = 0
    invalid_files = 0

    with open(list_path, 'r') as f:
        for line in f:
            yaml_path, txt_path = line.strip().split(',')

            # 检查yaml文件
            if os.path.exists(yaml_path):
                try:
                    with open(yaml_path, 'r') as yf:
                        config = yaml.load(yf)
                    valid_files += 1
                except Exception as e:
                    print(f"YAML文件 {yaml_path} 格式错误: {e}")
                    invalid_files += 1
            else:
                print(f"YAML文件不存在: {yaml_path}")
                invalid_files += 1

    print(f"验证结果: {valid_files} 个有效文件, {invalid_files} 个无效文件")


if __name__ == "__main__":
    # 执行修改
    single_param_modify()