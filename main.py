
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import Dict, List
# import tqdm


class HeightDataLoader:
    def __init__(self):
        self.data = None
        self.extent = None
        self.resolution = None
        self.projection = None

    def load(self, folder_path):
        """加载Heights文件夹数据"""
        index_info = self._parse_index(folder_path)
        self.projection = self._parse_projection(folder_path)
        self._load_binary_data(folder_path, index_info)

    def _parse_index(self, folder_path):
        with open(os.path.join(folder_path, "index.txt"), 'r') as f:
            parts = f.readline().strip().split()
            return {
                'file_name': parts[0],
                'east_min': float(parts[1]),
                'east_max': float(parts[2]),
                'north_min': float(parts[3]),
                'north_max': float(parts[4]),
                'resolution': float(parts[5])
            }

    def _parse_projection(self, folder_path):
        with open(os.path.join(folder_path, "projection.txt"), 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            return {
                'spheroid': lines[0],
                'zone': lines[1] if len(lines) > 1 else None,
                'projection': lines[2] if len(lines) > 2 else None,
                'central_meridian': lines[3].split() if len(lines) > 3 else None
            }

    def _load_binary_data(self, folder_path, index_info):
        """加载二进制高度数据"""
        with open(os.path.join(folder_path, index_info['file_name']), 'rb') as f:
            rows = int(round((index_info['north_max'] - index_info['north_min']) / index_info['resolution']))
            cols = int(round((index_info['east_max'] - index_info['east_min']) / index_info['resolution']))

            self.data = np.flipud(
                np.fromfile(f, dtype=np.int16).reshape(rows, cols)
            )
            self.extent = [
                index_info['east_min'],
                index_info['east_max'],
                index_info['north_min'],
                index_info['north_max']
            ]
            self.resolution = index_info['resolution']


class ClutterDataLoader:
    def __init__(self):
        self.data = None
        self.extent = None
        self.resolution = None
        self.menu = None

    def load(self, folder_path):
        """加载Clutter文件夹数据"""
        index_info = self._parse_index(folder_path)
        self.menu = self._parse_menu(folder_path)
        self._load_binary_data(folder_path, index_info)

    def _parse_index(self, folder_path):
        with open(os.path.join(folder_path, "index.txt"), 'r') as f:
            parts = f.readline().strip().split()
            return {
                'file_name': parts[0],
                'east_min': float(parts[1]),
                'east_max': float(parts[2]),
                'north_min': float(parts[3]),
                'north_max': float(parts[4]),
                'resolution': float(parts[5])
            }

    def _parse_menu(self, folder_path):
        menu = {}
        with open(os.path.join(folder_path, "menu.txt"), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    code = int(parts[0])
                    name = ' '.join(parts[1:])
                    menu[code] = name
        return menu

    def _load_binary_data(self, folder_path, index_info):
        """加载二进制地貌数据"""
        with open(os.path.join(folder_path, index_info['file_name']), 'rb') as f:
            rows = int(round((index_info['north_max'] - index_info['north_min']) / index_info['resolution']))
            cols = int(round((index_info['east_max'] - index_info['east_min']) / index_info['resolution']))

            self.data = np.flipud(
                np.fromfile(f, dtype=np.int16).reshape(rows, cols))
            self.extent = [
                index_info['east_min'],
                index_info['east_max'],
                index_info['north_min'],
                index_info['north_max']
            ]
            self.resolution = index_info['resolution']

    def get_clutter_colormap(self):
        """优化的颜色映射"""
        return {
            0: (0, 0, 0, 0),  # Nodata (透明)
            1: (0, 0.5, 1, 0.6),  # Water (浅蓝色)
            2: (0, 0.3, 0.8, 0.7),  # Sea (深蓝色)
            3: (0.4, 0.8, 0.4, 0.6),  # Wet_land (浅绿色)
            4: (0.8, 0.8, 0.6, 0.7),  # Suburban_Open (米色)
            5: (0.9, 0.9, 0.9, 0.7),  # Urban_Open (浅灰色)
            6: (0.2, 0.8, 0.2, 0.7),  # Green_Land (绿色)
            7: (0.1, 0.5, 0.1, 0.7),  # Forest (深绿色)
            8: (0.8, 0.2, 0.2, 0.5),  # High_Building (半透明红色)
            9: (0.7, 0.2, 0.2, 0.5),  # Ordinary_Building
            10: (0.6, 0.2, 0.2, 0.5),  # Parallel_Building
            11: (0.8, 0.4, 0.2, 0.5),  # Irregular_Large
            12: (0.7, 0.5, 0.3, 0.5),  # Irregular
            13: (0.6, 0.4, 0.2, 0.5)  # Suburban_Village
        }


class BuildVectorLoader:
    def __init__(self):
        self.building_data = []

    def load_single_buildvector(self, bv_path, attr_path):
        """加载单个BuildVector和Attribute文件对"""
        buildings = []
        heights = []

        # 读取BuildVector文件
        try:
            with open(bv_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]

            i = 0
            while i < len(lines):
                parts = lines[i].split()
                if len(parts) < 3:
                    i += 1
                    continue

                building_id = int(parts[0])
                num_points = int(parts[-1])

                # 读取点坐标
                points = []
                for j in range(i + 1, i + 1 + num_points):
                    if j >= len(lines):
                        break
                    coords = list(map(float, lines[j].split()))
                    if len(coords) >= 2:
                        points.append((coords[0], coords[1]))

                if len(points) > 2:
                    buildings.append((building_id, points))
                i += num_points + 1
        except Exception as e:
            print(f"Error reading {bv_path}: {str(e)}")
            return []

        # 读取Attribute文件获取高度
        try:
            with open(attr_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        building_id = int(parts[0])
                        height = float(parts[2])
                        heights.append((building_id, height))
        except Exception as e:
            print(f"Error reading {attr_path}: {str(e)}")
            return []

        # 合并数据
        building_data = []
        for bid, points in buildings:
            height = next((h for b, h in heights if b == bid), 0.0)
            building_data.append({
                'id': bid,
                'points': points,
                'height': height
            })

        return building_data

    def load_all_buildvectors(self, folder_path):
        """加载文件夹下所有BuildVector和Attribute文件对"""
        all_buildings = []

        # 自动发现匹配的文件对
        bv_files = sorted([f for f in os.listdir(folder_path) if f.startswith('buildings')])
        attr_files = sorted([f for f in os.listdir(folder_path) if f.startswith('attribute')])

        file_pairs = min(len(bv_files), len(attr_files))

        for i in range(file_pairs):
            bv_path = os.path.join(folder_path, bv_files[i])
            attr_path = os.path.join(folder_path, attr_files[i])

            print(f"Loading file pair: {bv_files[i]} + {attr_files[i]}")
            buildings = self.load_single_buildvector(bv_path, attr_path)
            all_buildings.extend(buildings)
            print(f"Loaded {len(buildings)} buildings from this pair")

        self.building_data = all_buildings
        return all_buildings

class CombinedVisualizer:
    def __init__(self):
        self.height_loader = HeightDataLoader()
        self.clutter_loader = ClutterDataLoader()
        self.buildvector_loader = BuildVectorLoader()
        self.df = None  # 网络覆盖数据

    def load_data(self, heights_path: str, clutter_path: str, buildvector_path: str):
        """加载所有基础数据"""
        self.height_loader.load(heights_path)
        self.clutter_loader.load(clutter_path)
        self.buildvector_loader.load_all_buildvectors(buildvector_path)
        self._validate_data_alignment()

    def load_coverage_data(self, coverage_path: str):
        """加载网络覆盖数据"""
        self.df = pd.read_csv(coverage_path)
        print(f"加载覆盖数据点: {len(self.df)}个")

    def _validate_data_alignment(self):
        """验证数据空间一致性"""
        print(f"高程数据维度: {self.height_loader.data.shape}")
        print(f"地貌数据维度: {self.clutter_loader.data.shape}")
        print(f"建筑物数量: {len(self.buildvector_loader.building_data)}")

    def plot_combined_2d(self, downsample: int = 5) -> tuple:
        """
        绘制2D组合视图

        参数:
            downsample: 降采样因子

        返回:
            (fig, ax) matplotlib图形和坐标轴对象
        """
        fig, ax = plt.subplots(figsize=(15, 12))

        # 统一数据维度
        min_rows = min(self.height_loader.data.shape[0],
                       self.clutter_loader.data.shape[0])
        min_cols = min(self.height_loader.data.shape[1],
                       self.clutter_loader.data.shape[1])

        h_data = self.height_loader.data[:min_rows:downsample, :min_cols:downsample]
        c_data = self.clutter_loader.data[:min_rows:downsample, :min_cols:downsample]

        x = np.linspace(
            self.height_loader.extent[0],
            self.height_loader.extent[0] + min_cols * self.height_loader.resolution,
            h_data.shape[1]
        )
        y = np.linspace(
            self.height_loader.extent[2],
            self.height_loader.extent[3],
            h_data.shape[0]
        )
        X, Y = np.meshgrid(x, y)

        # 绘制地貌类型
        colors = self.clutter_loader.get_clutter_colormap()
        legend_elements = []
        for code, color in colors.items():
            if code == 0:
                continue
            mask = (c_data == code)
            if mask.any():
                ax.scatter(X[mask], Y[mask], c=[color], s=1, label='_nolegend_')
                legend_elements.append(
                    Patch(facecolor=color,
                          edgecolor='k',
                          label=f"{code}: {self.clutter_loader.menu.get(code, str(code))}")
                )

        # 绘制高程等高线
        contour = ax.contour(X, Y, h_data, levels=15, colors='k',
                             linewidths=0.5, alpha=0.5, linestyles='dashed')
        ax.clabel(contour, inline=True, fontsize=8)

        # 绘制建筑物
        for building in self.buildvector_loader.building_data:
            polygon = Polygon(building['points'], closed=True,
                              facecolor=(0.6, 0.2, 0.8, 0.5),
                              edgecolor='darkred', linewidth=0.8)
            ax.add_patch(polygon)

        # 添加建筑物图例
        legend_elements.append(
            Patch(facecolor=(0.6, 0.2, 0.8, 0.5), edgecolor='darkred',
                  label='Buildings')
        )

        # 添加网络覆盖
        if self.df is not None:
            coverage_sample = self.df.sample(min(5000, len(self.df)))
            sc = ax.scatter(
                coverage_sample['x'],
                coverage_sample['y'],
                c=coverage_sample['RSRP'],
                cmap='jet',
                s=1,
                alpha=0.7,
                vmin=-200,
                vmax=-80
            )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.5)
            cbar.set_label('RSRP (dBm)')
            legend_elements.append(
                Patch(facecolor='red', edgecolor='k', label='Coverage')
            )

        if legend_elements:
            ax.legend(handles=legend_elements,
                      bbox_to_anchor=(1.05, 1),
                      loc='upper left')

        ax.set_xlim(self.height_loader.extent[0],
                    self.height_loader.extent[0] + min_cols * self.height_loader.resolution)
        ax.set_ylim(self.height_loader.extent[2],
                    self.height_loader.extent[3])
        ax.set_aspect('equal')
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_title('2D View (Terrain + Buildings + Coverage)')
        ax.grid(True, alpha=0.3)

        return fig, ax

    def plot_interactive_3d(self, z_scale=0.5, downsample=10, coverage_alpha=0.7, max_buildings=500):
        """
        创建交互式3D绘图（显示所有建筑物）

        参数:
            z_scale: 高度缩放因子（建议0.3-1.0）
            downsample: 地形数据降采样因子（1=不降采样）
            coverage_alpha: 网络覆盖透明度（0-1）
            max_buildings: 最大建筑物数量（性能优化）
        """
        fig = go.Figure()

        # ========== 1. 准备数据 ==========
        min_rows = min(self.height_loader.data.shape[0],
                       self.clutter_loader.data.shape[0])
        min_cols = min(self.height_loader.data.shape[1],
                       self.clutter_loader.data.shape[1])

        h_data = self.height_loader.data[:min_rows:downsample, :min_cols:downsample]
        c_data = self.clutter_loader.data[:min_rows:downsample, :min_cols:downsample]

        # 创建网格坐标
        x = np.linspace(
            self.height_loader.extent[0],
            self.height_loader.extent[0] + min_cols * self.height_loader.resolution,
            h_data.shape[1]
        )
        y = np.linspace(
            self.height_loader.extent[2],
            self.height_loader.extent[3],
            h_data.shape[0]
        )
        X, Y = np.meshgrid(x, y)
        Z = h_data * z_scale

        # ========== 2. 添加地形表面 ==========
        terrain_surface = go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=c_data,
            colorscale=self._get_clutter_colorscale(),
            cmin=1,
            cmax=13,
            showscale=True,
            name='Terrain',
            opacity=0.9,
            hoverinfo='skip',
            colorbar=dict(
                title='Terrain Type',
                tickvals=list(self.clutter_loader.menu.keys()),
                ticktext=list(self.clutter_loader.menu.values())
            )
        )
        fig.add_trace(terrain_surface)

        # ========== 3. 添加所有建筑物 ==========
        # 按高度排序建筑物（优先显示高层建筑）
        sorted_buildings = sorted(
            self.buildvector_loader.building_data,
            key=lambda x: x['height'],
            reverse=True
        )

        # 限制建筑物数量
        buildings_to_show = sorted_buildings[:max_buildings]
        print(f"渲染 {len(buildings_to_show)}/{len(sorted_buildings)} 个建筑物 (最高{max_buildings}个)...")

        # # 创建进度条
        # progress_bar = tqdm.tqdm(self.buildvector_loader.building_data, desc="渲染建筑物")

        for building in buildings_to_show:
            try:
                fig.add_trace(self._create_building_mesh(building, z_scale))
            except Exception as e:
                print(f"渲染建筑物{building['id']}时出错: {str(e)}")
                continue

        # progress_bar.close()

        # ========== 4. 添加网络覆盖 ==========
        if self.df is not None:
            # 自动计算合适的采样率保持性能
            sample_rate = min(1.0, 50000 / len(self.df))
            coverage_sample = self.df.sample(frac=sample_rate) if sample_rate < 1.0 else self.df

            coverage = go.Scatter3d(
                x=coverage_sample['x'],
                y=coverage_sample['y'],
                z=coverage_sample['h'] * z_scale,
                mode='markers',
                marker=dict(
                    size=4,
                    color=coverage_sample['RSRP'],
                    colorscale='Jet',
                    cmin=-120,
                    cmax=-80,
                    opacity=coverage_alpha,
                    colorbar=dict(
                        title='RSRP (dBm)',
                        x=1.1
                    )
                ),
                name='Coverage',
                hoverinfo='skip'
            )
            fig.add_trace(coverage)

        # ========== 5. 优化布局和性能 ==========
        fig.update_layout(
            margin=dict(t=100),
            title={
                'text': f'3D View - 显示{len(buildings_to_show)}栋建筑物 (共{len(sorted_buildings)}栋)',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            scene=dict(
                xaxis_title='East (m)',
                yaxis_title='North (m)',
                zaxis_title='Height (m)',
                aspectratio=dict(x=1, y=1, z=0.3),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8)
                ),
                # 优化渲染性能
                xaxis=dict(showspikes=False),
                yaxis=dict(showspikes=False),
                zaxis=dict(showspikes=False)
            ),
            height=900,
            # margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # 防止UI重置
            uirevision='constant'
        )

        # 添加图层可见性控制
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True, True, True]}],
                            label="显示全部",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [True, False, True]}],
                            label="隐藏建筑物",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [True, True, False]}],
                            label="隐藏覆盖",
                            method="update"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )

        return fig

    def _get_clutter_colorscale(self) -> list:
        """创建符合Plotly要求的颜色映射"""
        colors = self.clutter_loader.get_clutter_colormap()

        # 获取有效的地貌类型代码(排除0)
        valid_codes = sorted([c for c in colors.keys() if c != 0])
        if not valid_codes:
            return [[0, 'rgb(200,200,200)'], [1, 'rgb(200,200,200)']]

        # 计算每个颜色对应的位置(0-1之间)
        min_code, max_code = min(valid_codes), max(valid_codes)
        code_range = max_code - min_code

        colorscale = []
        for code in valid_codes:
            rgba = colors[code]
            rgb_str = f"rgb({int(rgba[0] * 255)},{int(rgba[1] * 255)},{int(rgba[2] * 255)})"

            # 归一化位置到0-1之间
            if code_range > 0:
                position = (code - min_code) / code_range
            else:
                position = 0.5

            colorscale.append([position, rgb_str])

        # 确保第一个位置是0，最后一个位置是1
        if colorscale[0][0] != 0:
            colorscale.insert(0, [0, colorscale[0][1]])
        if colorscale[-1][0] != 1:
            colorscale.append([1, colorscale[-1][1]])

        return colorscale

    def _create_building_mesh(self, building, z_scale):
        """创建单个建筑物的3D网格"""
        points = building['points']
        height = max(building['height'], 3.0) * z_scale  # 确保最小高度

        # 创建顶点
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        z_bottom = [0] * len(points)
        z_top = [height] * len(points)

        # 创建三角形索引
        triangles = []
        n = len(points)

        # 底面和顶面
        for i in range(1, n - 1):
            triangles.append([0, i, i + 1])  # 底面
            triangles.append([n, n + i, n + i + 1])  # 顶面

        # 侧面
        for i in range(n):
            next_i = (i + 1) % n
            triangles.append([i, next_i, n + next_i])
            triangles.append([i, n + next_i, n + i])

        i_tri, j_tri, k_tri = zip(*triangles)

        # 根据高度设置颜色
        height = building['height']
        if height > 60:
            color = 'rgba(200, 50, 50, 0.7)'  # 高层建筑-红色
        elif height > 30:
            color = 'rgba(180, 80, 180, 0.7)'  # 中层建筑-紫色
        else:
            color = 'rgba(80, 120, 200, 0.7)'  # 低层建筑-蓝色

        return go.Mesh3d(
            x=x + x,
            y=y + y,
            z=z_bottom + z_top,
            i=i_tri,
            j=j_tri,
            k=k_tri,
            color=color,
            flatshading=True,
            lighting=dict(
                ambient=0.3,
                diffuse=0.8,
                roughness=0.1
            ),
            name=f'Building {building["id"]}',
            hoverinfo='text',
            text=f'高度: {height:.1f}m\n面积: {self._calc_building_area(points):.0f}㎡',
            showlegend=False
        )

    def _calc_building_area(self, points):
        """计算建筑物多边形面积"""
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(points) - 1)))

    def _add_3d_legend(self, ax):
        """添加3D图例"""
        legend_elements = []
        clutter_colors = self.clutter_loader.get_clutter_colormap()
        # 添加地貌图例
        for code, color in clutter_colors.items():
            if code in self.clutter_loader.menu and code != 0:
                legend_elements.append(
                    Patch(facecolor=color,
                          edgecolor='k',
                          label=f"{code}: {self.clutter_loader.menu[code]}")
                )

        # 添加建筑物图例
        legend_elements.append(
            Patch(facecolor=(0.6, 0.2, 0.8, 0.6), edgecolor='darkred',
                  label='Buildings')
        )
        # 添加RSRP图例
        if hasattr(self, 'df'):
            legend_elements.append(
                Patch(facecolor='red',
                      edgecolor='k',
                      label='RSRP Coverage')
            )

        if legend_elements:
            ax.legend(handles=legend_elements,
                      bbox_to_anchor=(1.05, 1),
                      loc='upper left')


if __name__ == "__main__":
    # 初始化可视化器
    visualizer = CombinedVisualizer()

    # 加载数据
    heights_path = r"D:\ISAC\chizhou\Height"
    clutter_path = r"D:\ISAC\chizhou\Clutter"
    buildvector_path = r"D:\ISAC\chizhou\Buildvector"

    visualizer.load_data(heights_path, clutter_path, buildvector_path)

    # 加载网络覆盖数据
    coverage_path = r"D:\ISAC\chizhou\pred\prediction.csv"
    visualizer.load_coverage_data(coverage_path)

    # 绘制2D组合视图
    fig_2d, ax_2d = visualizer.plot_combined_2d(downsample=5)
    plt.tight_layout()
    plt.show()

    # 创建并显示交互式3D视图
    fig = visualizer.plot_interactive_3d(
        z_scale=0.5,  # 高度缩放因子
        downsample=10,  # 地形降采样率
        coverage_alpha=0.7,  # 覆盖透明度
        max_buildings=1000
    )
    fig.show()