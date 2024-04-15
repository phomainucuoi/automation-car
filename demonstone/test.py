#this code draw raw data collect from real car testing

import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

loca_x = []
loca_y = []
goal_x = []
goal_y = []
node_numbers = []

# Mở tệp JSON và đọc dữ liệu từng dòng
with open('locadata.json', 'r') as file:
    for line in file:
        # Phân tích mỗi dòng là một đối tượng JSON
        data = json.loads(line)
        
        # Truy xuất và in ra các giá trị từ các khóa trong đối tượng JSON
        if "map_pos" in data:
            map_pos = data["map_pos"]
         

        elif "loca_x" in data and "loc_y" in data and "yaw" in data and "goal_x" in data and "goal_y" in data:
            loca_x.append(data["loca_x"])
            loca_y.append(data["loc_y"])
            goal_x.append(data["goal_x"])
            goal_y.append(data["goal_y"])
            yaw = data["yaw"]
            # Lưu chỉ số của node
            node_numbers.append(len(loca_x))
            node_numbers.append(len(loca_x)) # Append twice for position node and corresponding goal node
          
        else:
            print("Không có dữ liệu hợp lệ trong dòng này")

x_values = [point[0] for point in map_pos]
y_values = [point[1] for point in map_pos]

# Vẽ biểu đồ và nối các điểm lại với nhau
plt.plot(x_values, y_values, marker='o')

# Đặt nhãn cho trục x và y
plt.xlabel('X')
plt.ylabel('Y')

# Đặt tiêu đề cho biểu đồ
plt.title('BFMC')

# Vẽ các điểm và vị trí mục tiêu
for i in range(len(loca_x)):
    plt.scatter(loca_x[i], loca_y[i], color='red')
    plt.scatter(goal_x[i], goal_y[i], color='black')
    circle = Circle((loca_x[i], loca_y[i]), 0.25, color='blue', fill=False)
    plt.gca().add_patch(circle)
    # Đánh số node
    plt.text(loca_x[i], loca_y[i], str(node_numbers[i]), color='blue', fontsize=12, ha='center')
    plt.text(goal_x[i], goal_y[i], str(node_numbers[i+1]), color='black', fontsize=12, ha='center')

plt.show()
