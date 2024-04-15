import json

# Đọc dữ liệu từ tệp JSON
with open('locadata.json', 'r') as file:
    du_lieu = json.load(file)

# In dữ liệu đã đọc
print(du_lieu)
