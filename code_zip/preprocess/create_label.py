### 라벨을 생성하는 코드

path = 'd:\\train_label.csv' # 라벨을 만들 위치
file = open(path, 'w')

for i in  range(0,480):
    file.write(str(0)+'\n')
for i in  range(0,480):
    file.write(str(1)+'\n')
for i in  range(0,480):
    file.write(str(2)+'\n')
for i in  range(0,480):
    file.write(str(3)+'\n')
    
file.close()
