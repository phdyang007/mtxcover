import sys

cnt = []
with open(sys.argv[1] + "/col.txt", 'r') as f:
  for line in f:
    line = line.strip()
    if line:
      cnt.append(str(len(line.split())))
  
with open(sys.argv[1] + "/col_cnt.txt", "w") as f:
  f.write('\n'.join(cnt))

with open(sys.argv[1] + "/n_data.txt", "w") as f:
  f.write(str(len(cnt)))

row = []
with open(sys.argv[1] + "/matrix.txt", 'r') as f:
  pre_empty = True
  ccnt = 0
  for line in f:
    line = line.strip()
    if len(line) != 0:
        ccnt += 1
        pre_empty = False
    elif not pre_empty:
        row.append(str(ccnt))
        ccnt = 0
        pre_empty = True
  if ccnt != 0:
    row.append(str(ccnt))


with open(sys.argv[1] + "/row_cnt.txt", "w") as f:
  f.write('\n'.join(row))
