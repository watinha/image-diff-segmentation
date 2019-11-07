import sys, np, pandas as pd

from PIL import Image
from sklearn.cluster import KMeans

if len(sys.argv) < 2:
    print('first argument should be an image filename...')
    sys.exit(1)

if len(sys.argv) < 3:
    print('second argument should be an image filename...')
    sys.exit(1)

if len(sys.argv) < 4:
    print('third argument should the number of clusters for kmeans...')
    sys.exit(1)

if len(sys.argv) < 5:
    print('forth argument should the size of the image to be considered...')
    sys.exit(1)

f1 = sys.argv[1]
f2 = sys.argv[2]
n_clusters = int(sys.argv[3])
dim = int(sys.argv[4])

i1 = Image.open(f1).convert('L')
i2 = Image.open(f2).convert('L').resize(i1.size)
i1.save('output/screenshot-baseline.png')
i2.save('output/screeenshot-test.png')

diff = Image.new('RGBA', i1.size, (0, 0, 0, 0))
clustered_image = Image.new('RGBA', i1.size, (0, 0, 0, 0))
diffs = []

(width, height) = i1.size
for i in range(width):
    for j in range(height):
        pixel_i1 = i1.getpixel((i, j))
        pixel_i2 = i2.getpixel((i, j))
        if pixel_i1 != pixel_i2:
            diff.putpixel((i,j), (255, 0, 0, abs(pixel_i1 - pixel_i2)))
            diffs.append([i, j, abs(pixel_i1 - pixel_i2)])

model = KMeans(n_clusters=n_clusters)
model.fit(diffs)
clusters = model.predict(diffs)

for i in range(len(clusters)):
    cluster = clusters[i]
    d = diffs[i]
    color_hash = int((255 + 255 + 255) * (cluster/n_clusters))
    if color_hash < 255: color = (color_hash, 0, 0, d[2])
    if color_hash >= 255 and color_hash < 510: color = (0, color_hash - 255, 0, d[2])
    if color_hash >= 510: color = (0, 0, color_hash - 510, d[2])
    clustered_image.putpixel((d[0], d[1]), color)

diff.save('output/result.png')
clustered_image.save('output/clustered-result.png')

# --- clustering data
clustered_diffs = [ [] for i in range(n_clusters) ]
for i in range(len(diffs)):
    clustered_diffs[clusters[i]].append(diffs[i])
dataset = [ [] for i in range(n_clusters) ]

count = 0
for cluster in clustered_diffs:
    top = 9999
    left = 9999
    down = 0
    right = 0
    for d in cluster:
        if d[0] < left: left = d[0]
        if d[0] > right: right = d[0]
        if d[1] < top: top = d[1]
        if d[1] > down: down = d[1]
    if left < right and top < down:
        new_image = Image.new('RGBA', (right - left + 1, down - top + 1), (0, 0, 0, 0))
        new_image2 = Image.new('L', (right - left + 1, down - top + 1), (0))
        for d in cluster:
            new_image.putpixel((d[0] - left, d[1] - top), (255, 0, 0, d[2]))
            new_image2.putpixel((d[0] - left, d[1] - top), (d[2]))
        new_image.save('output/cluster-%d.png' % (count))
        new_image2 = new_image2.resize((dim, dim))
        row = new_image2.histogram()
        row.append(left)
        row.append(right)
        row.append(top)
        row.append(down)
        row.append('output/cluster-%d.png' % (count))
        for i in range(dim):
            for j in range(dim):
                row.append(new_image2.getpixel((i, j)))
        dataset[clustered_diffs.index(cluster)] = row
        count += 1

col = [ ('bin_%d' % (c)) for c in range(256) ]
col.append('left')
col.append('right')
col.append('top')
col.append('down')
col.append('diff_image')
col2 = [ '%dx%d-pixel' % (i,j) for i in range(dim) for j in range(dim) ]
frame = pd.DataFrame(dataset, columns=(col+col2))
frame.to_csv('output/dataset.csv')
frame.to_json('output/dataset.json', orient='split')

sys.exit(0)
