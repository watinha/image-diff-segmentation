import sys, np, os, shutil, pandas as pd

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

if len(sys.argv) < 6:
    print('fifth argument should be the directory to output the results...')
    sys.exit(1)

f1 = sys.argv[1]
f2 = sys.argv[2]
n_clusters = int(sys.argv[3])
dim = int(sys.argv[4])
folder = sys.argv[5]

i1 = Image.open(f1)
i2 = Image.open(f2)

if os.path.exists('./output/%s' % (folder)):
    print('output directory exists... is it ok?')
    sys.exit(1)

os.mkdir('./output/%s' % (folder))
i1.save('output/%s/screenshot-baseline.png' % (folder))
i2.save('output/%s/screeenshot-test.png' % (folder))

diff_width = i2.size[0] - i1.size[0]
min_height = min(i1.size[1], i2.size[1])
crop_area = (int(diff_width / 2), 0, i1.size[0] + int(diff_width / 2), min_height)
i1 = i1.convert('L')
i2 = i2.convert('L').crop(crop_area)
#i2 = i2.convert('L').resize(i1.size)
i1.save('output/%s/screenshot-baseline-croped.png' % (folder))
i2.save('output/%s/screenshot-test-croped.png' % (folder))

diff = Image.new('RGBA', i1.size, (0, 0, 0, 0))
clustered_image = Image.new('RGBA', i1.size, (0, 0, 0, 0))
diffs = []

(width, height) = i1.size
for i in range(width):
    for j in range(min_height):
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

diff.save('output/%s/result.png' % (folder))
clustered_image.save('output/%s/clustered-result.png' % (folder))

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
        new_image.save('output/%s/cluster-%d.png' % (folder, count))
        new_image2 = new_image2.resize((dim, dim))
        row = new_image2.histogram()
        row.append(left)
        row.append(right)
        row.append(top)
        row.append(down)
        row.append('output/%s/cluster-%d.png' % (folder, count))
        for i in range(dim):
            for j in range(dim):
                row.append(new_image2.getpixel((i, j)))
        roib = i1.crop((left, top, right, down))
        roib_histogram = np.histogram(roib.histogram())[0].tolist()
        roit = i2.crop((left, top, right, down))
        roit_histogram = np.histogram(roit.histogram())[0].tolist()
        roib.save('output/%s/cluster-%d-base.png' % (folder, count))
        roit.save('output/%s/cluster-%d-target.png' % (folder, count))
        dataset[clustered_diffs.index(cluster)] = row + roib_histogram + roit_histogram
        count += 1

col = [ ('bin_%d' % (c)) for c in range(256) ]
col.append('left')
col.append('right')
col.append('top')
col.append('down')
col.append('diff_image')
col2 = [ '%dx%d-pixel' % (i,j) for i in range(dim) for j in range(dim) ]
col3 = [ ('roib_bin_%d' % (c)) for c in range(10) ] # 10 bins Browserbite
col4 = [ ('roit_bin_%d' % (c)) for c in range(10) ] # 10 bins Browserbite
frame = pd.DataFrame(dataset, columns=(col+col2+col3+col4))
frame.to_csv('output/%s/dataset.csv' % (folder))
json = frame.to_json(orient='split')
f = open('output/%s/dataset.json' % (folder), 'w')
f.write('let dataset = %s' % (json))
f.close()

shutil.copyfile('output/index.html', ('output/%s/index.html' % (folder)))

sys.exit(0)
