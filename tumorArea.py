# import torch
# from torch.autograd import Variable
import os
import xml.dom.minidom
import h5py
import openslide


def point_in_polygon(x, y, verts):
    """
    - PNPoly算法
    - xyverts  [(x1, y1), (x2, y2), (x3, y3), ...]
    """
    try:
        x, y = float(x), float(y)
    except:
        return False
    vertx = [xyvert[0] for xyvert in verts]
    verty = [xyvert[1] for xyvert in verts]

    # N个点中，横坐标和纵坐标的最大值和最小值，判断目标坐标点是否在这个四边形之内
    if not verts or not min(vertx) <= x <= max(vertx) or not min(verty) <= y <= max(verty):
        return False

    # 上一步通过后，核心算法部分
    nvert = len(verts)
    is_in = False
    for i in range(nvert):
        j = nvert - 1 if i == 0 else i - 1
        if ((verty[i] > y) != (verty[j] > y)) and (
                x < (vertx[j] - vertx[i]) * (y - verty[i]) / (verty[j] - verty[i]) + vertx[i]):
            is_in = not is_in

    return is_in


def analysis(file):
    dom = xml.dom.minidom.parse(file)
    # 得到文档元素对象
    Annotations = dom.documentElement
    Annotation = Annotations.getElementsByTagName('Annotation')
    Regions = Annotation[0].getElementsByTagName('Regions')
    Regions = Regions[0].getElementsByTagName('Region')
    # print('This picture has {} regions.'.format(len(Regions)))
    n = len(Regions)
    tumorRegions = []
    normalRegions = []
    vertexes = []
    for i in range(n):
        if Regions[i].getAttribute('NegativeROA') == '1':
            Vertices = Regions[i].getElementsByTagName('Vertices')[0].getElementsByTagName('Vertex')
            vertex_num = len(Vertices)
            for j in range(vertex_num):
                vertexes.append((int(Vertices[j].getAttribute('X')), int(Vertices[j].getAttribute('Y'))))
            normalRegions.append(vertexes)
            vertexes = []
        else:
            Vertices = Regions[i].getElementsByTagName('Vertices')[0].getElementsByTagName('Vertex')
            vertex_num = len(Vertices)
            for j in range(vertex_num):
                vertexes.append((int(Vertices[j].getAttribute('X')), int(Vertices[j].getAttribute('Y'))))
            tumorRegions.append(vertexes)
            vertexes = []
    return tumorRegions, normalRegions


def whetherInArea(x, y, regions):
    region_num = len(regions[1])
    for i in range(region_num):
        if point_in_polygon(x, y, regions[1][i]):
            return True
    return False


if __name__ == "__main__":
    # 打开xml文档
    slide_path = 'C:/tumor_detect/try'
    slide_files = os.listdir(slide_path)
    files = [os.path.splitext(file)[0] for file in slide_files]
    xml_path = 'C:/tumor_detect/xml'
    h5_path = 'C:/tumor_detect/patch_gen/patches'
    img_path = 'C:/tumor_detect/patch_gen/images'
    for file in files:
        regions = analysis(xml_path + '/' + file + '.xml')
        h5_file = h5py.File(h5_path + '/' + file + '.h5', 'r')
        coords = h5_file['coords'][()]
        if os.path.isfile(slide_path + '/' + file + '.svs'):
            slide = openslide.OpenSlide(slide_path + '/' + file + '.svs')
        elif os.path.isfile(slide_path + '/' + file + '.tif'):
            slide = openslide.OpenSlide(slide_path + '/' + file + '.tif')
        else:
            continue
        if not os.path.isdir(img_path + '/' + file):
            os.mkdir(img_path + '/' + file)
        for x, y in coords:
            n = 0
            n += whetherInArea(x, y, regions).__int__()
            n += whetherInArea(x + 256, y, regions).__int__()
            n += whetherInArea(x, y + 256, regions).__int__()
            n += whetherInArea(x + 256, y + 256, regions).__int__()

            if n >= 3:
                slide.read_region((x, y), 0, (256, 256)).convert("RGB").save(
                    img_path + '/' + file + '/{}_{}_{}.png'.format(file, x, y))

# 思路：判断正方形的四个点中的三个点在里边的话就可以当癌patch
