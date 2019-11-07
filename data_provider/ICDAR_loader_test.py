import os
import numpy as np
import json
import cv2
import sys
#from data_loader import DataLoader
#from data_utils import label_to_array
import config

def label_to_array(label):
	try:
		#label = label.replace(' ', '')
		labelList=[]
		for ch in label:
			if ch == ' ':
				idx = 0
			else:
				idx = config.CHAR_VECTOR.index(ch)
			labelList.append(idx)
		return labelList
		#return [config.CHAR_VECTOR.index(x) for x in label]
	except Exception as ex:
		print(label)
		raise ex

def drawLine(img, p1, p2, color):
	cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color)

def drawSon(img, son_dic):
	text = son_dic['transcription']
	points = son_dic['points']
	conf = son_dic['confidence']
	drawLine(img, points[0], points[1], (0,255,0))
	drawLine(img, points[1], points[2], (0,255,0))
	drawLine(img, points[2], points[3], (0,255,0))
	drawLine(img, points[3], points[0], (0,255,0))
	cv2.putText(img, text, (points[0][0],points[0][1]),1,1,(0,0,255))

class ICDARLoader():
	def __init__(self, edition='13', shuffle=False):
		#super(ICDARLoader, self).__init__()
		self.edition = edition
		self.shuffle = shuffle # shuffle the polygons

	def load_annotation(self, gt_file):
		text_polys = []
		text_tags = []
		labels = []
		if not os.path.exists(gt_file):
			return np.array(text_polys, dtype=np.float32)
		with open(gt_file, 'r', encoding="utf-8-sig") as f:
			for line in f.readlines():
				try:
					line = line.replace('\xef\xbb\bf', '')
					line = line.replace('\xe2\x80\x8d', '')
					line = line.strip()
					line = line.split(',')
					if self.edition == '17':
						line.pop(8)  # since icdar17 has script
					# Deal with transcription containing ,
					if len(line) > 9:
						label = line[8]
						for i in range(len(line) - 9):
							label = label + "," + line[i + 9]
					else:
						label = line[-1]

					temp_line = list(map(eval, line[:8]))
					x1, y1, x2, y2, x3, y3, x4, y4 = map(float, temp_line)

					text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
					if label == '*' or label == '###' or label == '':
						text_tags.append(True)
						labels.append([-1])
					else:
						labels.append(label_to_array(label))
						text_tags.append(False)
				except Exception as e:
					print(e)
					continue
		text_polys = np.array(text_polys)
		text_tags = np.array(text_tags)

		return text_polys, text_tags, labels


def saveTxt(saveDir, imgName, son_dic):
	saveFn = saveDir + imgName + '.txt'
	saveFileFn = open((saveFn), 'w', encoding="utf-8")
	for idx in range(len(son_dic)):
		text = son_dic[idx]['transcription']
		pts = son_dic[idx]['points']
		#conf = son_dic[idx]['illegibility']
		line = str(pts[0][0]) + ',' + str(pts[0][1]) + ','+ \
			   str(pts[1][0]) + ',' + str(pts[1][1]) + ',' + \
		       str(pts[2][0]) + ',' + str(pts[2][1]) + ','+ \
			   str(pts[3][0]) + ',' + str(pts[3][1]) + ',' + \
			   str(text)
		saveFileFn.write(line + '\n')
	saveFileFn.close()

def  covJson2Txt(txtDir, imgNameList, jsonFn):
	if not os.path.exists(jsonFn):
		return
	with open(jsonFn, 'r', encoding="utf-8") as f:
		json_dic = json.load(f)
	for idx in range(len(imgNameList)):
		imgName = imgNameList[idx].replace('.jpg', '')
		print('%d-%d-%s' % (len(imgNameList), idx, imgName))
		if imgName in json_dic.keys():
			saveTxt(txtDir, imgName, json_dic[imgName])

def show_icdar2019_test(img, text_polys, text_tags):
	for idx in range(len(text_polys)):
		#rect = text_polys[idx]
		p1,p2,p3,p4 = text_polys[idx]
		cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (0,255,0))
		cv2.line(img, (p2[0], p2[1]), (p3[0], p3[1]), (0,255,0))
		cv2.line(img, (p3[0], p3[1]), (p4[0], p4[1]), (0,255,0))
		cv2.line(img, (p4[0], p4[1]), (p1[0], p1[1]), (0,255,0))
		cv2.putText(img, str(text_tags[idx]), (p1[0], p1[1]),1,1,(0,0,255))
	cv2.imshow('gt', img)
	cv2.waitKey(0)

def findChInDic(total_dic_list, uch):
	ret = False
	for ch in total_dic_list:
		if ch == uch:
			ret = True
			return ret
	return ret

def staticDic(total_dic_list, son_dic):
	total_list = total_dic_list
	for idx in range(len(son_dic)):
		text = son_dic[idx]['transcription']
		for ch in text:
			if findChInDic(total_list, ch) == False:
				total_list.append(ch)
	return total_list

def saveDic(total_list):
	saveFn = 'ICDAR_2019_dic.txt'
	saveFileFn = open((saveFn), 'w', encoding="utf-8")
	for idx in range(len(total_list)):
		saveFileFn.write(str(total_list[idx]) + '\n')
	saveFileFn.close()

# 获取所有字典
def getDic(imgNameList, jsonFn):
	total_list = []
	if not os.path.exists(jsonFn):
		return
	with open(jsonFn, 'r', encoding="utf-8") as f:
		json_dic = json.load(f)
	for idx in range(len(imgNameList)):
		imgName = imgNameList[idx].replace('.jpg', '')
		print('%d-%d-%s' % (len(imgNameList), idx, imgName))
		if imgName in json_dic.keys():
			total_list = staticDic(total_list, json_dic[imgName])
	return total_list


rootDir = 'E:/work/data/ocr/ICDAR 2019/ICDAR2019 Robust Reading Challenge on Large-scale Street View Text with Partial Labeling/'
imgDir = rootDir + 'train_full_images/'
txtDir = rootDir + 'train_full_labels/'
srcJson = rootDir + 'train_full_labels.json'
imgNameList = os.listdir(imgDir)
covJson2Txt(txtDir, imgNameList, srcJson)
#total_list = getDic(imgNameList, srcJson)
#saveDic(total_list)

gtFn = txtDir + 'gt_0.txt'
#data_loader = ICDARLoader(edition='13', shuffle=True)
#data_loader.load_annotation(gtFn)