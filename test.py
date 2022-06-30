import cv2
import numpy as np

img_file = 'Color_Checker.pdf.jpg'
# img_file = 'card2.jpeg'
# img_file = 'card3.jpg'
img = cv2.imread(img_file)
# print(img.shape)
# cv2.imshow('raw', img)
# detect
detector = cv2.mcc.CCheckerDetector_create()
detector.process(img, cv2.mcc.MCC24)
checker = detector.getBestColorChecker()
# draw
cdraw = cv2.mcc.CCheckerDraw_create(checker)
img_draw = img.copy()
cdraw.draw(img_draw)
cv2.imshow('draw', img_draw)
# get ccm
charts_rgb = checker.getChartsRGB()
src = charts_rgb[:, 1].copy().reshape(24, 1, 3)
src /= 255
# generate model
model = cv2.ccm_ColorCorrectionModel(src, cv2.ccm.COLORCHECKER_Macbeth)
# model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
model.setCCM_TYPE(cv2.ccm.CCM_3x3)
model.setDistance(cv2.ccm.DISTANCE_CIE2000)
model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
model.setLinearGamma(2.2)
model.setLinearDegree(3)
model.setSaturatedThreshold(0, 0.98)
model.run()
# ccm = model.getCCM()
loss = model.getLoss()
# print('ccm', ccm)
print('loss', loss)
# calibrate
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = img2.astype(np.float64)
img2 /= 255.0
calibrated = model.infer(img2)
out = calibrated * 255
out[out<0] = 0
out[out>255] = 255
out = out.astype(np.uint8)
out_img = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
cv2.imshow('calibrated', out_img)


cv2.waitKey(0)
cv2.destroyAllWindows()