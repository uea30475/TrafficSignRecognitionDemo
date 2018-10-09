import cv2,os
import numpy as np
import classification
lowerBound = np.array([100,128,0],np.uint8)
upperBound = np.array([215,255,255],np.uint8)
kernelOpen = np.ones((30,30))
signName = ["ERROR","Bien1","Bien2","Bien3","Bien4","Bien5","Bien6","Bien7","Bien8","Bien9","Bien10","Bien11","Bien12"]
#Can bang histogram, lam sang anh
def preprocess_image(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    #cv2.GaussianBlur(img_hist_equalized, (3,3), 0)#Lam mo anh
    return img_hist_equalized

#Loai bo nhieu
def MorpRemoveNoise(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
    temp = cv2.morphologyEx(imgin,cv2.MORPH_OPEN,w)
    imgout = cv2.morphologyEx(temp,cv2.MORPH_OPEN,w)
    return imgout

#Ham lam mo, giu lai cac canh
def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)#Lam mo anh
    gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)#Loc nhieu Laplacian
    LoG_image = cv2.convertScaleAbs(LoG_image)#chuyen sang uint8
    return LoG_image

#Chuyen anh sang nhi phan trang den
def binarization(image):
    image = preprocess_image(image)
    image = LaplacianOfGaussian(image)
    thresh = cv2.threshold(image,32,255,cv2.THRESH_BINARY)[1]
    return thresh
#Ham xoa cac thanh phan lien thong
def removeSmallComponents(image, threshold):
    #Tim cac thanh phan lien thong (cac dom mau trang trong anh)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #Giu lai thanh phan khi no tren nguong
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

#Ham kiem tra hull co the la duong tron khong
def is_contour_bad(c):
	# approximate  contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    # so diem di qua hull tu 8 den 23 diem thi hull co the la duong tron
	return not (len(approx) >= 8 and len(approx) <= 23)

#def Erosion(imgin):
#    w = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))#bao mon anh voi kich thuoc
#    imgout = cv2.erode(imgin,w)
#    return imgout

def contourIsSign(perimeter, centroid, threshold):
    #  perimeter, centroid, threshold
    # # Compute signature of contour
    result=[]
    for p in perimeter:
        p = p[0]
        distance = np.sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result ]
    # Check signature of contour.
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold: # is  the sign
        return True, max_value + 2
    else:                 # is not the sign
        return False, max_value + 2

#crop sign 

def crop(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    return image[top:bottom,left:right]


def cropContour(image, contours, threshold, distance_theshold):
    max_distance = 0
    coordinate = None
    sign = None
    for c in contours:
        #area = cv2.
        hull = cv2.convexHull(c)
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and distance > max_distance and distance > distance_theshold and not is_contour_bad(hull):
            max_distance = distance
            coordinate = np.reshape(c, [-1,2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinate = [(left-2,top-2),(right+3,bottom+1)]
            sign = crop(image,coordinate)
    return sign, coordinate


def writeContour():
    print("Read video and crop contour")
    count = 0   
    cam = cv2.VideoCapture("MVI_1063.avi")
    path = "63"
    while(1):
        _,img = cam.read()

        if not _:
            print("FINISHED")
            break
        else:
            original_image = img
            img = preprocess_image(original_image)
            imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(imgHSV,lowerBound,upperBound)
            mask_tmp = mask.copy()
            mask = cv2.dilate(mask,kernelOpen)
            img_binarization = binarization(img)
            mask = MorpRemoveNoise(mask)
            img_binarization = cv2.bitwise_and(mask_tmp,mask)
            mask = removeSmallComponents(img_binarization,300)
            _,contours,h = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            sign, coordinate = cropContour(img_binarization, contours, 0.65, 15)
            if sign is not None:
                count = count +1
                cv2.imwrite(os.path.join(path + '/' , path + "_" +str(count) +".png"), sign)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

def detectImageFromVideo(clf):
    font = cv2.FONT_HERSHEY_PLAIN
    count = 0   
    cam = cv2.VideoCapture("MVI_1049.avi")
    while(1):
        _,img = cam.read()

        if not _:
            print("FINISHED")
            break
        else:
            original_image = img
            img = preprocess_image(original_image)
            imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(imgHSV,lowerBound,upperBound)
            mask_tmp = mask.copy()
            mask = cv2.dilate(mask,kernelOpen)
            img_binarization = binarization(img)
            mask = MorpRemoveNoise(mask)
            img_binarization = cv2.bitwise_and(mask_tmp,mask)
            mask = removeSmallComponents(img_binarization,300)
            _,contours,h = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            sign, coordinate = cropContour(img_binarization, contours, 0.65, 15)
            text = ""
            
            sign_type = None
            if sign is not None:
                sign_type = classification.getLabel(clf,sign)
                sign_type = sign_type if sign_type > 0 else 0
                text = signName[sign_type]
            if sign_type > 0 :
                cv2.rectangle(original_image, coordinate[0],coordinate[1], (0, 255, 0), 1)
                cv2.putText(original_image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
            cv2.imshow("Result",original_image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()    
def main():
    #writeContour()
    try:
        clf = classification.reloadData()
        print("Load training file successful...")
    except :
        print("The file data trainning not exist, start trainning...")
        clf = classification.trainning()
    print("Start detect sign...")
    detectImageFromVideo(clf)
if __name__ == '__main__':
    main()
    
