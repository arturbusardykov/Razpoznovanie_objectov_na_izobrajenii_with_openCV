import cv2

# загрузка изображения
image = cv2.imread("image.jpg")

# создание объекта CascadeClassifier
cascade_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# преобразование изображения в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# распознавание объектов на изображении
objects = cascade_classifier.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=8, minSize=(30, 30))
# отображение распознанных объектов на изображении
for (x, y, w, h) in objects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# отображение изображения с распознанными объектами
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



