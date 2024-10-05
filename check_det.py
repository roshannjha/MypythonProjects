import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Roshan.1.Jha\Tesseract-OCR\tesseract.exe'


class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        self.height, self.width, self.channels = self.image.shape
        self.orignal = self.image.copy()
    
    def i_name(self,p):
        print(f"Function calling------{p}------->")
        return p
    
    
    def show_Im(self):
        cv2.imwrite('Img.png',self.image)
        cv2.imshow("im",self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        
        
        
        
    def fix_image(self,img):
        self.img=img
        self.img[self.img > 127] = 255
        self.img[self.img < 127] = 0
        return self.img



    def crop_image(self):
        crop_height = self.height // 3
        crop_width = self.width // 3
        start_row = 0
        end_row = crop_height
        start_col = 0
        end_col = self.width
        cropped_image = self.image[start_row:end_row, start_col:end_col]
        self.image = cropped_image
        #cv2.imsave('cropp.png',self.image)




    def convert_to_grayscale(self):
        grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = grey

    def binarize_image(self):
        _, img_bin = cv2.threshold(self.image, 180, 225, cv2.THRESH_OTSU)
        self.image = img_bin

    def find_bounding_boxes(self):
        lWidth = 2
        lineMinWidth = 15
        kernal1 = np.ones((lWidth, lWidth), np.uint8)
        kernal1h = np.ones((1, lWidth), np.uint8)
        kernal1v = np.ones((lWidth, 1), np.uint8)
        kernal6 = np.ones((lineMinWidth, lineMinWidth), np.uint8)
        kernal6h = np.ones((1, lineMinWidth), np.uint8)
        kernal6v = np.ones((lineMinWidth, 1), np.uint8)
        img_bin_h = cv2.morphologyEx(~self.image, cv2.MORPH_CLOSE, kernal1h)
        img_bin_h = cv2.morphologyEx(img_bin_h, cv2.MORPH_OPEN, kernal6h)
        img_bin_v = cv2.morphologyEx(~self.image, cv2.MORPH_CLOSE, kernal1v)
        img_bin_v = cv2.morphologyEx(img_bin_v, cv2.MORPH_OPEN, kernal6v)
        img_bin_final = self.fix_image(self.fix_image(img_bin_h) | self.fix_image(img_bin_v))
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8,ltype=cv2.CV_32S)

        self.bounding_boxes = []
        self.bounding_boxes1 = []

        for x, y, w, h, area in stats[2:]:
            # if area > 300 and area < 400:
            #     self.bounding_boxes.append([x, y, w, h, area])
            #     cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # if area>100 and area<200:
            #     self.bounding_boxes1.append([x,y,w,h,area])
            #     cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.bounding_boxes.append([x, y, w, h, area])
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imwrite("iii.jpg",self.image)

        
        if len(self.bounding_boxes)==0:
            self.bounding_boxes=self.bounding_boxes1

# class FormAnalyzer():

#     def __init__(self, image_processor):

#         self.image_processor = image_processor

        

    def analyze_form(self):
        densiti = []
        thresh = 128
        #orignal = self.image.copy()
        for x, y, w, h, area in self.bounding_boxes:
            roi = self.orignal[y:y + h, x:x + w]
            thresh_roi = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)[1]
            gray = cv2.cvtColor(thresh_roi, cv2.COLOR_BGR2GRAY)
            non_zero_pixels = cv2.countNonZero(gray)
            density = non_zero_pixels / area
            densiti.append(density)

        min_density = min(densiti)
        most_dense_box_index = densiti.index(min_density)
        most_dense_box = self.bounding_boxes[most_dense_box_index]

        x, y, w, h, area = most_dense_box
        cv2.rectangle(self.orignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
        same_line_boxes = []
        threshold = 5
        for i in range(len(self.bounding_boxes)):
            for j in range(i + 1, len(self.bounding_boxes)):
                y_diff = abs(self.bounding_boxes[i][1] - self.bounding_boxes[j][1])
                if y_diff <= threshold:
                    same_line_boxes.append(self.bounding_boxes[i])
                    same_line_boxes.append(self.bounding_boxes[j])

        unique_boxes = []
        for box in same_line_boxes:
            if box not in unique_boxes:
                unique_boxes.append(box)

    #def extract_text_after_checkbox(self, image, most_dense_box):
        if most_dense_box in unique_boxes:
            x_checkbox_right = most_dense_box[0] + most_dense_box[2]
            new_roi_x = x_checkbox_right + 10
            new_roi_y = most_dense_box[1] + 5
            new_roi_w = 200
            new_roi_h = most_dense_box[3]
            text_after_checkbox = pytesseract.image_to_string(
                self.orignal[new_roi_y:new_roi_y + new_roi_h, new_roi_x:new_roi_x + new_roi_w], config='--psm 6')

            print(text_after_checkbox)
            self.alltext = text_after_checkbox.split(" ")
            print(self.alltext)
            print("1")
    #def extract_additional_text(self, image, most_dense_box, bounding_boxes):

        sorted_boxes = sorted(self.bounding_boxes, key=lambda box: box[1])
        if len(self.bounding_boxes)==7 and (most_dense_box == sorted_boxes[len(sorted_boxes) - 1]):
            x_checkbox_right = most_dense_box[0] + most_dense_box[2]
            new_roi_x = x_checkbox_right + 10
            new_roi_y = most_dense_box[1]
            new_roi_w = 200
            new_roi_h = most_dense_box[3]
            text_after_checkbox1 = pytesseract.image_to_string(
                self.orignal[new_roi_y:new_roi_y + new_roi_h, new_roi_x:new_roi_x + new_roi_w], config='--psm 6')

            print(text_after_checkbox1)

            self.alltext = text_after_checkbox1.split(" ")

            print(self.alltext)
            print("2")



        if len(self.bounding_boxes)==7 and (most_dense_box == sorted_boxes[len(sorted_boxes) - 2]):
            x_checkbox_right = most_dense_box[0] + most_dense_box[2]
            new_roi_x = x_checkbox_right + 10
            new_roi_y = most_dense_box[1]
            new_roi_w = self.orignal.shape[1] - new_roi_x
            new_roi_h = most_dense_box[3] + 100
            text_after_checkbox2 = pytesseract.image_to_string(
                self.orignal[new_roi_y:new_roi_y + new_roi_h, new_roi_x:new_roi_x + new_roi_w], config='--psm 6')

            print(text_after_checkbox2)

            self.alltext = text_after_checkbox2.split(" ")

            print(self.alltext)
            print("3")
        if len(self.alltext)==0:
            x_checkbox_right = most_dense_box[0] + most_dense_box[2]
            new_roi_x = x_checkbox_right + 10
            new_roi_y = most_dense_box[1]
            new_roi_w = 200
            new_roi_h = most_dense_box[3]
            text_after_checkbox1 = pytesseract.image_to_string(
                self.orignal[new_roi_y:new_roi_y + new_roi_h, new_roi_x:new_roi_x + new_roi_w], config='--psm 6')

            print(text_after_checkbox1)

            self.alltext = text_after_checkbox1.split(" ")

            print(self.alltext)
            print("4")
            



    def check_options(self):

        options = [

            ["Individual / sole proprietor or single-member LLC"],

            ["C Corporation"],

            ["S Corporation"],

            ["Partnership"],

            ["Trust/estate"],

            ["""Limited liability company.Enter the tax classification(C=C corporation,S=S corporation,P=Partnership)____

Note:CHeck the appropriate box in the line above for the tax classification of the single-member owner.Do not check

LLC if the LLC is classified as a single-memeber LLC that is disregarded from the owner unless the owner of the LLC is

another LLC that is not disregarded from the owner for the US fedral tax purposes.Otherwise,a single-member LLC that

is disqregarded from the owner should check the appropriate box for the tax classification of its owner."""],

            ["Other(see instructions)"]        ]



        for option in options:
            for text in self.alltext:
                if text in option:
                    print(option)
                    break