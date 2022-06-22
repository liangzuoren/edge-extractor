import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import cv2
from abc import ABC

#Abstract base class for the different types of edge extractors
class edge_extractor(ABC):
    def __init__(self):
        self.image = None
    
    def set_image(self,image):
        self.image = image
    
    def get_image(self):
        return self.image
    
    def extract_edges(self):
        pass


#Sobel filter
class sobel_edge_extractor(edge_extractor):
    def __init__(self):
        super(sobel_edge_extractor,self).__init__()

    def extract_edges(self):
        #Vertical Sobel Filter
        vertical_filter = [[-1,-2,-1],[0,0,0],[1,2,1]]
        
        #Horizontal Sobel Filter
        horizontal_filter = [[-1,0,1],[-2,0,2],[-1,0,1]]

        #Take shape of image where n = length, m = width, d = dimensions
        n,m,d = self.image.shape

        #Initialize edge image
        edges_img = np.empty(self.image.shape)

        #Initialize ims
        fig, ax = plt.subplots()
        ims = []
        ax.imshow(edges_img)

        #Loop through all pixels in image
        for row in range(n-2):
            for col in range(m-2):
                #Create 3x3 pixel box to look at
                pixel_box = self.image[row:row+3,col:col+3,0]

                #Apply vertical filter
                vertical_filtered_pixels = vertical_filter * pixel_box
                #Create vertical score
                vertical_score = vertical_filtered_pixels.sum()/9

                #Apply horizontal filter
                horizontal_filtered_pixels = horizontal_filter * pixel_box
                #Create horizontal score
                horizontal_score = horizontal_filtered_pixels.sum()/9

                #Calculate edge score using average
                edge_score = (vertical_score**2 + horizontal_score**2)**0.5

                #Insert into edges image
                edges_img[row,col] = [edge_score]*3

            if row%100 == 0:
                #Renormalize to between 0-1 range in each image frame
                im = ax.imshow(edges_img/edges_img.max(), animated=True)
                ims.append([im])
        
        #Renormalize to between 0-1 range
        edges_img = edges_img/edges_img.max()

        ani = animation.ArtistAnimation(fig, ims, interval = 100, repeat_delay = 1000, blit = True)
        
        with open("Example.html","w") as f:
            print(ani.to_html5_video(), file=f)

#Canny filter using openCV
class canny_edge_extractor(edge_extractor):
    def __init__(self):
        super(canny_edge_extractor,self).__init__()

    def extract_edges(self):
        #Use OpenCV to apply a Canny Filter and wait until keypress to view image
        edges = cv2.Canny(self.image,100,200)
        cv2.imshow('Edges',edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    #Create Sobel extractor
    sobel = sobel_edge_extractor()
    #Create Canny extractor
    canny = canny_edge_extractor()
    #Set images
    image = plt.imread('DSC_0221.JPG')

    #Extract edges
    sobel.set_image(image)
    sobel.extract_edges()
    canny.set_image(image)
    canny.extract_edges()

if __name__ == "__main__":
    main()
