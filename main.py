import numpy as np
import matplotlib.pyplot as plt
import math
from abc import ABC

class edge_extracter(ABC):
    def __init__(self):
        self.image = None
    
    def set_image(self,image):
        self.image = image
    
    def get_image(self):
        return self.image
    
    def extract_edges(self):
        pass


class sobel_edge_extracter(edge_extracter):
    def __init__(self):
        super(sobel_edge_extracter,self).__init__()

    def extract_edges(self):
        #Vertical Sobel Filter
        vertical_filter = [[-1,-2,-1],[0,0,0],[1,2,1]]
        
        #Horizontal Sobel Filter
        horizontal_filter = [[-1,0,1],[-2,0,2],[-1,0,1]]

        #Take shape of image where n = length, m = width, d = dimensions
        n,m,d = self.image.shape

        #Initialize edge image
        edges_img = np.empty(self.image.shape)

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
        
        #Renormalize to between 0-1 range
        edges_img = edges_img/edges_img.max()

        return edges_img

