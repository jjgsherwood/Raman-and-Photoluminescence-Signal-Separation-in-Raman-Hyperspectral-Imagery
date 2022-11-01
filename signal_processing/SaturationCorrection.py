import numpy as np
import copy


class remove_saturation():
    def __init__(self, region_width=3):
        """
        This removes saturation by averaging the neighbourhood.

        region_width: This determines the area used for averaging.
        """
        self.region_width = region_width//2

    def __call__(self, img):
        return self._find_saturation(img)

    def _find_saturation(self, img):
        """
        Saturated points can be recognized by zeros in the data.
        """
        X,Y,_ = np.where(img == 0)
        saturation_points = set(zip(X,Y))

        if not saturation_points:
            return img

        return self._fix_saturation(img, saturation_points)

    def _fix_saturation(self, img, saturation_points):
        """
        copy is needed to make sure that "fix" saturation points
        do not influence to be fix saturation points
        """
        img2 = copy.copy(img)
        for x,y in saturation_points:
            neighbourhood = []
            for i in range(-self.region_width,self.region_width+1):
                for j in range(-self.region_width,self.region_width+1):
                    if (img[x+i,y+j] == 0).any():
                        continue
                    neighbourhood.append(img[x+i,y+j])
            new = np.mean(neighbourhood, 0)
            img2[x,y] = new

        return img2
