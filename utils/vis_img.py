"""
Reference: 
    MIT License
    Copyright (c) 2019 Sagar Vinodababu
"""


from PIL import Image, ImageDraw, ImageFont

class BoxVis(object):
    def __init__(self, confidence_level, classes, label_color_map, font_path):
        self.confidence_level = confidence_level
        self.classes = classes
        self.label_color_map = {k: label_color_map[i] for i, k in enumerate(self.classes)}
        self.font_path = font_path

    def draw_box(self, img_path, pred, width, height):
        
        img = Image.open(img_path)
        annotated_img = img
        draw = ImageDraw.Draw(annotated_img)
        # font = ImageFont.truetype(font=self.font_path, size=15)

        # background in pred(whose index was 0) already excluded
        for l_i, label in enumerate(self.classes):
            find_index = (pred[l_i][:, 0] >= self.confidence_level).nonzero().squeeze(1)
            
            for k in find_index:
                xmin = (pred[l_i][k][1] * width).long()
                ymin = (pred[l_i][k][2] * height).long()
                xmax = (pred[l_i][k][3] * width).long()
                ymax = (pred[l_i][k][4] * height).long()

                # Boxes
                box_location = [xmin, ymin, xmax, ymax]
                draw.rectangle(xy=box_location, outline=self.label_color_map[label])
                draw.rectangle(xy=[l + 1. for l in box_location], outline=self.label_color_map[label])

                # Text
                # text_size = font.getsize(self.classes[l_i].upper())
                # text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
                # textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4., box_location[1]]
                # draw.rectangle(xy=textbox_location, fill=self.label_color_map[label])
                # draw.text(xy=text_location, text=label.upper(), fill='white', font=font)
        
        del draw

        return annotated_img

    def save_img(self, img, path):
        img.save(path)
        
