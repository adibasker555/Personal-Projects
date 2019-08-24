import json
import numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk, ExifTags
import os
import traceback
import csv
from multiprocessing import Process, Pool, Queue, Manager

class VertexManipulator():

    def find_token_attributes(self, vertices):
        min_x, max_x, min_y, max_y = self.find_min_max(vertices)
        x_mid = (min_x + max_x)/2
        y_mid = (min_y + max_y)/2
        width = max_x - min_x
        height = max_y - min_y
        return x_mid,y_mid,width,height

    def find_min_max(self, vertices):
        x_vals = []
        y_vals = []
        for vertex in vertices:
            x_vals.append(vertex['x'])
            y_vals.append(vertex['y'])
        return min(x_vals),max(x_vals),min(y_vals),max(y_vals)

class FeaturesOutputsCalculator():
    child_labels = ["childs first name", "name of child", "childs name", "child", "name", "name of registration", "middle", "last", "second"]
    date_labels = ["date and time of birth", "date of birth", "birth date", "date of", "birthdate"]
    member_labels = ["full maiden name of mother", "maiden name of mother", "mothers maiden name", "name of mother", "mothers name", "mother name", "maiden name", "mother",
                     "full name of father", "name of father", "name of parent", "fathers name", "father name", "parent name", "father", "parent", "name",
                     "mother current legal name", "father current legal name", "Mother name prior to first marriage", "civil union", "first", "middle", "second", "last"]
    spouse_labels = ["full maiden name of mother", "maiden name of mother", "mothers maiden name", "name of mother", "mothers name", "mother name", "maiden name", "mother",
                    "full name of father", "name of father", "name of parent", "fathers name", "father name", "parent name", "father", "parent", "name",
                     "mother current legal name", "father current legal name", "Mother name prior to first marriage", "civil union", "first", "middle", "second", "last"]

    def __init__(self, fileIn):
        self.frame_width = 800
        self.frame_height =  800
        self.feature_output = {}
        self.fileIn = fileIn
        self.filepath = "/Users/abasker/Desktop/BCData/BC/" + self.fileIn.jpeg
        data = fileIn.loadJSON()
        self.root_data = data['responses'][0]['textAnnotations']
        self.doc_width, self.doc_height = self.find_document_dimensions()
        self.image = self.resolve_orientation()
        self.avg_height = self.find_avg_box_height()
        self.child_tokens = self.tokenize(self.child_labels)
        self.date_tokens = self.tokenize(self.date_labels)
        self.member_tokens = self.tokenize(self.member_labels)
        self.spouse_tokens = self.tokenize(self.spouse_labels)
        self.num_features = 11

    def tokenize(self, labels):
        tokens = []
        for label in labels:
            for str in label.split():
                if str not in tokens:
                    tokens.append(str)
        return tokens

    def find_document_dimensions(self):
        img = Image.open(self.filepath)
        width, height = img.size
        return width,height

    def find_avg_box_height(self):
        bb_heights = []
        for i in range(1, len(self.root_data)):
            token_vertices = self.root_data[i]['boundingPoly']['vertices']
            height = VertexManipulator().find_token_attributes(token_vertices)[3]
            bb_heights.append(height)
        avg_height = np.average(bb_heights)
        return avg_height

    def calculate_features(self):
        for i in range(1,len(self.root_data)):
            token_vertices = self.root_data[i]['boundingPoly']['vertices']
            token_description = self.root_data[i]['description']
            feature_array = self.calculate_feature_array(token_vertices, token_description)
            new_feature = {'feature':feature_array, 'output':None}
            if token_description in list(self.feature_output.keys()) and self.feature_output[token_description] is not None:
                existing_features = self.feature_output[token_description]
                existing_features.append(new_feature)
                self.feature_output[token_description] = existing_features
            else:
                self.feature_output[token_description] = [new_feature]

    def calculate_feature_array(self, token_vertices, token_description):
        doc_origin = self.doc_width/2, self.doc_height/2
        feature_array = []
        x_mid,y_mid,width,height = VertexManipulator().find_token_attributes(token_vertices)
        xy_distance = list(np.array([x_mid,y_mid]) - np.array(doc_origin))
        feature_array.extend([xy_distance[0], xy_distance[1], self.fileIn.jpeg])
        return feature_array

    def calc2(self, token, x_dist, y_dist):
        feature_array = []
        for i in range(1,len(self.root_data)):
            token_vertices = self.root_data[i]['boundingPoly']['vertices']
            token_description = self.root_data[i]['description']
            x_mid,y_mid,width,height = VertexManipulator().find_token_attributes(token_vertices)
            doc_origin = self.doc_width/2, self.doc_height/2
            xy_distance = list(np.array([x_mid,y_mid]) - np.array(doc_origin))
            if token == token_description and xy_distance[0] == x_dist and xy_distance[1] == y_dist:
                label_length = len(token_description)
                relative_height = height / self.avg_height
                child_dist = self.find_lowest_levenshtein(token_description, self.child_tokens)
                date_dist = self.find_lowest_levenshtein(token_description, self.date_tokens)
                member_dist = self.find_lowest_levenshtein(token_description, self.member_tokens)
                spouse_dist = self.find_lowest_levenshtein(token_description, self.spouse_tokens)
                parts = self.split(self.root_data,1)
                child_neighbor, date_neighbor, member_neighbor, spouse_neighbor = self.multi_process(parts,token_vertices, token_description)
                new_features = [label_length, width, relative_height, child_dist, date_dist, member_dist, spouse_dist, child_neighbor,
                                date_neighbor, member_neighbor, spouse_neighbor]
                feature_array.extend(new_features)
                self.num_features = len(new_features)
                return feature_array


    def find_class_neighbors(self, data, token_vertices, token_description, queue):

        child_neighbor,  date_neighbor, member_neighbor, spouse_neighbor = 0, 0, 0, 0
        x_search_min, x_search_max, y_search_min, y_search_max = self.searchThreshold(token_vertices)

        for i in range(1,len(data)):
            token_vertices = data[i]['boundingPoly']['vertices']
            x_min, x_max, y_min, y_max = VertexManipulator().find_min_max(token_vertices)
            description = data[i]['description']
            if x_max >= x_search_min and x_min <= x_search_max and y_max >= y_search_min and y_min <= y_search_max:
                child_neighbor += self.num_tokens_in_levenshtein(description, self.child_tokens)
                date_neighbor += self.num_tokens_in_levenshtein(description, self.date_tokens)
                member_neighbor += self.num_tokens_in_levenshtein(description, self.member_tokens)
                spouse_neighbor += self.num_tokens_in_levenshtein(description, self.spouse_tokens)

        queue.put((child_neighbor, date_neighbor, member_neighbor, spouse_neighbor))
        # return child_neighbor, date_neighbor, member_neighbor, spouse_neighbor

    def split(self, a, n):
        k, m = divmod(len(a), n)
        return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    def multi_process(self, parts, token_vertices, token_description):
        print(token_description)
        child_neighbor, date_neighbor, member_neighbor, spouse_neighbor = 0,0,0,0
        # p = Pool()
        # result = p.starmap(self.find_class_neighbors, [(part, token_vertices, token_description) for part in parts])
        # p.close()
        # p.join()
        # return result
        # for curr in result:
        #     child_neighbor += curr[0]
        #     date_neighbor +=  curr[1]
        #     member_neighbor += curr[2]
        #     spouse_neighbor += curr[3]

        processes = []
        q = Queue()

        for part in parts:
            processes.append(Process(target=self.find_class_neighbors, args=(part, token_vertices, token_description, q)))

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        while not q.empty():
            curr = q.get()
            child_neighbor += curr[0]
            date_neighbor +=  curr[1]
            member_neighbor += curr[2]
            spouse_neighbor += curr[3]

        return child_neighbor, date_neighbor, spouse_neighbor, member_neighbor


    def searchThreshold(self, token_vertices):
        x_mid,y_mid,width,height = VertexManipulator().find_token_attributes(token_vertices)
        x_search_min = x_mid - self.doc_width/8 if x_mid - self.doc_width/8 >= 0 else 0
        x_search_max = x_mid + self.doc_width/8 if x_mid + self.doc_width/8 <= self.doc_width else self.doc_width
        y_search_min = y_mid - self.doc_height/16 if y_mid - self.doc_height/16 >= 0 else 0
        y_search_max = y_mid + self.doc_height/16 if y_mid + self.doc_height/16 <= self.doc_height else self.doc_height
        return x_search_min, x_search_max, y_search_min, y_search_max


    def find_lowest_levenshtein(self, description, tokens):
        lowest_dist = 50
        for token in tokens:
            this_lev_dist = self.levenshtein(description, token)
            if this_lev_dist < lowest_dist:
                lowest_dist = this_lev_dist
        return lowest_dist

    def num_tokens_in_levenshtein(self, description, tokens):
        matches = 0
        for token in tokens:
            this_lev_dist = self.levenshtein(description, token)
            if this_lev_dist <= 1:
                matches += 1
        return matches

    def update_text_match(self,x,y,slot):
        output = [0,0,0,0,0]
        scaled_x = x*(self.doc_width/self.frame_width)
        scaled_y = y*(self.doc_height/self.frame_height)
        for i in range(1,len(self.root_data)):
            token_vertices = self.root_data[i]['boundingPoly']['vertices']
            min_x,max_x,min_y,max_y = VertexManipulator().find_min_max(token_vertices)
            if scaled_x >= min_x and scaled_x <= max_x and scaled_y >= min_y and scaled_y <= max_y:
                description = self.root_data[i]['description']
                output[slot] = 1
                for feature_dict in self.feature_output[description]:
                    if self.calculate_feature_array(token_vertices, description) == feature_dict['feature']:
                        feature_dict['output'] = output
                return description


    def get_existing_matches(self,slot):
        output = [0,0,0,0,0]
        output[slot] = 1
        display_list = []
        for key in list(self.feature_output.keys()):
            curr_feature_dict = self.feature_output[key]
            for dict in curr_feature_dict:
                if dict['output'] == output:
                    display_list.append(key)

        return display_list

    def resolve_orientation(self):
        image=Image.open(self.filepath)
        if hasattr(image, '_getexif'): # only present in JPEGs
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation]=='Orientation':
                    break
            e = image._getexif()       # returns None if no EXIF data
            if e is not None:
                exif=dict(e.items())
                orientation = exif.get(orientation,None)
                if orientation is not None:
                    if orientation == 3 or orientation == 6 or orientation == 8:
                        temp_height = self.doc_height
                        self.doc_height = self.doc_width
                        self.doc_width = temp_height

                    if orientation == 3:   image = image.transpose(Image.ROTATE_180)
                    elif orientation == 6: image = image.transpose(Image.ROTATE_270)
                    elif orientation == 8: image = image.transpose(Image.ROTATE_90)

        image.thumbnail((self.frame_width, self.frame_height), Image.ANTIALIAS)
        image.save("/Users/abasker/Desktop/testOrientation.jpg")
        return image


    def levenshtein(self, seq1, seq2):
        seq1 = seq1.lower()
        seq2 = seq2.lower()
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros ((size_x, size_y))
        for x in range(size_x):
            matrix [x, 0] = x
        for y in range(size_y):
            matrix [0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x-1] == seq2[y-1]:
                    matrix [x,y] = min(
                        matrix[x-1, y] + 1,
                        matrix[x-1, y-1],
                        matrix[x, y-1] + 1
                    )
                else:
                    matrix [x,y] = min(
                        matrix[x-1,y] + 1,
                        matrix[x-1,y-1] + 1,
                        matrix[x,y-1] + 1
                    )
        return (matrix[size_x - 1, size_y - 1])

class fileInitializer():
    def __init__(self,jpeg_file,json_file):
        self.jpeg = jpeg_file
        self.json = json_file

    def loadJSON(self):
        with open(self.json, "r") as read_file:
            data = json.load(read_file)
        return data

class GUI():
    def __init__(self,fileIn,foc):
        self.fileIn = fileIn
        self.foc = foc

        self.root = Toplevel()
        self.quit_event = False
        self.txt = None
        self.this_match = None
        self.curr_matches = None
        self.slot = -1
        self.color = "red"
        self.root.geometry('{}x{}+0+0'.format(self.foc.frame_width, self.foc.frame_height))

        frame = Frame(self.root, bd=2, relief=SUNKEN)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        xscroll = Scrollbar(frame, orient=HORIZONTAL)
        xscroll.grid(row=1, column=0, sticky=E+W)
        yscroll = Scrollbar(frame)
        yscroll.grid(row=0, column=1, sticky=N+S)
        self.canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        self.canvas.grid(row=0, column=0, sticky=N+S+E+W)
        xscroll.config(command=self.canvas.xview)
        yscroll.config(command=self.canvas.yview)
        frame.pack(fill=BOTH,expand=1)

        image = foc.image
        r_image = image.resize((self.foc.frame_width , self.foc.frame_height), Image.ANTIALIAS)
        r_image = ImageTk.PhotoImage(r_image)

        x_origin, y_origin = foc.frame_width/2, foc.frame_height/2
        print(foc.doc_width, foc.doc_height)

        self.canvas.create_image(0,0,image=r_image,anchor="nw")
        self.canvas.config(scrollregion=self.canvas.bbox(ALL))
        self.canvas.create_rectangle(20, 20, 30, 30, fill='green yellow')
        self.canvas.create_rectangle(40, 20, 50, 30, fill='orange')
        self.canvas.create_rectangle(60, 20, 70, 30, fill='cyan')
        self.canvas.create_rectangle(80, 20, 90, 30, fill='pink')
        self.canvas.create_rectangle(100, 20, 110, 30, fill='white')
        self.canvas.create_rectangle(x_origin - 5, y_origin - 5, x_origin + 5, y_origin + 5, fill='orange')

        self.canvas.bind("<Button 1>",self.printcoords)
        Button(self.root, text="Quit", command=self.quit).pack()

        while True:
            if self.quit_event:
                self.quit_event = False
                break
            self.root.update_idletasks()
            self.root.update()

    def quit(self):
        self.quit_event = True
        self.root.quit()

    def printcoords(self,event):
        if (event.x >= 20 and event.x <= 30) and (event.y >= 20 and event.y <= 30):
            self.slot = 0
            self.color = "green yellow"
            self.canvas.delete(self.txt)
            self.txt = self.canvas.create_text(40,40,fill=self.color,font="Times 10 italic bold",
                                     text="child name")
        elif (event.x >= 40 and event.x <= 50) and (event.y >= 20 and event.y <= 30):
            self.slot = 1
            self.color = "orange"
            self.canvas.delete(self.txt)
            self.txt = self.canvas.create_text(40,40,fill=self.color,font="Times 10 italic bold",
                                     text="date of birth")
        elif (event.x >= 60 and event.x <= 70) and (event.y >= 20 and event.y <= 30):
            self.slot = 2
            self.color = "cyan"
            self.canvas.delete(self.txt)
            self.txt = self.canvas.create_text(40,40,fill=self.color,font="Times 10 italic bold",
                                     text="member name")
        elif (event.x >= 80 and event.x <= 90) and (event.y >= 20 and event.y <= 30):
            self.slot = 3
            self.color = "pink"
            self.canvas.delete(self.txt)
            self.txt = self.canvas.create_text(40,40,fill=self.color,font="Times 10 italic bold",
                                     text="spouse name")
        elif (event.x >= 100 and event.x <= 110) and (event.y >= 20 and event.y <= 30):
            self.slot = 4
            self.color = "white"
            self.canvas.delete(self.txt)
            self.txt = self.canvas.create_text(40,40,fill=self.color,font="Times 10 italic bold",
                                               text="No Match")
        else:
            x,y = self.canvas.canvasx(event.x),self.canvas.canvasy(event.y)
            print(x,y)
            description = self.foc.update_text_match(x,y, self.slot)
            self.canvas.delete(self.this_match)
            self.this_match = self.canvas.create_text(130,20,fill=self.color,anchor="nw",font="Times 10 italic bold",
                                                      text=description)

        if self.slot != -1:
            self.canvas.delete(self.curr_matches)
            self.curr_matches = self.canvas.create_text(230,20,fill=self.color,anchor="nw",font="Times 10 italic bold",
                                                                                      text=str(self.foc.get_existing_matches(self.slot)))

def writeRow(writer, key, feature):
    row_dict = {}
    row_dict["Token"] = key
    row_dict["x-distance"] = feature[0]
    row_dict["y-distance"] = feature[1]
    row_dict["filename"] = feature[2]
    row_dict["Output"] = output
    writer.writerow(row_dict)


if __name__ == "__main__":

    all_features = {}
    bc_directory = '/Users/abasker/Desktop/BCData/BC'
    json_directory = '/Users/abasker/Desktop/BCData/Json'

    print("Gathering Features")
    for filename in sorted(os.listdir(bc_directory)):
        print(filename)
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            json_file = str(filename) + ".json"
            if json_file in os.listdir(json_directory):
                json_path = json_directory + "/" + json_file
                fileIn = fileInitializer(filename, json_path)
                foc = FeaturesOutputsCalculator(fileIn)
                foc.calculate_features()
                all_features[filename] = foc
                all_features[filename + "init"] = fileIn
    print("Collection Complete")

    fieldnames = ["Token", "x-distance", "y-distance", "filename", "Output"]

    if not os.path.exists('/Users/abasker/Desktop/MLData.csv'):
        with open('/Users/abasker/Desktop/MLData.csv', mode='w') as csv_file:
            ml_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            ml_writer.writeheader()

    for filename in sorted(os.listdir(bc_directory)):
        print(filename)
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            json_file = str(filename) + ".json"
            if json_file in os.listdir(json_directory):
                fileIn = all_features[filename + "init"]
                foc = all_features[filename]
                gui = GUI(fileIn, foc)
                with open('/Users/abasker/Desktop/MLData.csv',mode='r') as ml_file, open('/Users/abasker/Desktop/Output.csv', mode='w') as output_file:
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                    writer.writeheader()
                    reader = csv.reader(ml_file)
                    next(reader)
                    inside = False
                    for row in reader:
                        row_output = row[4]
                        row_file = row[3]
                        core_rf = [float(row[1]), float(row[2])]
                        curr_foc = all_features[row_file]
                        if row_file != filename:
                            for key in list(curr_foc.feature_output.keys()):
                                for feature_dict in curr_foc.feature_output[key]:
                                    feature = feature_dict['feature']
                                    output = feature_dict['output']
                                    core_feature = [float(feature[0]), float(feature[1])]
                                    if key == row[0] and core_feature == core_rf:
                                        writeRow(writer, key, feature)
                        else:
                            replace = False
                            feature_match = None
                            key_match = None
                            for key in list(curr_foc.feature_output.keys()):
                                for feature_dict in curr_foc.feature_output[key]:
                                    feature = feature_dict['feature']
                                    output = feature_dict['output']
                                    core_feature = [float(feature[0]), float(feature[1])]
                                    if key == row[0] and core_feature == core_rf:
                                        feature_match = feature
                                        key_match = key
                                        if output is not None or output == [0,0,0,0,1]:
                                            replace = True
                            if replace == False and feature_match is not None and key_match is not None:
                                writeRow(writer, key, feature)
                    curr_foc = all_features[filename]
                    for key in list(curr_foc.feature_output.keys()):
                        for feature_dict in curr_foc.feature_output[key]:
                            feature = feature_dict['feature']
                            output = feature_dict['output']
                            if output is not None and output != [0, 0, 0, 0, 1]:
                                writeRow(writer, key, feature)
                prev_file = filename
                os.rename('/Users/abasker/Desktop/Output.csv','/Users/abasker/Desktop/MLData.csv')
                prev_file = filename

            num_features = foc.num_features

            fieldnames = ["Token", "x-distance", "y-distance", "filename", "Output"]
            for i in range(0, num_features):
                fieldnames.append("feature" + str(i+6))

            with open('/Users/abasker/Desktop/Features.csv', mode='w') as data_points:
                writer = csv.DictWriter(data_points, fieldnames=fieldnames)
                writer.writeheader()
                for filename in sorted(os.listdir(bc_directory)):
                    print(filename)
                    iterOne = True
                    if filename in list(all_features.keys()):
                        curr_foc = all_features[filename]
                        key_list = list(curr_foc.feature_output.keys())
                        for token in list(curr_foc.feature_output.keys()):
                            for feature_dict in curr_foc.feature_output[token]:
                                addNone = True
                                feature = feature_dict['feature']
                                output = feature_dict['output']
                                core_feature = [float(feature[0]), float(feature[1])]
                                feature_array = curr_foc.calc2(token, float(feature[0]), float(feature[1]))
                                with open('/Users/abasker/Desktop/MLData.csv',mode='r') as ml_file:
                                    reader = csv.reader(ml_file)
                                    next(reader)
                                    for row in reader:
                                        row_file = row[3]
                                        if row_file == filename:
                                            row_output = row[4]
                                            row_token = row[0]
                                            x_dist = float(row[1])
                                            y_dist = float(row[2])
                                            core_rf = [float(row[1]), float(row[2])]
                                            if iterOne:
                                                row_features = curr_foc.calc2(row_token, x_dist, y_dist)
                                                row_dict = {}
                                                row_dict["Token"] = row_token
                                                row_dict["x-distance"] = x_dist
                                                row_dict["y-distance"] = y_dist
                                                row_dict["filename"] = row_file
                                                row_dict["Output"] = row_output
                                                for i in range(0, num_features):
                                                    k = "feature" + str(i+6)
                                                    row_dict[k] = row_features[i]
                                                writer.writerow(row_dict)
                                            if core_feature == core_rf and row_token == token:
                                                addNone = False
                                iterOne = False
                                if addNone == True:
                                    row_dict = {}
                                    row_dict["Token"] = token
                                    row_dict["x-distance"] = feature[0]
                                    row_dict["y-distance"] = feature[1]
                                    row_dict["filename"] = feature[2]
                                    row_dict["Output"] = [0,0,0,0,1]
                                    for i in range(0, num_features):
                                        key = "feature" + str(i+6)
                                        row_dict[key] = feature_array[i]
                                    writer.writerow(row_dict)






