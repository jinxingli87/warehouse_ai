#!/usr/bin/env python
# coding: utf-8

# In[85]:


# run this on terminal:    jupyter nbconvert --to script pod_from_all_v3.4.ipynb
import pandas as pd
from pypdf import PdfReader
import re
from pytesseract import pytesseract
import glob
import fitz
from pprint import pprint
from tqdm import tqdm
from openpyxl import load_workbook
from openpyxl.styles import Font
from fuzzywuzzy import fuzz
import os
from openai import OpenAI
openai_api_key = os.environ.get("OPENAI_API_KEY")
import base64
import requests
import datetime
from pytz import timezone
import numpy as np

from PIL import Image, ImageEnhance, ImageFilter
from sqlalchemy import create_engine
import sqlite3

import easyocr
easyocr_reader = easyocr.Reader(['en'],gpu =  True)

def internal_date_to_pacific_time(internal_date):
    # Convert Unix Time to datetime object
    unix_time = int(internal_date) / 1000  # Convert milliseconds to seconds
    date_obj = datetime.datetime.fromtimestamp(unix_time)

    # Convert datetime object to UTC
    #date_utc = date_obj.astimezone(datetime.timezone.utc)
    date_pt = date_obj.astimezone(timezone('US/Pacific'))

    return str(date_pt)

def count_word_occurrences(text, word_to_count):
    return text.lower().count(word_to_count.lower())

def add_space_before_uppercase(text):
    result = ""
    for char in text:
        if char.isupper():
            result += " " + char
        else:
            result += char
    return result.lstrip()  # Remove the leading space if it starts with an uppercase letter

# Most common Amazon PODs have the following information
table_list = ['Appointment ID','Appointment Reference Code','Destination FC','Status',\
              'Scheduled Time','Freight Type','Load Type','Is Freight Clampable','Trailer Number',\
              'Carrier Requested Delivery Date','Arrival Time','CheckIn Time','Closed Time','Parse Method',\
              'Filename','Comment1','Comment2','Filepath','Unix_Time','Received_Time_PT','Discard','Confirmed']
n_col = len(table_list)


# In[86]:


# table_list_2 = ['Appointment ID', 'Destination FC','Amazon Warehouse','Auctual Arrival Date','Parse Method','Filename','Receive Date','Comment1','Extracted text']
prompt = """ Please extract the following information from the text: Appointment ID, Destination FC, Amazon Warehouse, Auctual Arrival Date.
If a value is not found, please leave it blank.
The returned text should only have two rows, and the sample output is shown as follows:
Appointment ID | Destination FC | Amazon warehouse | Auctual arrival date
9084351999 | LAS1 | LAS1 | 2024/03/15 09:20 PDT
"""

def extract_text_from_image_gpt4(image_path,prompt):
    # encode image to base64
    image_file = open(image_path, "rb")
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4-turbo", #gpt-4o",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 4096
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # check if the response is successful
    if response.status_code != 200:
        print(f"Error: {response.json()}")
        return None
    else:
        extracted_text = response.json()['choices'][0]['message']['content']
    return extracted_text

#table_list_2 = ['Appointment ID', 'Destination FC','Amazon Warehouse','Auctual Arrival Date','Parse Method','Filename','Receive Date','Comment1','Extracted text']
def extract_pod_using_openai(filepath):
    filename = filepath.split('/')[-1]
    
    if filename.endswith('.pdf'):# if filename is a pdf file
        doc = fitz.open(filepath) # open document
        pix = doc[0].get_pixmap(dpi=300)#matrix=mat)  # use 'mat' instead of the identity matrix
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        # enhance the image
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
        img = img.convert("L") # convert to grayscale
        filepath_tmp = 'tmp.jpg'
        img.save(filepath_tmp)
        text_from_gpt4 = extract_text_from_image_gpt4(filepath_tmp,prompt)
        os.remove(filepath_tmp) # remove the temporary file
    elif filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        text_from_gpt4 = extract_text_from_image_gpt4(filepath,prompt)

    values = ['']*5
    if text_from_gpt4 is None:
        values[4] = 'Not a POD from GPT4'
        return values
    
    text_split = text_from_gpt4.split("\n")
    values1 = text_split[-1].split('|') # We use -1 instead of 1, because sometimes the extracted text has 3 rows, with the second row be --|---|---|---
    
    # If values[0] has non number characters, then it is not a POD from Amazon
    if not values1[0].isdigit() or len(values1) != 4:
        values[4] = 'Not a POD from GPT4'
    else:
        values[0:4] = [value.strip() for value in values1]
    return values



# In[87]:


root_dir = '/Users/ljx/Documents/ByteMelodies/data/cts/'
db_path = root_dir + 'pods/pod.db'
engine = create_engine(f'sqlite:///{db_path}')

conn = sqlite3.connect(db_path)
query = """
SELECT a.Filename, a.Filedir,e.Unix_Time, e.Date_UTC
FROM files AS a
LEFT JOIN emails AS e
ON a.EmailId = e.Id
--ORDER BY a.ROWID DESC 
--LIMIT 100
"""
df_files = pd.read_sql_query(query, conn)
df_pods = pd.read_sql_query("SELECT * FROM pods", conn)
#if len(df_files) == len(df_pods):
#    return None


# In[ ]:





# In[88]:


time_formats = [
        r'\d{4}-\d{2}-\d{2}\d{2}:\d{2}[A-Z]{3}',
        r'\d{4}/\d{2}/\d{2}\d{2}:\d{2}[A-Z]{3}',
        r'\d{2}-\d{2}-\d{4}\d{2}:\d{2}[A-Z]{3}',
        r'\d{2}/\d{2}/\d{4}\d{2}:\d{2}[A-Z]{3}']
class POD:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = filepath.split('/')[-1]
        self.img = None
        self.imgL = None
        self.df = pd.DataFrame([['']*n_col], columns=table_list )
        self.df.loc[0,'Filepath'] = filepath
        self.text = ''
        self.text_from_img = ''
        self.text_from_img_easyocr = ''
        if filepath.endswith('.pdf'):
            pdf_reader = PdfReader(filepath)
            page = pdf_reader.pages[0]
            text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False)
            # Why keep extraction_mode="layout"? Because it returns the text line by line.

            lines = text.split('\n') # Othersie, we may replace '/n' with '' in the text
            # remove leading whitespaces in each line
            modified_lines = [re.sub(r'\s{2,}', ' ', line.lstrip()) for line in lines] 
            self.text = '\n'.join(modified_lines)
            
    def extract_text_from_image(self):
        if self.filepath.endswith('.pdf'):
            doc = fitz.open(self.filepath)
            pix = doc[0].get_pixmap(dpi=300)
            img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        elif self.filepath.endswith('.jpg') or self.filepath.endswith('.png') or self.filepath.endswith('.jpeg'):
            img = Image.open(self.filepath)

        #img = img.filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
        try:
            # We use try because sometimes the image_to_osd function may not work properly
            # Rotate the image if necessary
            osd_text = pytesseract.image_to_osd(img)
            osd_text_split = osd_text.split('\n')
            orientation = osd_text_split[1].split(': ')[1]
            if orientation != '0':
                img = img.rotate(int(orientation))
        except:
            pass
        self.img = img
        self.imgL = img.convert("L")
        # It is important to set the language to 'eng' for English, and the page segmentation mode to 6. Otherwise, the OCR may not work properly.
        text = pytesseract.image_to_string(self.imgL, lang='eng', config='--psm 6')
        self.text_from_img = text

    def extract_text_from_image_easyocr(self):
        # Suppose we have already processed extracted_text_from_image using pytesseract, and thus have already have self.imgL
        result = easyocr_reader.readtext(np.array(self.imgL),detail = 1, canvas_size=3600, paragraph = True, add_margin=0.0,x_ths = 0.3,y_ths=0.0)
        text = ''
        len_result = len(result)
        #(horizontal, vertical)
        # top-left, top-right, bottom-right, bottom-left
        for i in range(len_result):
            # If the text is not in the same line as the previous text, add a new line
            # ith top-left horizontal (0,0) < (i-1)th top-right horizontal (1,0), OR ith top-left vertical (0,1) > (i-1)th bottom-left vertical (3,1)
            if (i>1 and (result[i][0][0][0] < result[i-1][0][1][0]) or (result[i][0][0][1] > result[i-1][0][3][1])):
                text += '\n'
            else:
                text += ' '

            text += result[i][1]
        self.text_from_img_easyocr = text
        return None
            

    def correct_pod(self):

        # convert to uppercase
        self.df.loc[0,'Destination FC'] = self.df.loc[0,'Destination FC'].upper()
        self.df.loc[0,'Appointment Reference Code'] = self.df.loc[0,'Appointment Reference Code'].upper()

        replacement_dict = {'I': '1',   'T': '1',   'O': '0',   'S': '5',   'B': '8',   'G': '6',   'Z': '2',   '?': '2'}
        # Check if the last character is in the replacement dictionary
        AFC_code = self.df.loc[0,'Destination FC']
        if AFC_code!='' and AFC_code[-1] in replacement_dict:
            # Replace the last character using the dictionary
            self.df.loc[0,'Destination FC'] = AFC_code[:-1] + replacement_dict[AFC_code[-1]]
        
        ARC = self.df.loc[0,'Appointment Reference Code']
        if ARC != '' and len(ARC)>3 and ARC[3] in replacement_dict:
            self.df.loc[0,'Appointment Reference Code'] = ARC[:3] + replacement_dict[ARC[3]] + ARC[4:]

    
        # For the most common cases, we correct the Appointment Reference Code and Destination FC
        if self.df.loc[0,'Appointment Reference Code'] != '' and self.df.loc[0,'Destination FC'] != '':
            ARC = self.df.loc[0,'Appointment Reference Code']
            AFC = self.df.loc[0,'Destination FC']
            if len(ARC)>0 and len(AFC)>0:
                if len(AFC)<4:
                    self.df.loc[0,'Comment1'] = 'Destination FC length less than 4. Corrected by ARC'
                    self.df.loc[0,'Destination FC'] = ARC[0:4]
                elif ARC[:4] != AFC:
                    if AFC.lower() in self.filename.lower():
                        self.df.loc[0,'Comment1'] = 'ARC was '+ARC[:4]+' ; Should be corrected by Destination FC'
                    elif ARC[0:4].lower() in self.filename.lower():
                        self.df.loc[0,'Comment1'] = 'Destination FC was '+AFC+' ; Corrected by ARC'
                        self.df.loc[0,'Destination FC'] = ARC[0:4]
                    else:
                        self.df.loc[0,'Comment1'] = 'ARC not consistent with Destination FC'


    def pod_from_file(self,text):

        self.df.loc[0,'Comment2'] = text
        text = text.replace(' ','')
        # We check if the text contains the word 'appointment' 2+ times, or 'amazon' 2+ times, or 'proof of delivery' 1+ times. Otherwise, it is not a POD.
        if count_word_occurrences(text,'appointment') >= 2 or count_word_occurrences(text,'amazon') >= 2 or count_word_occurrences(text,'proof of delivery') >= 1:
            text_split = text.split("\n")
            i = 0
            for k,line in enumerate(text_split):
                str_col = table_list[i].replace(" ", "")
                #line = line.replace(" ", "")
                len_col_str = len(str_col)
                if bool(re.match('AppointmentInformation', line, re.I)) or fuzz.partial_ratio('AppointmentInformation', line[0:len('AppointmentInformation')])>80:
                    match = re.search(r'\d{8,}$', text_split[k+1])
                    if match:
                        self.df.iloc[0,0] = match.group(0)
                        i+=1 # if 
                else:
                    if bool(re.match(str_col, line, re.I)) or fuzz.partial_ratio(str_col, line[0:len_col_str])>80:
                        len_t = len(str_col)
                        if len(line) > len_t:
                            if str_col == 'AppointmentID':
                                match = re.search(r'\d{8,}', line[len_t:])
                                self.df.iloc[0,i] = match.group(0) if match else line[len_t:].strip()
                            elif str_col == 'DestinationFC':
                                match = re.search(r'[A-Z]{3,}\d', line[len_t:])
                                self.df.iloc[0,i] = match.group(0) if match else line[len_t:].strip()
                            # elif str_col contains 'Time' or 'Date'
                            elif 'Time' in str_col or 'Date' in str_col:
                                matches = []
                                for time_format in time_formats:
                                    matches.extend(re.findall(time_format, text))
                                self.df.iloc[0,i] = matches[0] if len(matches)>0 else line[len_t:].strip()
                            else:
                                self.df.iloc[0,i] = line[len_t:].strip()
                        i+=1
                if i >= n_col:
                    break
            
            # For files like CR20240503005-CSNU7475121-ABE8-pod.pdf or 0515-GYR2-166873008993 .pdf (yes, there is a space)
            # the Destination FC is Amazon Warehouse, and the Arrival Time is the Actual Arrival Date
            if self.df.loc[0,'Destination FC']=='' or self.df.loc[0,'Arrival Time'] == '':
                for k,line in enumerate(text_split):
                    #line = line.replace(" ", "")
                    if bool(re.match('AmazonWarehouse', line,re.I)) or fuzz.partial_ratio('AmazonWarehouse', line)>80:
                        if len(line) > len('AmazonWarehouse'):
                            match = re.search(r'[A-Z]{3,}\d', line[len('AmazonWarehouse'):])
                            self.df.loc[0,'Destination FC'] = match.group(0) if match else line[len('AmazonWarehouse'):]
                    if bool(re.match('ActualArrivalDate', line,re.I)) or fuzz.partial_ratio('ActualArrivalDate', line)>80:
                        if len(line) > len('ActualArrivalDate'):
                            matches = []
                            for time_format in time_formats:
                                matches.extend(re.findall(time_format, text))
                            self.df.loc[0,'Arrival Time'] = matches[0] if len(matches)>0 else line[len('ActualArrivalDate'):]

            # We restore the 'Freight Type' and 'Load Type'
            self.df.iloc[0,5] = add_space_before_uppercase(self.df.iloc[0,5])
            self.df.iloc[0,6] = add_space_before_uppercase(self.df.iloc[0,6])

            # We standalize the datetime format
            for i in [4,9,10,11,12]:
                
                dt_str = self.df.iloc[0,i].replace(' ','').replace('.',':') 
                # check if dt_str is in format of 'yyyy-mm-dd hh:mm AAA', if yes, we skip the following steps
                # This is because we may use multiple approaches, including text, Tesseract, EasyOCR, or OpenAI 
                # to extract the information.
                if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2} [A-Z]{3}$', dt_str) and len(dt_str)>15:
                    # check if dt_str[0:10] is in format of 'mm/dd/yyyy', 'mm-dd-yyyy' or 'yyyy/mm/dd'. If yes, convert it to 'yyyy-mm-dd'
                    if dt_str[2] == '/' and dt_str[5] == '/':
                        self.df.iloc[0,i] = dt_str[6:10]+'-'+dt_str[0:2]+'-'+dt_str[3:5]+' '+dt_str[10:15]+' '+dt_str[15:]
                    elif dt_str[2] == '-' and dt_str[5] == '-':
                        self.df.iloc[0,i] = dt_str[6:10]+'-'+dt_str[0:2]+'-'+dt_str[3:5]+' '+dt_str[10:15]+' '+dt_str[15:]
                    elif dt_str[4] == '/' and dt_str[7] == '/':
                        self.df.iloc[0,i] = dt_str[0:4]+'-'+dt_str[5:7]+'-'+dt_str[8:10]+' '+dt_str[10:15]+' '+dt_str[15:]
                    else:
                        self.df.iloc[0,i] = dt_str[0:10]+' '+dt_str[10:15]+' '+dt_str[15:]
        else:
            self.df.loc[0,'Parse Method'] = 'Not a POD '+self.df.loc[0,'Parse Method']
            self.df.loc[0,'Comment1'] = 'Lack appointment,amazon,proof of delivery'
        self.correct_pod()


    def extract_pod(self):
        filename = self.filepath.split('/')[-1]
        self.df.loc[0,'Filename'] = filename

        # if the file format is not supported, we skip it
        if not (filename.endswith('.pdf') or filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')):
            self.df.loc[0,'Parse Method'] = 'Not a POD'
            self.df.loc[0,'Comment1'] = 'The file format is not supported'
            return None
        # if filename contains the word 'invoice', we skip it
        if 'invoice' in filename.lower():
            self.df.loc[0,'Parse Method'] = 'Not a POD'
            self.df.loc[0,'Comment1'] = 'The filename contains the word "invoice".'
            return None
        # if the size of the file is less than 20kb, and if the filename begins with 'image' followed by 3 digits, we skip it
        if os.path.getsize(self.filepath) < 20000 and filename[0:5] == 'image' and filename[5:8].isdigit():
            self.df.loc[0,'Parse Method'] = 'Not a POD'
            self.df.loc[0,'Comment1'] = 'The file size is less than 20 kB and the filename begins with image.'
            return None
        
        if len(self.text) > 100:
            text = self.text
            self.df.loc[0,'Parse Method'] = 'Text'
            self.pod_from_file(text)
        
        # Usually it succeeds if the text length > 100. But occasionally, the page may contain text in image. In this case, we use OCR
        if len(self.text)<=100 or (len(self.text)>100 and 'Not a POD' in self.df.loc[0,'Parse Method']):
            self.df.loc[0,'Parse Method'] = 'Tesseract'
            self.df.loc[0,'Comment1'] = ''
            self.extract_text_from_image()
            text = self.text_from_img
            self.pod_from_file(text)

            # If Tesseract fails, we use EasyOCR
            if 'Not a POD' in self.df.loc[0,'Parse Method'] or not self.df.loc[0,'Appointment ID'].isdigit():
                self.df.loc[0,'Appointment ID'] = ''
                self.df.loc[0,'Parse Method'] = 'EasyOCR'
                self.df.loc[0,'Comment1'] = ''
                self.extract_text_from_image_easyocr()
                text = self.text_from_img_easyocr
                self.pod_from_file(text)
                
                # if the length of the extracted text is less than 100, we are for sure that this is not a POD.
                if len(self.text_from_img_easyocr) < 100:
                    self.df.loc[0,'Comment1'] = 'text len < 100'
                    return None
                
                # If the above method fails, we use OpenAI to extract the information
                elif 'Not a POD' in self.df.loc[0,'Parse Method'] or not self.df.loc[0,'Appointment ID'].isdigit():
                    self.df.loc[0,'Appointment ID'] = ''
                    values = extract_pod_using_openai(self.filepath)
                    if values[4] == '':
                        self.df.loc[0,'Appointment ID'] = values[0]
                        self.df.loc[0,'Destination FC'] = values[1] if len(values[1])>=len(values[2]) else values[2]
                        self.df.loc[0,'Arrival Time'] = values[3]
                        self.df.loc[0,'Parse Method'] = 'GPT4'
                    else:
                        self.df.loc[0,'Parse Method'] = 'Not a POD from GPT4'
                        self.df.loc[0,'Appointment ID'] = ''
                    self.correct_pod()

        


# In[89]:


len_prefix = len(root_dir+'pods/'+'2024/')
df_all = pd.DataFrame()
for i_file in tqdm(range(len(df_pods),len(df_files))):
    pdf_file = root_dir+'pods/'+df_files.loc[i_file,'Filedir']+'/'+df_files.loc[i_file,'Filename']
    pod1 = POD(pdf_file)
    pod1.extract_pod()
    pod1.df.loc[0,'Unix_Time'] = df_files.loc[i_file,'Unix_Time']
    pod1.df.loc[0,'Received_Time_PT'] = internal_date_to_pacific_time(df_files.loc[i_file,'Unix_Time'])
    
    
    
    # replace pod1.df columns names from ' ' to '_'
    pod1.df.columns = [col.replace(' ','_') for col in pod1.df.columns]
    # add a column named 'fileId' to pod1.df
    pod1.df['fileId'] = i_file+1
    # move 'fileId' to the first column
    cols = pod1.df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    pod1.df = pod1.df[cols]
    pod1.df.to_sql('pods', engine, if_exists='append', index=False)
    df_all = pd.concat([df_all,pod1.df],ignore_index=True)





# In[ ]:





# In[90]:


engine = create_engine(f'sqlite:///{db_path}')
conn = sqlite3.connect(db_path)
df_pods = pd.read_sql_query("SELECT * FROM pods", conn)

mp_isa_id = {}
for i in range(len(df_pods)):
    # if df_pods.loc[i,'Appointment_ID'] is in df_pods.loc[i,'Filename'], then the 'Confirmed' column is 'Yes'
    isa = str(df_pods.loc[i,'Appointment_ID'])
    if isa != '' and isa in df_pods.loc[i,'Filename']:
        df_pods.loc[i,'Confirmed'] = 'Yes'
    elif df_pods.loc[i,'Parse_Method'] == 'Text':
        df_pods.loc[i,'Confirmed'] = 'Yes'

    if isa != '':
        # if isa not in mp_isa_id, then add it to mp_isa_id
        if isa not in mp_isa_id:
            mp_isa_id[isa] = i
        else:
            id0 = mp_isa_id[isa]

            # We keep the one with Destination_FC information
            if df_pods.loc[id0,'Destination_FC'] == '' and df_pods.loc[i,'Destination_FC'] != '':
                mp_isa_id[isa] = i
                df_pods.loc[id0,'Discard'] = 'Yes'
                df_pods.loc[i,'Discard'] = 'No'
            elif df_pods.loc[id0,'Destination_FC'] != '' and df_pods.loc[i,'Destination_FC'] == '':
                df_pods.loc[i,'Discard'] = 'Yes'
                df_pods.loc[id0,'Discard'] = 'No'
            
            # We keep the one with pdf file and drop the one with jpg/png/jpeg file
            elif df_pods.loc[id0,'Filename'].split('.')[-1] == 'pdf' and df_pods.loc[i,'Filename'].split('.')[-1] != 'pdf':
                df_pods.loc[id0,'Discard'] = 'No'
                df_pods.loc[i,'Discard'] = 'Yes'
            elif df_pods.loc[id0,'Filename'].split('.')[-1] != 'pdf' and df_pods.loc[i,'Filename'].split('.')[-1] == 'pdf':
                mp_isa_id[isa] = i
                df_pods.loc[i,'Discard'] = 'No'
                df_pods.loc[id0,'Discard'] = 'Yes'

            # We keep the one with a status of "closed"
            elif df_pods.loc[id0,'Status'] == 'closed' and df_pods.loc[i,'Status'] != 'closed':
                df_pods.loc[id0,'Discard'] = 'No'
                df_pods.loc[i,'Discard'] = 'Yes'
            elif df_pods.loc[id0,'Status'] != 'closed' and df_pods.loc[i,'Status'] == 'closed':
                mp_isa_id[isa] = i
                df_pods.loc[id0,'Discard'] = 'Yes'
                df_pods.loc[i,'Discard'] = 'No'
            
            # If the two documents are identical, and their size is the same, then we keep the one with the most recent received time
            # the size of df_pods.loc[id0,'Filepath']
            elif os.path.getsize(df_pods.loc[id0,'Filepath']) == os.path.getsize(df_pods.loc[i,'Filepath']):
                df_pods.loc[id0,'Discard'] = 'Yes'
                df_pods.loc[i,'Discard'] = 'No'
                mp_isa_id[isa] = i
                
            else: # adopt the most recent one
                df_pods.loc[id0,'Discard'] = 'First'
                df_pods.loc[i,'Discard'] = 'Second'
                mp_isa_id[isa] = i
    


df_pods.to_sql('pods', engine, if_exists='replace', index=False)


# In[92]:


query = """
SELECT fileId, Appointment_ID, Destination_FC, Arrival_Time, Filename, Filepath, Received_Time_PT, Discard, Confirmed
FROM pods
"""

engine = create_engine(f'sqlite:///{db_path}')
conn = sqlite3.connect(db_path)
df_pods = pd.read_sql_query(query, conn)


df4 = df_pods[(df_pods['Discard'] != 'Yes') & (df_pods['Discard'] != 'First')]
df4 = df4.drop(columns=['Discard'])
# ignore the index for df4
df4 = df4.reset_index(drop=True)


# Read data from the database, and save the monthly table to an HTML file
now = datetime.datetime.now()
yyyy = now.strftime("%Y")
mm = now.strftime("%m")

tstr_start = yyyy+'-'+mm+'-01 00:00:00'
tstr_end   = yyyy+'-'+mm+'-31 23:59:60'

df2 = df4.copy()
df2 = df2.loc[(df2['Received_Time_PT']>=tstr_start) & (df2['Received_Time_PT']<=tstr_end)].reset_index(drop=True)
# select only those 'Appointment_ID' that are not empty
df2 = df2.loc[df2['Appointment_ID']!='' ].reset_index(drop=True)

for i in range(len(df2)):
    df2.loc[i,'Filename'] = f'<a href="{df2.loc[i,"Filepath"][len_prefix:].replace("#","%23").replace("+","%2B") }" target="_blank">{df2.loc[i,"Filename"]}</a>'
    df2.loc[i,'Received_Time_PT'] = df2.loc[i,'Received_Time_PT'][0:19]




# Define a function to apply the red color style to values in Column 'A' if the corresponding value in Column 'C' is greater than 5
def highlight_column_A(row):
    # color = 'red' if row['Confirmed'] == 'Yes' else 'black'
    return ['color: red' if (col == 'Appointment_ID' and row['Confirmed'] == 'Yes') else '' for col in row.index]

df2 = df2.iloc[::-1,:]
df2 = df2.drop(columns=['Filepath'])
styled_df = df2.style.apply(highlight_column_A, axis=1)
# styled_df = styled_df.drop(columns=['Confirmed'])

# Convert the styled dataframe to HTML
# html = styled_df.to_html()
html_table = styled_df.set_table_styles([
    {'selector': 'th', 'props': [('border', '1px solid grey')]},
    {'selector': 'td', 'props': [('border', '1px solid grey')]}
]).to_html()


# Step 4: HTML for the webpage
html_page = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Links</title>
</head>
<body>
    <h1>POD Files {yyyy} {datetime.datetime.strptime(mm, "%m").strftime("%B")}</h1>
    <br><br>
    {html_table}
</body>
</html>
"""

# Optionally, write to an HTML file
with open(root_dir+'pods/'+yyyy+'/pods_'+yyyy+'_'+mm+'_all.html', 'w') as f:
    f.write(html_page)

print("Webpage created successfully!")
print(root_dir+'pods/'+yyyy+'/pods_'+yyyy+'_'+mm+'_all.html')


# In[ ]:




