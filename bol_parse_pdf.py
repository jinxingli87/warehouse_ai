import base64
from pytesseract import pytesseract
import pandas as pd
import glob, fitz, re
from pprint import pprint
import numpy as np
from tqdm import tqdm
import easyocr
from pypdf import PdfReader
from openpyxl import load_workbook
from openpyxl.styles import Font
from fuzzywuzzy import fuzz
import os, pickle, io, sys, json
from openai import OpenAI
openai_api_key = os.environ.get("OPENAI_API_KEY")
import base64,requests, time, pytz # track CPU run time; Timezone
import cts
from datetime import datetime
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
# from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import mimetypes
# mimetypes.init()
# mimetypes.knownfiles
import zipfile,sqlite3
from sqlalchemy import create_engine
from PIL import Image, ImageEnhance, ImageFilter,ImageDraw, ImageFont
from cts_email import gmail_authenticate, search_messages,search_msg_subject

sys.path.insert(0, 'email') # add email folder to path
# import gdrive_jli
# gdrive_jli.main() # Sync Google Drive Folder to Local Folder

easyocr_reader = easyocr.Reader(['en'],gpu =  True)
easyocr_reader_ch = easyocr.Reader(['ch_sim','en'],gpu =  True)
#AFC_set = cts.get_afc_set()



import gdrive_jli
gdrive_jli.main() # Sync Google Drive Folder to Local Folder



class PdfFile:
    def __init__(self,filepath) -> None:
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.dir_out = os.path.dirname(filepath)+'/output/'+'.'.join(self.filename.split('.')[:-1])+'/'
        self.json = self.dir_out+'text.json'
        self.doc = fitz.open(filepath)
        self.num_pages = self.doc.page_count
        self.rotation = [0]*self.num_pages
        self.imgEs = [None]*self.num_pages
        self.imgBs = [None]*self.num_pages
        self.imgELs = [None]*self.num_pages
        self.imgBLs = [None]*self.num_pages

        self.text_easyocr = [None]*self.num_pages
        self.text_easyocr_b = [None]*self.num_pages
        self.text_easyocr_c = [None]*self.num_pages
        self.text_tesseract = [None]*self.num_pages
        self.text_tesseract_b = [None]*self.num_pages
        self.text_tesseract_c = [None]*self.num_pages
        
        self.text_openai = [None]*self.num_pages
        # tmp = [[1, 2, 3] for _ in range(4)]
        self.scac_mtgh = [[0,0,0,0,0,0] for _ in range(self.num_pages)] # coordinates for Key words SCAC, MTGH and 'Pro Number'
        self.signature = [[0,0,0,0,0,0] for _ in range(self.num_pages)] # coordinates for Key words signature; 3 signatures at most
        # Absotely not to use the following line, it will create a list of the same object
        # self.scac_mtgh = [[0,0,0,0,0,0]]*self.num_pages 
        # self.signature = [[0,0,0,0,0,0]]*self.num_pages 
    
    def img_process(self,page_num):
        img_E_path = self.dir_out+'page_'+str(page_num)+'_enhanced.png'
        if os.path.exists(img_E_path):
            imgE = Image.open(img_E_path)
            self.imgEs[page_num] = imgE
            imgB = Image.open(self.dir_out+'page_'+str(page_num)+'_remove_boarder.png')
            self.imgBs[page_num] = imgB
            imgEL = imgE.convert('L')
            self.imgELs[page_num] = imgEL
            imgBL = imgB.convert('L')
            self.imgBLs[page_num] = imgBL
            return

        page = self.doc[page_num]
        image_list = page.get_images(full=True)
        base_image = self.doc.extract_image(image_list[0][0])

        #base_image = page.extract_image(image_list[0][0])
        img = Image.open(io.BytesIO(base_image["image"]))

        # Rotate the image if necessary
        try:
            osd_text = pytesseract.image_to_osd(img)
            osd_text_split = osd_text.split('\n')
            orientation = osd_text_split[1].split(': ')[1]
            if orientation != '0':
                #print('Rotate the image by:', orientation, 'degrees.  ', filepath.split('/')[-1])
                img = img.rotate(int(orientation))
                self.rotation[page_num] = int(orientation)
            #img = img.filter(ImageFilter.MedianFilter())
        except:
            pass
        enhancer = ImageEnhance.Contrast(img)
        imgE = enhancer.enhance(2)
        #pixL = np.array(imgL)
        pixE = np.array(imgE)
        pixB = cts.image_remove_border(pixE)
        imgB = Image.fromarray(pixB)
        imgEL = imgE.convert('L') # convert image to grayscale
        imgBL = imgB.convert('L') # convert image to grayscale
        #img = imgE.convert('1') # convert image to binary, i.e., black and white
        self.imgEs[page_num] = imgE
        self.imgBs[page_num] = imgB
        self.imgELs[page_num] = imgEL
        self.imgBLs[page_num] = imgBL
        # save imgE, imgL, pixB
        if not os.path.exists(self.dir_out):
            os.makedirs(self.dir_out)
        imgE.save(self.dir_out+'page_'+str(page_num)+'_enhanced.png')
        imgB.save(self.dir_out+'page_'+str(page_num)+'_remove_boarder.png')
        #imgL.save(self.dir_out+self.filename + '_page_'+str(page_num)+'_gray.png')


    def text_extract_easyocr(self,page_num):
        if self.imgELs[page_num] is None:
            self.img_process(page_num)
        result = easyocr_reader.readtext(np.array(self.imgELs[page_num]),detail = 1, paragraph = False, canvas_size=3600,add_margin=0.0)
        text = ''
        len_result = len(result)
        i_signature = 0
        for i in range(len_result):
            if 'SCAC' in result[i][1].upper():
                self.scac_mtgh[page_num][0] = int(result[i][0][3][0]) # Bottom left horizontal
                self.scac_mtgh[page_num][1] = int(result[i][0][3][1]) # Bottom left vertical
            elif 'MTGH' in result[i][1].upper():
                self.scac_mtgh[page_num][2] = int(result[i][0][3][0])
                self.scac_mtgh[page_num][3] = int(result[i][0][3][1])
            elif 'PRO' in result[i][1].upper() and 'NUMBER' in result[i][1].upper():
                self.scac_mtgh[page_num][4] = int(result[i][0][0][0]) # Top left horizontal
                self.scac_mtgh[page_num][5] = int(result[i][0][0][1])
            elif 'SIGNATURE' in result[i][1].upper() and i_signature < 3:
                self.signature[page_num][2*i_signature  ] = int(result[i][0][0][0]) # Top left horizontal
                self.signature[page_num][2*i_signature+1] = int(result[i][0][0][1]) # Top left vertical
                i_signature += 1
            
            # If the text is not in the same line as the previous text, add a new line
            # ith top-left horizontal (0,0) < (i-1)th top-right horizontal (1,0), OR ith top-left vertical (0,1) > (i-1)th bottom-left vertical (3,1)
            if (i>1 and (result[i][0][0][0] < result[i-1][0][1][0]) or (result[i][0][0][1] > result[i-1][0][3][1])):
                text += '\n'
            else:
                text += '\t'
            text += result[i][1]
        self.text_easyocr[page_num] = text
        return text
    
    def text_extract_easyocr_b(self,page_num):
        if self.imgBLs[page_num] is None:
            self.img_process(page_num)
        result = easyocr_reader.readtext(np.array(self.imgBLs[page_num]),detail = 1, paragraph = False, canvas_size=3600,add_margin=0.0)
        text = ''
        len_result = len(result)
        for i in range(len_result):
            if (i>1 and (result[i][0][0][0] < result[i-1][0][1][0]) or (result[i][0][0][1] > result[i-1][0][3][1])):
                text += '\n'
            else:
                text += '\t'
            text += result[i][1]
        self.text_easyocr_b[page_num] = text
        return text
    
    def text_extract_easyocr_c(self,page_num):
        if self.imgBLs[page_num] is None:
            self.img_process(page_num)
        result = easyocr_reader_ch.readtext(np.array(self.imgBLs[page_num]),detail = 1, paragraph = False, canvas_size=3600,add_margin=0.0)
        text = ''
        len_result = len(result)
        i_signature = 0
        for i in range(len_result):
            if 'SCAC' in result[i][1].upper():
                self.scac_mtgh[page_num][0] = int(result[i][0][3][0]) # Bottom left horizontal
                self.scac_mtgh[page_num][1] = int(result[i][0][3][1]) # Bottom left vertical
            elif 'MTGH' in result[i][1].upper():
                self.scac_mtgh[page_num][2] = int(result[i][0][3][0])
                self.scac_mtgh[page_num][3] = int(result[i][0][3][1])
            elif 'PRO' in result[i][1].upper() and 'NUMBER' in result[i][1].upper():
                self.scac_mtgh[page_num][4] = int(result[i][0][0][0]) # Top left horizontal
                self.scac_mtgh[page_num][5] = int(result[i][0][0][1])
            elif 'SIGNATURE' in result[i][1].upper() and i_signature < 3:
                self.signature[page_num][2*i_signature  ] = int(result[i][0][0][0]) # Top left horizontal
                self.signature[page_num][2*i_signature+1] = int(result[i][0][0][1]) # Top left vertical
                i_signature += 1

            if (i>1 and (result[i][0][0][0] < result[i-1][0][1][0]) or (result[i][0][0][1] > result[i-1][0][3][1])):
                text += '\n'
            else:
                text += '\t'
            text += result[i][1]
        self.text_easyocr_c[page_num] = text
        return text
    
    
    def text_extract_tesseract(self,page_num):
        if self.imgELs[page_num] is None:
            self.img_process(page_num)
        text = pytesseract.image_to_string(self.imgELs[page_num], lang = 'eng', config='--psm 6') # lang = 'eng+chi_sim'
        self.text_tesseract[page_num] = text
        return text
    
    def text_extract_tesseract_b(self,page_num):
        if self.imgBLs[page_num] is None:
            self.img_process(page_num)
        text = pytesseract.image_to_string(self.imgBLs[page_num], lang = 'eng', config='--psm 6') # lang = 'eng+chi_sim'
        self.text_tesseract_b[page_num] = text
        return text
    

    def text_extract_tesseract_c(self,page_num):
        if self.imgELs[page_num] is None:
            self.img_process(page_num)
        text = pytesseract.image_to_string(self.imgELs[page_num], lang = 'eng+chi_sim', config='--psm 6') # lang = 'eng+chi_sim'
        self.text_tesseract_c[page_num] = text
        return text
    
    def process_all_pages(self):
        # if os.path.exists(self.json) and (time.time() - os.path.getmtime(self.json)) > 186400:
        #         os.remove(self.json)
        if os.path.exists(self.json):
            with open(self.json, 'r') as f:
                data = json.load(f)
                self.text_easyocr = data['text_easyocr']
                self.text_tesseract = data['text_tesseract']
                self.text_easyocr_b = data['text_easyocr_b']
                self.text_tesseract_b = data['text_tesseract_b']
                self.text_easyocr_c = data['text_easyocr_c'] if 'text_easyocr_c' in data else [None]*self.num_pages
                self.text_tesseract_c = data['text_tesseract_c'] if 'text_tesseract_c' in data else [None]*self.num_pages
                self.rotation = data['rotation']
                
                if 'text_tesseract_c' not in data:
                    for i in range(self.num_pages):
                        self.text_extract_tesseract_c(i)
                    dic1 = {x: self.__dict__[x] for x in ("num_pages","rotation","text_easyocr","text_tesseract","text_easyocr_b","text_easyocr_c","text_tesseract_b","text_tesseract_c","text_openai","scac_mtgh","signature")}
                    with open(self.dir_out+'text.json', 'w') as f:
                        json.dump(dic1, f, default=str,indent=4,sort_keys=True, separators=(',', ':'))
                    print("Written to JSON file", self.json)

                # if 'scac_mtgh' not in data:
                #     for i in range(self.num_pages):
                #         self.text_extract_easyocr(i)
                #     dic1 = {x: self.__dict__[x] for x in ("num_pages","rotation","text_easyocr","text_tesseract","text_easyocr_b","text_easyocr_c","text_tesseract_b","text_tesseract_c","text_openai","scac_mtgh","signature")}
                #     with open(self.dir_out+'text.json', 'w') as f:
                #         json.dump(dic1, f, default=str,indent=4,sort_keys=True, separators=(',', ':'))
                #     print("Written to JSON file", self.json)

        else:
            for i in range(self.num_pages):
                if os.path.exists(self.dir_out+'page_'+str(i)+'_enhanced.png'):
                    self.imgEs[i] = Image.open(self.dir_out+'page_'+str(i)+'_enhanced.png')
                    self.imgBs[i] = Image.open(self.dir_out+'page_'+str(i)+'_remove_boarder.png')
                    self.imgELs[i] = self.imgEs[i].convert('L')
                    self.imgBLs[i] = self.imgBs[i].convert('L')
                else:
                    self.img_process(i)
                self.text_extract_easyocr(i)
                self.text_extract_easyocr_b(i)
                self.text_extract_easyocr_c(i)
                self.text_extract_tesseract(i)
                self.text_extract_tesseract_b(i)
                self.text_extract_tesseract_c(i)
            dic1 = {x: self.__dict__[x] for x in ("num_pages","rotation","text_easyocr","text_tesseract","text_easyocr_b","text_easyocr_c","text_tesseract_b","text_tesseract_c","text_openai","scac_mtgh","signature")}
            with open(self.dir_out+'text.json', 'w') as f:
                json.dump(dic1, f, default=str,indent=4,sort_keys=True, separators=(',', ':'))



def correct_letter_to_digit(text, index):
    mp = {'O':'0','I':'1','L':'1','Z':'2','S':'5','B':'8','Q':'0','D':'0','G':'6','A':'4','T':'1'}
    if type(index) == int:
        index = [index]
    for i in index:
        if text[i] in mp:
            text = text[:i] + mp[text[i]] + text[i+1:]
    return text

def correct_digit_to_letter(text, index):
    mp = {'0':'O','1':'I','2':'Z','5':'S','8':'B','6':'G','4':'A'}
    if type(index) == int:
        index = [index]
    for i in index:
        if text[i] in mp:
            text = text[:i] + mp[text[i]] + text[i+1:]
    return text







# re.findall is sometimes better than re.search because it can find all the matches in the text
# Example:
# pattern1 = r'.*?([A-Za-z]{3}\d)[:;]\s*(\d{7,})' # pattern like "IDN9: 12345678*"
# line1 = 'IDN9: 12345678    cl beautiful   LAX1: 98776382 dk'
# match = re.search(pattern1, line1)
# print(match.group()) # IDN9: 12345678
# isa_all = re.findall(pattern1,line1)
# print(isa_all) # [('IDN9', '12345678'), ('LAX1', '98776382')]

split_punctuation = r'\s|,|/|-|:|\.|\+|\'|\"' # added on 04/22/2024. Sometimes the GPT4 output has single or double quotes.
pattern_ISA = r'^\d{8,}$'
pattern_AFC = r'^[A-Z]{3}\d$'

def extract_info_by_pattern(text):
    # Sometimes the AFC can be mistekenly recognized as lower case letters
    pattern1 = r'.*?([A-Za-z]{3}\d)[:;]\s*(\d{8,})' # pattern like "IDN9: 12345678*"
    pattern2 = r'.*?([A-Za-z]{3}\d)\s*(\d{8,})'  # pattern like "IDN9  12345678*"
    pattern3 = r'^.*?(\d{8,})\s+at\s+([A-Za-z]{3}\d).*$' # pattern like "12345678* at IDN9"
    pattern4 = r'.*?([A-Za-z]{4})[:;]\s*(\d{8,})' # pattern like "IDNS: 12345678*", sometimes the digits may be regonized as letters
    pattern5 = r'.*?([A-Za-z]{4})\s*(\d{8,})'  # pattern like "IDNS  12345678*"
    patterns = [pattern1, pattern2,pattern3,pattern4,pattern5]

    ISAs = []
    AFCs = []
    line = text.replace('\n',' ') # Why this? The OCR results may have line breaks.
    for pattern in patterns:
        results = re.findall(pattern, line)
        for result in results:
            if pattern != pattern3:
                AFC, isa_number = result
            else:
                isa_number, AFC = result
            
            if AFC != "":
                AFC = correct_letter_to_digit(AFC.upper(), 3)

            # A special container number is CSWY1234567890, which is 13 digits long, while a typical container number is like CSNU1234567, which is 11 digits long.
            # We need to exclude such a exception.
            if (AFC == 'CSWY' or AFC == 'SWY2'):
                continue
            ISAs.append(isa_number)
            AFCs.append(AFC.upper())
    ISAs = cts.remove_duplicates_keep_order(ISAs)
    AFCs = cts.remove_duplicates_keep_order(AFCs)
    AFCs = [cts.correct_AFC_code(afc) for afc in AFCs]
    return ISAs, AFCs


# If the word 'ISA' is found in the text, extract the information
def extract_info_by_isa(text):
    text_split = text.split('\n')
    ISAs = []
    AFCs = []
    for line in text_split:
        if 'ISA' in line:
            # The minimum length of ISA number is actually 8. We should not set it to 7 for flexible, because container numbers have 7 numbers.
            isa_all = re.findall(r'\d{8,}', line)
            for isa1 in isa_all:
                ISAs.append(isa1)

        split_list = re.split(split_punctuation, line)
        split_list = [x for x in split_list if x]
        
        for s in split_list:
            match = re.search(pattern_AFC, s)
            if match:
                AFCs.append(match.group())
    # remove duplicates in ISAs and AFCs
    ISAs = cts.remove_duplicates_keep_order(ISAs)
    AFCs = cts.remove_duplicates_keep_order(AFCs)
    AFCs = [cts.correct_AFC_code(afc) for afc in AFCs]
    return ISAs, AFCs


# If the string 'Appintment information' is found in the text, extract the information
def extract_info_by_appointment_information(text):
    text_split = text.split("\n")
    ISAs = []
    AFCs = []
    # text is already in UPPER case
    for j,line in enumerate(text_split):
        #if bool(re.match('Appointment Information', line, re.I)) :
        if bool('APPOINTMENT INFORMATION' in line) or fuzz.partial_ratio('APPOINTMENT INFORMATION', line)>80:
            # Sometimes, there could be noise characters ahead of or behind the ISA number
            if j+1 < len(text_split):
                isa_all = re.findall(r'\d{8,}', text_split[j+1])
                for isa1 in isa_all:
                    ISAs.append(isa1)
        else:
            if bool('DESTINATION FC' in line) or fuzz.partial_ratio('DESTINATION FC', line)>80:
                split_list = re.split(split_punctuation, line)
                split_list = [x for x in split_list if x]
                for s in split_list:
                    match = re.search(pattern_AFC, s)
                    if match:
                        AFCs.append(match.group())
                
                # if not found in the same line, search the next line
                if len(AFCs) == 0 and j+1 < len(text_split):
                    split_list = re.split(split_punctuation, text_split[j+1])
                    split_list = [x for x in split_list if x]
                    for s in split_list:
                        match = re.search(pattern_AFC, s)
                        if match:
                            AFCs.append(match.group())
                
                # if not found in the next line, search the previous line
                if len(AFCs) == 0 and j-1 >= 0:
                    split_list = re.split(split_punctuation, text_split[j-1])
                    split_list = [x for x in split_list if x]
                    for s in split_list:
                        match = re.search(pattern_AFC, s)
                        if match:
                            AFCs.append(match.group())
    ISAs = cts.remove_duplicates_keep_order(ISAs)
    AFCs = cts.remove_duplicates_keep_order(AFCs)
    AFCs = [cts.correct_AFC_code(afc) for afc in AFCs]
    return ISAs, AFCs

def extract_info_by_appointment_id(text):
    text_split = text.split('\n')
    ISAs = []
    AFCs = []
    for i,line in enumerate(text_split):
        if "APPOINTMENT ID" in line:# or fuzz.partial_ratio('APPOINTMENT ID', line)>90:
            isa_all = re.findall(r'\d{8,}', line)
            for isa in isa_all:
                ISAs.append(isa)
            # if not found in the same line, search the previous line
            if len(ISAs) == 0 and i>0:
                isa_all = re.findall(r'\d{8,}', text_split[i-1])
                for isa in isa_all:
                    ISAs.append(isa)

        if "DESTINATION FC" in line or fuzz.partial_ratio('DESTINATION FC', line)>80:
            split_list = re.split(split_punctuation, line)
            split_list = [x for x in split_list if x]
            for s in split_list:
                match = re.search(pattern_AFC, s)
                if match:
                    AFCs.append(match.group())
    ISAs = cts.remove_duplicates_keep_order(ISAs)
    AFCs = cts.remove_duplicates_keep_order(AFCs)
    AFCs = [cts.correct_AFC_code(afc) for afc in AFCs]
    return ISAs, AFCs

def extract_info_by_default(text):
    text_split = text.split('\n')
    ISAs = []
    AFCs = []
    for line in text_split:
        split_list = re.split(split_punctuation, line)
        # split_list = [x for x in split_list if x]
        for s in split_list:
            match1 = re.search(pattern_ISA, s)
            if match1:
                ISAs.append(match1.group())
                
            match2 = re.search(pattern_AFC, s)
            if match2:
                AFCs.append(match2.group())

    ISAs = cts.remove_duplicates_keep_order(ISAs)
    AFCs = cts.remove_duplicates_keep_order(AFCs)
    AFCs = [cts.correct_AFC_code(afc) for afc in AFCs]
    return ISAs, AFCs

def extract_pod_info(text):
    text_upper = text.upper()
    Parse_method = ''
    if "1P2D" in text_upper or "1P2D" in text_upper:
        ISAs, AFCs = extract_info_by_pattern(text_upper)
        Parse_method = 'Pattern'
    elif "ISA" in text_upper:
        ISAs, AFCs = extract_info_by_isa(text_upper)
        Parse_method = 'ISA'
    elif "APPOINTMENT INFORMATION" in text_upper or fuzz.partial_ratio('APPOINTMENT INFORMATION', text_upper)>80:
        ISAs, AFCs = extract_info_by_appointment_information(text_upper)
        Parse_method = 'Appointment Information'
    elif "APPOINTMENT ID" in text_upper:
        ISAs, AFCs = extract_info_by_appointment_id(text_upper)
        Parse_method = 'Appointment ID'
    else:
        ISAs, AFCs = extract_info_by_pattern(text_upper)
        Parse_method = 'Pattern'

    if (len(ISAs) == 0) or (len(AFCs) == 0):
        ISAs2, AFCs2 = extract_info_by_default(text_upper)
        if len(ISAs2)>0 and len(AFCs2)>0:
            ISAs = ISAs2
            AFCs = AFCs2
            Parse_method = 'Default' # The last parse method
        elif len(ISAs)==0 and len(ISAs2)>0:
            ISAs = ISAs2
            Parse_method = 'Default' # The last parse method
    return ISAs, AFCs, Parse_method





def main():
    root_dir = '/Users/ljx/Documents/ByteMelodies/data/cts/'
    yyyy = '2024'
    mm = '07'
    dir0 = root_dir+'bols/'+yyyy+'/'+mm+'/'
    pdf_files_all = glob.glob(dir0+'*.pdf')
    pdf_files_all = sorted(pdf_files_all, key=os.path.getmtime, reverse=False)

    db_bol_path = root_dir + 'bols/bol_'+yyyy+'.db'
        
    engine = create_engine(f'sqlite:///{db_bol_path}')

    conn = sqlite3.connect(db_bol_path)
    query = f"SELECT * FROM file_pages"
    df_pages = pd.read_sql_query(query, conn)

    df_bunch = df_pages[df_pages['Page']==0]
    filenames_exist = df_bunch['Filename Scan'].values
    # filenames_exist = []
    pdf_files_new = [file for file in pdf_files_all if os.path.basename(file) not in filenames_exist]

    # pdf_files_new = pdf_files_all

    i_last = df_pages.loc[len(df_pages)-1,'FileId'] if len(df_pages)>0 else -1








    import importlib
    importlib.reload(cts)
    # for i in tqdm(range(116,len(pdf_files_new))):
    columns_list = ['FileId','Page','ISA_count', 'Lading_count','CTS_info_count','Lading_end_count', 'Rotation', 'BOL Number', 'Container Number','AFC Code',\
                                'Total Weight (KGS)','Calculated Pallet','Pallet_gpt4', 'Handwritten', 'Page Category','ISA Number','Parse Method','Filename Scan','Filedir Scan','JSON_path','Filepath','Comment1','Process_Time','Received_Time']
    df_total = pd.DataFrame(columns = columns_list)

    for j in tqdm(range(0,len(pdf_files_new))):
    # for j in tqdm(range(0,1)):
        filepath1 =pdf_files_new[j]
        pdf1 = PdfFile(filepath1)
        # print(filepath1)
        # print(pdf1.num_pages)
        pdf1.process_all_pages()
        json_file = pdf1.dir_out+'text.json'
        with open(json_file) as f:
            data = json.load(f)

        isa_list = ['提', '车', '仓', '厢', '装', '卸', '板','拆', '预约','柜','appointment','clampable','proof of delivery','isa','1p2d','1p3d']
        lading_list = ['lading','address']
        #cts_info_list = ['2300','90040','626-709-3199']
        lading_end_list = ['signature','consignee','ackowledge','placard','emergency','response','equivalent','documentation','vehicle','property']

        stat_file = np.zeros([pdf1.num_pages,6])
        df = pd.DataFrame(columns = columns_list)
        for i in range(pdf1.num_pages):
            text = data["text_easyocr"][i]
            text_lower = text.lower() # For general purpose
            text_easyocr_b = data["text_easyocr_b"][i] # for extracting BOL information, the text from removed bolder image is better
            #text = data["text_tesseract"][i]

            # To identify the ISA information page, we use the text with Chinese characters
            # check if text_easyocr_c is in the JSON file
            if ('text_easyocr_c' not in data) or (data["text_easyocr_c"][i] is None):
                text_for_isa = data["text_easyocr_b"][i]
            else:
                text_for_isa = data["text_easyocr_c"][i]
            isa_count = sum([text_for_isa.lower().count(x) for x in isa_list])

            lading_count = sum([text_lower.count(x) for x in lading_list])
            lading_end_count = sum([text_lower.count(x) for x in lading_end_list])
            cts_info_count = 0
            for line in text_lower.split('\n'):
                cts_info_list = ['2300','eastern','commerce','90040']
                cnt_info_address = sum([line.count(x) for x in cts_info_list])
                cts_info_count += 1 if (cnt_info_address >= 3) else 0
            cts_info_count += text.count('626-709-3199')

            #['ISA_count', 'Lading_count','CTS_info_count','Lading_end_count', 'Rotation', 
            df.loc[i] = [None]*len(df.columns)
            df.loc[i,'FileId'] = j + i_last + 1
            df.loc[i,'Page'] = i
            df.loc[i,'ISA_count'] = isa_count
            df.loc[i,'Lading_count'] = lading_count
            df.loc[i,'CTS_info_count'] = cts_info_count
            df.loc[i,'Lading_end_count'] = lading_end_count
            df.loc[i,'Rotation'] = data["rotation"][i]
            df.loc[i,'Pallet_gpt4'] = ''
            df.loc[i,'Handwritten'] = ''

            df.loc[i,'Filename Scan'] = pdf1.filename
            df.loc[i,'Filedir Scan'] = '/'.join(pdf1.filepath[len(root_dir)+5:].split('/')[0:-1])
            df.loc[i,'JSON_path'] = json_file
            df.loc[i,'Filepath'] = filepath1
            df.loc[i,'Comment1'] = ''
            tstm = os.path.getmtime(filepath1)
            df.loc[i,'Received_Time'] = datetime.fromtimestamp(tstm).strftime("%Y-%m-%d %H:%M:%S")

            net_count = isa_count - lading_count - cts_info_count - lading_end_count

            source = ['text_easyocr','text_easyocr_b','text_tesseract','text_tesseract_b']
            # net_count >= 2: POD pages
            if (net_count) >= 2:
                df.loc[i,'Page Category'] = 'POD'
                ISAs = ''
                i_source = 0
                while (i_source < len(source) and ISAs == ''):
                    ISAs, AFCs, Parse_method = extract_pod_info(data[source[i_source]][i])
                    # print(source[i_source],'\t',ISAs,'\t',AFCs,'\t',Parse_method)
                    i_source += 1

                AFCs = [cts.correct_AFC_code(afc) for afc in AFCs]
                df.loc[i,'ISA Number'] = ','.join(ISAs)
                df.loc[i,'AFC Code'] = ','.join(AFCs)
                df.loc[i,'Parse Method'] = Parse_method

                # In some cases, the POD file may spread over multiple pages, and the key information is on the first page
                # In this case, we modify the Page Category of the following pages to POD_1
                if len(ISAs) == 0 and i>0 and 'POD' in df.loc[i-1,'Page Category']:
                    df.loc[i,'Page Category'] = 'POD_1'
                    df.loc[i,'Parse Method'] = df.loc[i-1,'Parse Method']
            
            # net_count <2: BOL pages
            elif lading_count > 0: # Should be the begin of a BOL
                if cts_info_count > 0:
                    df.loc[i,'Page Category'] = 'BOL_Amazon'
                    bols=['']*4
                    cns = ['']*4
                    afcs = ['']*4
                    container_number = ''
                    bol_number = ''
                    AFC_code = ''
                    for i_source in range(len(source)):
                        bols[i_source], cns[i_source], afcs[i_source] = cts.extract_bol_info(data[source[i_source]][i].split('\n'))
                    for i_source in range(len(source)):
                        # check if cns[i_source] is in this pattern: The first 4 characters are upper-case letters, and the last 7 characters are digits
                        if re.match(r'^[A-Z]{4}\d{7}$', cns[i_source]):
                            container_number = cns[i_source]
                            break
                    if container_number in ['','None','NONE']:
                        container_number = cns[0]
                    
                    for i_source in range(len(source)):
                        if re.match(r'^[A-Z]{3}\d$', afcs[i_source]):
                            AFC_code = afcs[i_source]
                            break
                    if AFC_code in ['','None','NONE']:
                        AFC_code = afcs[0]
                    
                    for i_source in range(len(source)):
                        if re.match(r'^[A-Z]{2}\d{10,}$', bols[i_source]):
                            bol_number = bols[i_source]
                            break
                    if bol_number in ['','None','NONE']:
                        bol_number = bols[0]
                    # bol_number, container_number, AFC_code = cts.extract_bol_info(text_easyocr_b.split('\n'))

                    df.loc[i,'BOL Number'] = bol_number
                    df.loc[i,'Container Number'] = container_number
                    df.loc[i,'AFC Code'] = AFC_code
                    total_weight, comment = cts.extract_total_weight(text_easyocr_b.split('\n'))
                    df.loc[i,'Total Weight (KGS)'] = total_weight if total_weight!='-1' else ''
                    df.loc[i,'Comment1'] += comment
                else:
                    df.loc[i,'Page Category'] = 'BOL_Others'
            else:
                if (i>0):
                    df.loc[i,'Page Category'] = df.loc[i-1,'Page Category']

                    if df.loc[i,'Page Category'] is None:# check if df.loc[i,'Page Category'] is Nontype
                        df.loc[i,'Page Category'] = '_1'
                    elif df.loc[i,'Page Category'][-2] != '_':
                        df.loc[i,'Page Category'] += '_1'
                    total_weight, comment = cts.extract_total_weight(text_easyocr_b.split('\n'))
                    df.loc[i,'Total Weight (KGS)'] = total_weight if total_weight!='-1' else ''
                    df.loc[i,'Comment1'] += comment

            if df.loc[i,'Process_Time'] is None or df.loc[i,'Process_Time'] == '':
                df.loc[i,'Process_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # I need to imporve the detection of the total weight in future
            # sometimes the decimal point is not recognized
            try:
                pallet = float(df.loc[i,'Total Weight (KGS)'])*2.20462/1000.0
                pallet = np.floor(pallet+0.9)
                if pallet < 1:
                    pallet = 1
                df.loc[i,'Calculated Pallet'] = int(pallet)
            except:
                ValueError

        # Remove Nones in the dataframe
        # df = df.fillna('')
        df = df.map(lambda x: '' if pd.isna(x) else x)

    
        i_pods = df[df['Page Category'] == 'POD'].index
        i_bols = df[df['Page Category'] == 'BOL_Amazon'].index

        for i in range(len(i_pods)):
            bol_start = i_pods[i]+1
            bol_end = i_pods[i+1] if i < len(i_pods)-1 else len(df) # not including the last page
            isas = df.loc[i_pods[i],'ISA Number'].split(',')
            afcs = df.loc[i_pods[i],'AFC Code'].split(',')

            if len(isas)!=1 and len(isas)!= len(afcs):
                df.loc[i_pods[i],'Comment1'] += 'Len of ISAs and AFCs unmatch. '
                

            afc_map = {}
            for k in range(len(afcs)):
                if len(isas) > k:
                    afc_map[afcs[k]] = isas[k] 
                else:
                    afc_map[afcs[k]] = isas[-1] if len(isas) > 0 else ''

            for k in range(bol_start,bol_end):
                if df.loc[k,'Page Category'] == 'BOL_Amazon':
                    if len(isas) == 1:
                        df.loc[k,'ISA Number'] = isas[0]
                    else:
                        fuzz_ratio = 0.0
                        for i,afc in enumerate(afcs):
                            ratio = fuzz.ratio(afc,df.loc[k,'AFC Code'])
                            if ratio > fuzz_ratio:
                                fuzz_ratio = ratio
                                df.loc[k,'ISA Number'] = afc_map[afc]
                elif df.loc[k,'Page Category'] == 'BOL_Others':
                    if len(isas) == 1:
                        df.loc[k,'ISA Number'] = isas[0]
                                
        # cancatenate df_total and df, and ignore the index
        
        df_total = pd.concat([df_total,df],axis=0, ignore_index=True)



if __name__ == '__main__':
    main()