import base64
import requests
import os
import io
from PIL import Image, ImageEnhance, ImageFilter,ImageDraw, ImageFont
from pytesseract import pytesseract
import pandas as pd
import glob
import fitz
from pprint import pprint
import re
import numpy as np
from tqdm import tqdm
from pypdf import PdfReader
from openpyxl import load_workbook
from openpyxl.styles import Font
from fuzzywuzzy import fuzz
from openai import OpenAI
import easyocr
from datetime import datetime
import difflib
import cv2
import PyPDF2

openai_api_key = os.environ.get("OPENAI_API_KEY")
root_dir = '/Users/ljx/Documents/ByteMelodies/data/cts/'
def get_afc_set():
    df = pd.read_excel(root_dir+'Amazon-Fulfillment-Centers-Warehouse-Codes_USA.xlsx')
    AFC_set = set(df['CODE'])
    return AFC_set

def remove_duplicates_keep_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

split_punctuation = r'\s|,|/|-|:|\.|\+|\'|\"|\t' # added on 04/22/2024. Sometimes the GPT4 output has single or double quotes.
pattern_ISA = r'^\d{8,}$'
pattern_AFC = r'^[A-Z]{3}\d$'
col_names = ['Is BOL', 'BOL Number', 'Container Number','AFC Code','Is End','Total Weight (KGS)','Is ISA','ISA Number',\
             'Orentation','Pallet OCR','Pallet GPT4','Pallet calculated','Date','Comment1','Comment2','Comment3']
i_is_bol, i_bol_number, i_container_number,i_afc_code, i_is_end, i_total_weight, i_is_isa, i_isa_number,\
      i_orentation, i_pallet_ocr, i_pallet_gpt4, i_pallet_calculated, i_date, i_comment1,i_comment2,i_comment3 = range(len(col_names))

# check if a string is a number (including float number that has a decimal point)
def is_number(s):
    try:
        float(s)  # Try converting the string to float
        return True
    except ValueError:
        return False

dict_letter_to_number = {'I': '1', 'l':'1', 'L':'1',  'T': '1',   'O': '0',  'o':'0',  'S': '5', 's':'5',  'B': '8',   'G': '6',   'g':9,  'Z': '2',  'z':'2', '?': '2'}
dict_number_to_letter = {'1': 'I',   '0': 'O',   '5': 'S',   '8': 'B',   '6': 'G',   '2': 'Z'}
def correct_AFC_code(AFC_code):
    if (len(AFC_code) < 4):
        return AFC_code
    afc_list = list(AFC_code)
    for i in range(3):
        if afc_list[i] in dict_number_to_letter:
            afc_list[i] = dict_number_to_letter[afc_list[i]]
    if afc_list[-1] in dict_letter_to_number:
        afc_list[-1] =  dict_letter_to_number[afc_list[-1]]
    return ''.join(afc_list)

def correct_BOL_number(bol_number):
    if len(bol_number) < 2:
        return bol_number
    bol_number = list(bol_number.upper())
    for i in range(2):
        if bol_number[i] in dict_number_to_letter:
            bol_number[i] = dict_number_to_letter[bol_number[i]]
    for i in range(2, len(bol_number)):
        if bol_number[i] in dict_letter_to_number:
            bol_number[i] = dict_letter_to_number[bol_number[i]]

    # Usually the BOL nubmer is like 'CD2024052600007', but sometimes the letter 'D' is recognized as 'O' or '0'
    if len(bol_number)>2 and bol_number[0]=='C' and bol_number[1]=='O' and bol_number[2]=='2':
        bol_number[1] = 'D'
    return ''.join(bol_number)

def correct_container_number(container_number):
    container_number = container_number.upper()
    # Sometimes the extracted container number starts with an extra 'I' because it is in a table. 
    # Remove the leading 'I' if the first 5 characters are all letters.
    if container_number[0] =='I' and re.match(r'[A-Z]{5}', container_number):
        container_number = container_number[1:]

    container_number = list(container_number.upper())
    for i in range(4):
        if container_number[i] in dict_number_to_letter:
            container_number[i] = dict_number_to_letter[container_number[i]]
    for i in range(4, len(container_number)):
        if container_number[i] in dict_letter_to_number:
            container_number[i] = dict_letter_to_number[container_number[i]]
    return ''.join(container_number)

def extract_text_from_image_gpt4o(image_path):
    # encode image to base64
    image_file = open(image_path, "rb")
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "Please extract all printed English text from the image line by line."
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

    extracted_text = response.json()['choices'][0]['message']['content']
    return extracted_text

def extract_pallet_number_from_image_gpt4o(image_path):
    # encode image to base64
    image_file = open(image_path, "rb")
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": """
                This image contains a handwritten number of pallets. It contains a number followed by a letter 'P'. 
                Please extract the full characters from the image in format similar to "Extracted text: 3P". 

                Examples of the returned text:
                Extracted text: 3P
                Extracted text: 12P
                Extracted text: 1P
                """
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
        "max_tokens": 1024
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    extracted_text = response.json()#['choices'][0]['message']['content']
    return extracted_text


def extract_bol_info(text):
    # values = ['']*len(col_names)
    text_upper = [x.upper().replace('\t',' ') for x in text]
    text_upper_no_space = [x.replace(' ','') for x in text_upper]
    
    
    bill_of_lading_number = ""
    container_number = ""
    AFC_code = ""

    # Search for the Amazon FC code
    for j,line in enumerate(text_upper):
        if ('SHIPTO') in line.replace(' ','') and j+1 < len(text_upper):
            line1 = text_upper[j+1]
            split_list = re.split(split_punctuation, line1)
            split_list = [x for x in split_list if x]
            for s in split_list:
                match2 = re.search(pattern_AFC, s)
                if match2:
                    AFC_code = match2.group()

            if AFC_code == "":
                # extract the text following "NAME" in split_list
                for i in range(len(split_list)):
                    if split_list[i] == 'NAME' and i+1 < len(split_list):
                        AFC_code = split_list[i+1]

            # Sometimes, 'LAS1' is recognized as 'LAST' or 'LASI'; Sometimes, 'LIT2' is recognized as 'Litz', funny!
            if AFC_code == "":
                for s in split_list:
                    if (len(s)==4) and re.match(r'[A-Z]{3}', s):
                        AFC_code = s
    AFC_code = correct_AFC_code(AFC_code)



    # Search for the BOL number
    for line in text_upper:
        if ("BILL OF LADING" in line) and (len(line)>20):
            bill_of_lading_number = re.findall(r'BILL OF LADING[:;]\s*([A-Z0-9]+)', line)
            if bill_of_lading_number:
                bill_of_lading_number = bill_of_lading_number[0]
            else:
                bill_of_lading_number = 'None'#line
            break

    if bill_of_lading_number == "":
        for line in text_upper:
            if ("LADING" in line) and (len(line)>12):
                bill_of_lading_number = re.findall(r'LADING[:;]\s*([A-Z0-9]+)', line)
                if bill_of_lading_number:
                    bill_of_lading_number = bill_of_lading_number[0]
                else:
                    bill_of_lading_number = 'None'#line
                if bill_of_lading_number != 'None':
                    break
    bill_of_lading_number = correct_BOL_number(bill_of_lading_number)

    # Search for the container number
    # container_patterns = [r'CONTAINER NUMBER[:;]\s*\|*\s*([A-Z0-9]+)', r'CONTAINER_NUMBER[:;]\s*\|*\s*([A-Z0-9]+)']
    container_patterns = [r'CONTAINERNUMBER[:;]*\|*([A-Z0-9]+)', r'CONTAINER_NUMBER[:;]*\|*([A-Z0-9]+)']
    container_number = 'None'
    for j,line in enumerate(text_upper_no_space):
        if ("CONTAINER" in line and j < len(text_upper)-2):
            for container_pattern in container_patterns:
                container_numbers = re.findall(container_pattern, line)
                if container_numbers:
                    split_list = re.split(split_punctuation, container_numbers[0])
                    container_number = split_list[-1] if len(split_list[-1]) > 8 else 'None'
                if container_number != 'None':
                    break
            
            # if container_number is not found in the current line, search the next two lines
            if container_number == 'None':
                if j<len(text_upper)-1:
                    line1 = line + ' ' + text_upper[j+1]
                if j<len(text_upper)-2:
                    line1 += ' ' + text_upper[j+2]
                for container_pattern in container_patterns:
                    container_numbers = re.findall(container_pattern, line1)
                    if container_numbers:
                        split_list = re.split(split_punctuation, container_numbers[0])
                        container_number = split_list[-1] if len(split_list[-1]) > 8 else 'None'
                    if container_number != 'None':
                        break
            
            if container_number == 'None':
                # find all line numbers that contain the string "Lading"
                lading_line_numbers = [index for index, line in enumerate(text_upper) if 'LADING' in line]
                if len(lading_line_numbers) > 0:
                    result = []
                    for p in range(lading_line_numbers[-1],j):
                        pattern_container = r'[A-Z]{4}\d{6,}'
                        match2 = re.search(pattern_container, text_upper[p])
                        if match2:
                            result.append(match2.group())
                    if len(result) > 0:
                        container_number = result[-1]
            break
    container_number = correct_container_number(container_number)
    if container_number == 'None':
        container_number = ''
    return bill_of_lading_number, container_number, AFC_code


def extract_total_weight(text):
    text_upper = [x.upper() for x in text]
    # Find if the total weight information is in the text
    indices = [index for index, line in enumerate(text_upper) if 'TOTAL' in line]
    indices_sig = [index for index, line in enumerate(text_upper) if 'SIGNATURE' in line]
    total_weight = '-1'
    if (len(indices) > 0):
        words = text_upper[indices[0]].replace('\t',' ').split(' ')
        i=0
        while i < len(words) and 'TOTAL' not in words[i]:
            i += 1
            
        if (i+2 < len(words)):
            total_weight = words[i+2]
        elif (i+1 < len(words)):
            total_weight = words[i+1]
        
        total_weight = total_weight if is_number(total_weight) else '-1'
    
    comment = ''
    if (len(indices) > 1):
        comment += 'Multiple Totals '
    return total_weight, comment

# Remmove image border. 
# The input is a numpy array of the image. 
# The output is also a numpy array of the image.
def image_remove_border(pixE):
    pixc = pixE.copy()
    gray = cv2.cvtColor(pixE,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(pixc, [c], -1, (255,255,255), 5)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(pixc, [c], -1, (255,255,255), 5)

    return pixc


def save_first_page(input_pdf_path, output_pdf_path):
    with open(input_pdf_path, 'rb') as input_pdf_file:
        pdf_reader = PyPDF2.PdfReader(input_pdf_file)
        
        first_page = pdf_reader.pages[0]
        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_page(first_page)
        with open(output_pdf_path, 'wb') as output_pdf_file:
            pdf_writer.write(output_pdf_file)