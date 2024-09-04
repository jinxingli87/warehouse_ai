import re
def clean_email(body):
    pattern = re.compile(r'\nFrom:.*?\nTo:.*?\nSubject:', re.DOTALL) 
    match = re.search(pattern, body)
    if match:
        body = body[:match.start()].strip()

    # For Windows
    pattern = re.compile(r'\r\nFrom:.*?\r\nTo:.*?\r\nSubject:', re.DOTALL) 
    match = re.search(pattern, body)
    if match:
        body = body[:match.start()].strip()


    # For Mac OS
    pattern2 = re.compile(r'\n在20.*?写道：\n', re.DOTALL) 
    match2 = re.search(pattern2, body)
    if match2:
        body = body[:match2.start()].strip()
    
    # For Windows
    pattern2 = re.compile(r'\r\n在20.*?写道：\r\n', re.DOTALL) 
    match2 = re.search(pattern2, body)
    if match2:
        body = body[:match2.start()].strip()
    return body
    