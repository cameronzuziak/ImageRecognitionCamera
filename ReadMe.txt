Image Recognition Security Camera
*************************
@author Cameron Zuziak

DESCRIPTION: 
Built for deployment on a raspberrypi, this program utilizes tensorflow and opencv to detect people 
and then texts the user an image of the detected person. 

*************************

set up:
(This is assuming you are on a raspberrypi and have a camera connected the camera input via ribbon cable).

1. Gmail API key:
    In order to send an image via text message, the app utilizes gmail app passwords. 
    you can learn how to set up and create an app password here: https://support.google.com/accounts/answer/185833?hl=en
    Once you have the account set up, you can your account info, including the app password, into the corresponding
    fields of the sendEmail() method or you can set up environment variables and import from there.
    
 2. Install requirements:
    CD into the working directory where the git repo was cloned to. 
    Prior to installing the requirements, you can optionally set up a virtual environment using:
      python3 -m venv ./venv
    If a virtual environment was created than activate it using:
      source venv/bin/activate
    Next install requirements from requirements.txt using:
      pip3 install -r requirements.txt
      
3. Run the script with:
  python3 security_cam.py
  
  
    
    
