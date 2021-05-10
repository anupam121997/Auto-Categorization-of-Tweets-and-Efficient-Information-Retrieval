# -*- coding: utf-8 -*-
"""
Created on Fri May  7 14:39:19 2021

@author: kumar
"""
def gettext():
    import speech_recognition as sr
    import pyttsx3 
            
        
    r = sr.Recognizer() 
    
    with sr.Microphone() as source2:

        # wait for a second to let the recognizer
        # adjust the energy threshold based on
        # the surrounding noise level 
        r.adjust_for_ambient_noise(source2, duration=0.2)

        #listens for the user's input 
        audio2 = r.listen(source2)

        # Using ggogle to recognize audio
        MyText = r.recognize_google(audio2)
        MyText = MyText.lower()

        return MyText

