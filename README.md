



auroc_reader.py:
main function, takes pdfs from pdfs folders and finds interesting regions (text boxes, etc) 
and outputs these regions into jpg_regions folder.

interesting regions is determined by wordlist.txt, i.e. if anything in the textbox matches 
wordlist.txt, then its interesting. 

to deploy, you need pytesseract and cv2.



count_pdfs.py: 
counts the number of interesting regions in a pdf and outputs them to count.csv.

