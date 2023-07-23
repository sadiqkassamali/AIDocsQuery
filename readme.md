#Remember to make sure you have installed the necessary dependencies Tesseract OCR properly installed on your system. Also, ensure that you have the required language data downloaded.

Install python from devshell, you can use intelij

Then do:
##pip install -r requirements.txt
##pip install -r req.txt


>  I have kept in mind that this should work inside JPMC
> so sercivice.py is going to launc flash service on localhost. 
> You can run this Flask application, and it will expose an API endpoint at http://localhost:5000/extract where you can send a POST request with JSON data containing the file paths you want to extract text from. The response will be a JSON object containing the extracted text.

Here's an example of how you can use the API with cURL:
>> curl -X POST -H "Content-Type: application/json" -d '{"file_paths": ["path/to/file1.jpg", "path/to/file2.pdf"]}' http://localhost:5000/extract


#main.py runs server and everything else