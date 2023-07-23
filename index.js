const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const app = express();
const upload = multer({ dest: 'uploads/' });
const pythonAppUrl = 'http://localhost:5000'; // Replace with your Python Flask app URL

// Serve the HTML page with the file upload form

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');
});

// Handle the file upload
app.post('/upload', upload.single('file'), async (req, res) => {
  try {
    const file = req.file;
    if (!file) {
      return res.status(400).json({ error: 'No file uploaded.' });
    }

    // Create form data
    const formData = new FormData();
    formData.append('file', fs.createReadStream(file.path), { filename: file.originalname });

    // Make a POST request to your Python Flask app to upload the file
    const response = await axios.post(`${pythonAppUrl}/upload`, formData, {
      headers: formData.getHeaders(),
    });

    // Handle the response from the Python app as needed
    console.log(response.data);

    return res.status(200).json({ message: 'File uploaded successfully.' });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: 'An error occurred while uploading the file.' });
  }
});

// Start the server
const port = 3000;
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
